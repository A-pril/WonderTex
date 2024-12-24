from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
from tqdm import tqdm
from PIL import Image
import numpy as np
from loguru import logger
from os.path import join, isdir, abspath, dirname, basename, splitext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.pipelines.controlnet import MultiControlNetModel
# from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import prepare_control_image

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings,
    FoVOrthographicCameras,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    BlendParams,
    AmbientLights,
    HardPhongShader
)
from pytorch3d.structures import Meshes


# def configure_optimizer():
#     opt = cfg.training
#     if opt.optim == 'adan':
#         from lib.common.optimizer import Adan
# 
#         optimizer = lambda model: Adan(
#             model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
#     else:  # adam
#         optimizer = lambda model: torch.optim.Adam(model.get_params(5 * opt.lr), betas=(0.9, 0.99), eps=1e-15)
# 
#     scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.1 ** min(x / opt.iters, 1))
#     return scheduler, optimizer

@torch.no_grad()
def get_text_embeds(pipe, prompt):
    """
    Args:
        prompt: str

    Returns:
        text_embeddings: torch.Tensor
    """
    # Tokenize text and get embeddings
    text_input = pipe.tokenizer(
        [prompt],
        padding='max_length',
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt'
    )
    prompt_embeds = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0].to(torch.float16)

    return prompt_embeds


class TexModel(nn.Module):
    def __init__(self, path_dir, resolution=1024, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.texture_resolution = resolution

        if path_dir is not None:
            texture = torch.Tensor(np.array(Image.open(path_dir).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).to(device=self.device, dtype=torch.float16).unsqueeze(0) / 255.0
        else:
            texture = torch.ones(1, 3, self.texture_resolution, self.texture_resolution).to(device=self.device, dtype=torch.float16) # torch.Size([1, 3, 1024, 1024])
        
        self.texture = nn.Parameter(texture) # torch.Size([1, 3, 1024, 1024])
        




class UpdateTex:
    def __init__(
            self, 
            output_dir, 
            mesh_path, tex_path, norm_path, 
            optimizer=None,  # optimizer
            ema_decay=None,  # if use EMA, set the decay
            lr_scheduler=None,  # scheduler
            gpu=0,
        ):
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.tex_path = tex_path
        self.norm_path = norm_path
        self.fp16 = True
        
    
        # Set logger
        self.init_logger()
        logger.info(f"Saving to {self.output_dir}")

        # Set texture Model which is to be updated
        self.model = TexModel(self.tex_path, device=self.device)

        # Set camera
        # self.camera_poses = opt.cameras # self.init_camera()
        # logger.info(f"Set camera pose: {self.camera_poses}")

        # Set mesh and render
        scale = ((1.5, 1.5, 1.5),)
        self.mesh = self.init_mesh(mesh_path=mesh_path)
        
        # Set pipeline
        self.pipe = self.init_update_pipeline()



        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)
        
        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)


        if ema_decay is not None:
            from torch_ema import ExponentialMovingAverage
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        logger.info(f'# parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(join(self.output_dir, 'log.txt'), colorize=False, format=log_format)

    def init_mesh(self, mesh_path):
        mesh = load_objs_as_meshes([mesh_path], device=self.device)

        verts = mesh.verts_packed()   # (V, 3) [13955, 3]
        faces = mesh.faces_packed()   # (F, 3) [26256, 3]

        verts_uvs = mesh.textures.verts_uvs_list()[0] # (V, 2) [15538, 2]
        faces_uvs = mesh.textures.faces_uvs_list()[0]  # (F, 3) [26256, 3]

        new_verts = torch.cat((verts_uvs*2-1, torch.zeros_like(verts_uvs[:,:1])), dim=-1)

        uv_nums = verts_uvs.shape[0]
        face_nums = faces_uvs.shape[0]

        uvs_to_verts = torch.zeros(uv_nums, dtype=torch.int32) # torch.long
        for i in range(face_nums):
            for j in range(3):
                vert_id = faces[i][j]
                uv_id = faces_uvs[i][j]
                uvs_to_verts[uv_id] = vert_id

        verts_rgb = torch.zeros((uv_nums,3))
        verts_min = verts.min(dim=0, keepdim=True)[0]
        verts_max = verts.max(dim=0, keepdim=True)[0]
        verts_normalized = (verts - verts_min) / (verts_max - verts_min)

        verts_rgb = verts_normalized[uvs_to_verts].unsqueeze(0)
        textures = TexturesVertex(verts_features=verts_rgb)

        new_mesh = Meshes(verts=[new_verts], faces=[faces_uvs], textures=textures, device=self.device)

        return new_mesh
    
    def get_UVPos(self):
        elev = torch.FloatTensor([0])
        azim = torch.FloatTensor([0])
        R, T = look_at_view_transform(dist=4.0, elev=elev, azim=azim, at=(0,0,0), device=self.device)
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T,)[0]
        lights = AmbientLights(ambient_color=((1.0,)*3,), device=self.device)
        raster_settings = RasterizationSettings(image_size=1024, blur_radius=0.0, faces_per_pixel=1)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=cameras, lights=lights)
        )

        # 渲染新的 Mesh
        images = renderer(self.mesh)
        uv_pos = images[0, ..., :3].cpu().numpy()
        uv_pos = (uv_pos * 255).astype(np.uint8)
        uv_pos_img = Image.fromarray(uv_pos)

        uv_pos_img.save(join(self.output_dir, "UV_Pos.jpg"))

        return uv_pos_img

    def init_update_pipeline(self, t_range=[0.02, 0.98], weighting_strategy='fantasia3d'):
        controlnet_list = []

        # load control net and stable diffusion v1-5
        controlnet_UVpos = ControlNetModel.from_pretrained(
            "/data/xuqi/xuqi/code/SyncMVD/pre-trained/Paint3d_UVPos_Control", 
            torch_dtype=torch.float16
            )
        controlnet_list.append(controlnet_UVpos)

        controlnet_HD = ControlNetModel.from_pretrained(
        "/data/xuqi/xuqi/code/SyncMVD/pre-trained/control_v11f1e_sd15_tile", 
            torch_dtype=torch.float16
        )

        
        controlnet_list.append(controlnet_HD)

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "/data/xuqi/xuqi/code/SyncMVD/pre-trained/realisticVisionV13_v13", 
            controlnet=controlnet_list,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16
            )

        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        self.num_train_timesteps = pipe.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = pipe.scheduler.alphas_cumprod.to(self.device)  # for convenience

        # pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_model_cpu_offload()
        pipe = pipe.to(self.device)
        self.vae = AutoencoderKL.from_config(pipe.vae.config).to(self.device)

        logger.info("Finish HD pipeline load")


        return pipe

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
    
    def get_loss(
            self, img_rgb, control_img, 
            prompt_embeds, 
            controlnet, generator, guidance_scale,
            control_guidance_start, control_guidance_end,
            controlnet_conditioning_scale
        ):

        with record_function("latent"):
            # 6. Prepare latent variables
            latents = self.encode_imgs(img_rgb)
        latents = torch.mean(latents, keepdim=True, dim=0)

        # 7. Prepare timesteps
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # 6. Prepare latent variables
        # add noise
        shape = latents.shape
        noise = torch.randn(shape, generator=generator, device=self.device, dtype=prompt_embeds.dtype)
        latents_noisy = self.pipe.scheduler.add_noise(latents, noise, t)

        # TODO: 处理controlNet
        # 7.2 Create tensor stating which controlnets to keep
        controlnet_keep = [
            1.0 - float(t / self.max_step < s or (t + 1) / self.max_step > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]

        # 8. Denoising
        with torch.no_grad():
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # no scale_model_input
            tt = torch.cat([t] * 2) # why?

            control_model_input = latent_model_input
            controlnet_prompt_embeds = prompt_embeds

            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep)]

            with record_function("controlNet"):
                torch.cuda.empty_cache()
                down_block_res_samples, mid_block_res_sample = controlnet(
                        control_model_input,
                        tt, # need check
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_img,
                        conditioning_scale=cond_scale,
                        guess_mode=False,
                        return_dict=False,
                    )

            with record_function("UNet"):
                torch.cuda.empty_cache()
                noise_pred = self.pipe.unet(
                    latent_model_input, 
                    tt,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                ).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        if self.pipe.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * F.mse_loss(latents, (latents - grad).detach(), reduction="sum") / latents.shape[0]

        return loss
        
    
    def update_step(
            self, prompt, negative_prompt, 
            texModel, 
            norm_img, tex_img, UVPos_img, 
            w, h, 
            generator,
            guidance_scale = 100,
            controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0
        ):
        
        controlnet = self.pipe.controlnet

        # 2. Define call parameters
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = (
            mult * [control_guidance_start],
            mult * [control_guidance_end],
        )

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)
        
        with record_function("prompt embeds"):
            # 3. Encode input prompt
            prompt_embeds = get_text_embeds(self.pipe, prompt=prompt)
            negative_prompt_embeds = get_text_embeds(self.pipe, prompt=negative_prompt)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # 4. Prepare image
        # todo: need change position
        # tex_img = self.pipe.image_processor.preprocess(texModel.texture, height=h, width=w).to(dtype=torch.float16)
        norm_img = self.pipe.image_processor.preprocess(norm_img, height=h, width=w).to(dtype=torch.float16)
       
       # 5. Prepare controlnet_conditioning_image
        control_imgs = [texModel.texture, UVPos_img]
        controls = []
        for control_img_ in control_imgs:
            control_img_ = self.pipe.prepare_control_image(
                image=control_img_,
                width=w,
                height=h,
                batch_size=1,
                num_images_per_prompt=1,
                device=self.device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=True,
            )
            controls.append(control_img_)
        control_img = controls

        # loss 1
        loss = self.get_loss(
            texModel.texture, 
            control_img, 
            prompt_embeds, 
            controlnet, generator, guidance_scale,
            control_guidance_start, control_guidance_end,
            controlnet_conditioning_scale
        )
        # loss 2
        loss += self.get_loss(
            torch.cat([norm_img, texModel.texture.detach()]),
            control_img, 
            prompt_embeds, 
            controlnet, generator, guidance_scale,
            control_guidance_start, control_guidance_end,
            controlnet_conditioning_scale
        )

        return texModel, loss
    

    def update(self, prompt, negative_prompt):
        w = h = 1024
        # calculate uv_pos  #### hard!!!
        # load uv_pos      ->         control imge
        UVPos_img = self.get_UVPos().resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        # load raw texture -> image & control imge
        tex_img = Image.open(self.tex_path).resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        # load normal map texture -> image 
        norm_img = Image.open(self.norm_path).resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        norm_img = torch.Tensor(np.array(norm_img)).permute(2, 0, 1).to(self.device).unsqueeze(0) / 255.0
        # inference
        control_img = [UVPos_img, tex_img]
        seed = 0
        generator = torch.Generator(self.device).manual_seed(seed)

        start_t = time.time()
        max_epoch = 100

        # 使用 torch.profiler 进行监控
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
            with_stack=True, profile_memory=True
        ) as prof:
            for epoch in range(max_epoch):
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model, loss = self.update_step(
                        prompt=prompt, 
                        negative_prompt=negative_prompt,
                        texModel=self.model, 
                        tex_img=tex_img,
                        norm_img=norm_img, 
                        UVPos_img=UVPos_img, 
                        w=w, h=h, 
                        generator=generator,
                    )
            
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step()
                loss_val = loss.item()

                # TODO: record loss

                if self.ema is not None:
                    self.ema.update()
                
                self.stats["loss"].append(loss_val)

                if epoch % 2 == 0:
                    # TODO: save texture now
                    tex_img = self.model.texture.detach().cpu().numpy()
                    tex_img = np.squeeze(tex_img, axis=0)
                    tex_img = (np.transpose(tex_img, (1, 2, 0)) * 255).astype(np.uint8)

                    tex = Image.fromarray(tex_img)
                    tex.save(join(self.output_dir, f"texture_{epoch}.jpg"))
                prof.step()
            
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

        end_t = time.time()
        logger.info(f"training takes {(end_t - start_t) / 60:.4f} minutes.")


        # res_image = self.pipe(prompt,
        #                       negative_prompt=negative_prompt,
        #                       image=tex_img,
        #                       control_image=control_img,
        #                       height=h,
        #                       width=w,
        #                     #   num_images_per_prompt=config.num_images_per_prompt,
        #                     #   guidance_scale=config.guidance_scale,  # 3.0
        #                       num_inference_steps=20,
        #                     #   strength=config.denoising_strength,  # 0.75
        #                       generator=generator).images[0] 
        # res_image.save(join(self.output_dir, "UV_HD_tex.jpg"))


if __name__ == '__main__':
    gpu = 0
    output_dir = "/data/xuqi/xuqi/code/WonderTex/exp_stage2"
    mesh_path = "/data/xuqi/xuqi/code/WonderTex/experiment/lucky-cat/Tex_04Jul2024-174221/results/textured.obj"
    tex_path = "/data/xuqi/xuqi/code/WonderTex/experiment/lucky-cat/Tex_04Jul2024-174221/results/textured.png"
    norm_path = "/data/xuqi/xuqi/code/WonderTex/normal_map.jpg"
    prompt = "UV map, Blue and white pottery style lucky cat with intricate patterns, high quality, best quality"
    negative_prompt = "blur, low quality, noisy image, over-exposed, shadow"
    

    tex_update = UpdateTex(output_dir, mesh_path, tex_path, norm_path, gpu=gpu)
    
    tex_update.update(prompt, negative_prompt)