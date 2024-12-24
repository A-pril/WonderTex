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

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
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


def trunc_rev_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

def configure_optimizer(optim_mod='adan', lr=0.001, iters=5000):
    if optim_mod == 'adan':
        from lib.optimizer import Adan
        optimizer = lambda model: Adan(
            model.get_params(5 * lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    else:  # adam
        optimizer = lambda model: torch.optim.Adam(model.get_params(5 * lr), betas=(0.9, 0.99), eps=1e-15)

    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.1 ** min(x / iters, 1))
    return scheduler, optimizer

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
    prompt_embeds = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0] # .to(torch.float16)

    return prompt_embeds


class RenderModel(nn.Module):
    def __init__(self, path_dir, resolution=512, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.texture_resolution = resolution
        if path_dir is not None:
            texture = torch.Tensor(np.array(Image.open(path_dir).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).to(device=self.device).unsqueeze(0) / 255.0
            if texture.shape[1] == 4:
                texture = texture[:, :3, :, :]
        else:
            texture = torch.ones((1, 3, self.texture_resolution, self.texture_resolution), dtype=torch.float32).to(device=self.device) * 0.5 # torch.Size([1, 3, 1024, 1024])
        
        self.texture = nn.Parameter(trunc_rev_sigmoid(texture), requires_grad=True) # torch.Size([1, 3, 1024, 1024])
    
    def get_params(self, lr):
        params = []
        
        params.append({'params': self.texture, 'lr': lr * 10})

        return params
    

class MeshModel(nn.Module):
    def __init__(self, mesh_dir, tex_dir=None, scale=1.35, auto_uv=False, resolution=512, device=torch.device('cpu')):
        # export PYTHONPATH="/data/xuqi/xuqi/code/WonderTex"
        from renderer.project import UVProjection as UVP

        super().__init__()
        self.device = device
        self.texture_resolution = resolution

        if tex_dir is not None:
            texture = torch.Tensor(np.array(Image.open(tex_dir).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).to(device=self.device).unsqueeze(0) / 255.0
            if texture.shape[1] == 4:
                texture = texture[:, :3, :, :]
        else:
            texture = torch.ones((1, 3, self.texture_resolution, self.texture_resolution), dtype=torch.float32).to(device=self.device) * 0.5 # torch.Size([1, 3, 1024, 1024])

        self.texture = nn.Parameter(trunc_rev_sigmoid(texture)) # torch.Size([1, 3, 1024, 1024])

        uvp_rgb = UVP(texture_size=1024, render_size=resolution, sampling_mode="nearest", channels=3, device=device)
        if mesh_dir.lower().endswith(".obj"):
            uvp_rgb.load_mesh(mesh_dir, scale_factor=1, autouv=auto_uv) # scale_factor no use, cause it's orthogonal projection
        elif mesh_dir.lower().endswith(".glb"):
            uvp_rgb.load_glb_mesh(mesh_dir, scale_factor=1, autouv=auto_uv)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."

        camera_poses = [(0,0)]
        render_scale = ((scale, scale, scale),)
        uvp_rgb.set_texture_map(self.texture.squeeze(0))
        uvp_rgb.set_cameras_and_render_settings(camera_poses, centers=None, camera_distance=4.0, scale=render_scale, cal_cosmap=False)
        
        self.mesh = uvp_rgb
    
    def get_params(self, lr):
        params = []
        
        params.append({'params': self.texture, 'lr': lr * 10})

        return params
        

class UpdateTex:
    def __init__(
            self, 
            output_dir, 
            rgb_path=None, norm_path=None, 
            mesh_path=None,
            mode="tex",
            optimizer=None,  # optimizer
            ema_decay=None,  # if use EMA, set the decay
            lr_scheduler=None,  # scheduler
            gpu=0,
            max_epoch=10000,
        ):
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu')
        self.max_epoch = max_epoch
        self.output_dir = output_dir
        self.rgb_path = rgb_path
        self.norm_path = norm_path
        self.fp16 = True
        
        # Set logger
        self.init_logger()
        logger.info(f"Saving to {self.output_dir}")

        # Set camera
        # self.camera_poses = opt.cameras # self.init_camera()
        # logger.info(f"Set camera pose: {self.camera_poses}")

        # Set mesh and render
        scale = 1.35

        # Set texture Model which is to be updated
        # TODO: set rgb_path to initialize texture
        # TODO: Not set rgb_path 
        self.mode = mode
        if mode == "tex":
            self.model = RenderModel(self.rgb_path, device=self.device)
        elif mode == "mesh":
            self.model = MeshModel(mesh_path, scale=scale, auto_uv=False, device=self.device) # tex_dir=self.rgb_path, 

        # Set pipeline
        # self.pipe = self.init_update_pipeline()
        self.init_update_pipeline()

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

        self.writer = SummaryWriter(join(self.output_dir, "run"))

        logger.info(f'# parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(join(self.output_dir, 'log.txt'), colorize=False, format=log_format)

    def init_update_pipeline(self, t_range=[0.02, 0.98], weighting_strategy='fantasia3d'):

        model_key = "/data/xuqi/xuqi/code/SyncMVD/pre-trained/stable-diffusion-v1-5"
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "/data/xuqi/xuqi/code/SyncMVD/pre-trained/stable-diffusion-v1-5", 
            safety_checker=None,
            torch_dtype=torch.float16
        ).to(self.device)

        # speed up diffusion process with faster scheduler and memory optimization
        # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler").to(self.device)
        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        self.weighting_strategy = weighting_strategy

        # pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_model_cpu_offload()
        # pipe = pipe.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        
        self.vae.requires_grad_(False)
        self.tokenizer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        logger.info("Finish HD pipeline load")

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """
        Args:
            prompt: str

        Returns:
            text_embeddings: torch.Tensor
        """
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            [prompt],
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0] # .to(torch.float16)

        return prompt_embeds

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        input_dtype = imgs.dtype
        imgs = 2.0 * imgs - 1.0
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(input_dtype)

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs
    
    def get_loss(
            self, img_rgb,
            prompt_embeds, 
            generator, guidance_scale,
            epoch=0
        ):

        with record_function("latent"):
            # 6. Prepare latent variables
            img_rgb = F.interpolate(
                img_rgb, (512, 512), mode="bilinear", align_corners=False
            )
            latents = self.encode_imgs(img_rgb)
        latents = torch.mean(latents, keepdim=True, dim=0)

        # 7. Prepare timesteps
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

       
        with torch.no_grad():
            # 6. Prepare latent variables
            # add noise
            # shape = latents.shape
            # noise = torch.randn(shape, generator=generator, device=self.device, dtype=prompt_embeds.dtype)
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            # no scale_model_input
            tt = torch.cat([t] * 2) # why?
            # 8. Denoising
            with record_function("UNet"):
                torch.cuda.empty_cache()
                noise_pred = self.unet(
                    latent_model_input, 
                    tt,
                    encoder_hidden_states=prompt_embeds,
                ).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        if self.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.weighting_strategy == "uniform":
            w = 1
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

        if epoch % 100 ==0:
            with torch.no_grad():
                pred_rgb_512 = self.decode_latents(latents)[0].permute(1, 2, 0)
                pred_rgb_512 = (pred_rgb_512 *255).to("cpu", torch.uint8).numpy()
                im = Image.fromarray(pred_rgb_512)
                im.save(join(self.output_dir, f"rgb_{epoch}.jpg"))

        return loss
        
    
    def update_step(
            self, prompt, negative_prompt, 
            norm_img, 
            w, h, 
            generator,
            guidance_scale = 100,
            epoch=0
        ):
        
        with record_function("prompt embeds"):
            # 3. Encode input prompt
            prompt_embeds = self.get_text_embeds(prompt=prompt) # get_text_embeds(self.pipe, prompt=prompt)
            negative_prompt_embeds = self.get_text_embeds(prompt=negative_prompt) # get_text_embeds(self.pipe, prompt=negative_prompt)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # 4. Prepare image
        # todo: need change position
        # tex_img = self.pipe.image_processor.preprocess(texModel.texture, height=h, width=w).to(dtype=torch.float16)
        norm_img = self.pipe.image_processor.preprocess(norm_img, height=h, width=w)

        if self.mode == "tex":
            # loss 1
            loss = self.get_loss(
                self.model.texture, 
                prompt_embeds, 
                generator, guidance_scale,
                epoch
            ).mean()
            # loss 2
            # loss += self.get_loss(
            #     torch.cat([norm_img, self.model.texture]),
            #     prompt_embeds, 
            #     generator, guidance_scale,
            # ).mean()  * 0.1
        
        # TODO: change to meshModel, and render first, then use the rendering to calculate loss
        elif self.mode == "mesh":
            self.model.mesh.set_texture_map(self.model.texture.squeeze(0))
            # render
            views = self.model.mesh.render_textured_views()
            views = [image[:-1,...] for image in views]
            front_view = views[0]
            # calculate loss
            # loss 1
            loss = self.get_loss(
                front_view, 
                prompt_embeds, 
                generator, guidance_scale,
            ).mean()
            # loss 2
            loss += self.get_loss(
                torch.cat([norm_img, front_view]),
                prompt_embeds, 
                generator, guidance_scale,
            ).mean()  * 0.1
            
        return loss
    

    def update(self, prompt, negative_prompt):
        w = h = 512
        # calculate uv_pos  #### hard!!!

        # load raw render -> image
        # rgb_img = Image.open(self.rgb_path).resize(size=(w, h), resample=Image.Resampling.BICUBIC)

        # load normal map texture -> image 
        norm_img = Image.open(self.norm_path).resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        norm_img = torch.Tensor(np.array(norm_img)).permute(2, 0, 1).to(self.device).unsqueeze(0) / 255.0
        if norm_img.shape[1] == 4:
            norm_img = norm_img[:, :3, :, :]

        # inference
        # control_img = [UVPos_img, tex_img]
        seed = 0
        generator = torch.Generator(self.device).manual_seed(seed)

        start_t = time.time()

        # 使用 torch.profiler 进行监控
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
            with_stack=True, profile_memory=True
        ) as prof:
            for epoch in tqdm(range(self.max_epoch)):
                self.model.train()
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    loss = self.update_step(
                        prompt=prompt, 
                        negative_prompt=negative_prompt, 
                        norm_img=norm_img, 
                        w=w, h=h, 
                        generator=generator,
                        epoch=epoch
                    )
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step()
                loss_val = loss.item()

                # TODO: record loss
                self.writer.add_scalar("train/loss", loss_val, epoch)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], epoch)


                if self.ema is not None:
                    self.ema.update()

                if epoch % 100 == 0:
                    logger.info(f"epoch:{epoch}, loss:{loss_val}")
                    # TODO: save texture now
                    tex_sig = torch.sigmoid(self.model.texture.detach().cpu()).numpy()
                    tex_sig = np.squeeze(tex_sig, axis=0)
                    tex_sig = (np.transpose(tex_sig, (1, 2, 0)) * 255).astype(np.uint8)

                    tex_img = self.model.texture.detach().cpu().numpy()
                    tex_img = np.squeeze(tex_img, axis=0)
                    tex_img = (np.transpose(tex_img, (1, 2, 0)) * 255).astype(np.uint8)

                    tex_sig = Image.fromarray(tex_sig)
                    tex_sig.save(join(self.output_dir, f"tex_sig_{epoch}.jpg"))

                    tex = Image.fromarray(tex_img)
                    tex.save(join(self.output_dir, f"texture_{epoch}.jpg"))
            
                prof.step()
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

        self.writer.close()
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


# python update_texture_sd.py 
# gpu --> gpu = ?
if __name__ == '__main__':
    gpu = 2
    max_epoch = 5000
    output_dir = "/data/xuqi/xuqi/code/WonderTex/exp_sd/new_lr"
    # mesh_path = "/data/xuqi/xuqi/code/textured.obj"
    render_path = "/data/xuqi/xuqi/code/WonderTex/exp_sd/render-1.png"
    norm_path = "/data/xuqi/xuqi/code/WonderTex/exp_sd/normal-1.png"
    prompt = "Blue and white pottery style lucky cat with intricate patterns." # "A camouflage military boot."
    negative_prompt = "oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect."
    # negative_prompt = "blur, low quality, noisy image, over-exposed, shadow"
    

    scheduler, optimizer = configure_optimizer(lr=0.0001, iters=max_epoch)
    # A. load initial texture
    # tex_update = UpdateTex(output_dir, render_path, norm_path, 
    #                        optimizer=optimizer, lr_scheduler=scheduler, gpu=gpu, max_epoch=max_epoch)
    # 
    # tex_update.update(prompt, negative_prompt)

    # B. use default color to initilize
    tex_update = UpdateTex(output_dir=output_dir, norm_path=norm_path, 
                        #    mode="mesh", mesh_path=mesh_path,
                        optimizer=optimizer, lr_scheduler=scheduler, gpu=gpu, max_epoch=max_epoch)
    
    tex_update.update(prompt, negative_prompt)