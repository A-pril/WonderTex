import os
from os.path import join, isdir, abspath, dirname, basename, splitext
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from loguru import logger
import torch
import torch.nn.functional as F
from renderer.project import UVProjection as UVP
from utils import save_img, split_grid_img, encode_latents, latent_preview, numpy_to_pil, decode_latents, \
                    get_rgb_texture, split_grids, make_grids
from torchvision.utils import make_grid
from step import step_tex


from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel



# Background colors
color_constants = {"black": [-1, -1, -1], "white": [1, 1, 1], "maroon": [0, -1, -1],
			"red": [1, -1, -1], "olive": [0, 0, -1], "yellow": [1, 1, -1],
			"green": [-1, 0, -1], "lime": [-1 ,1, -1], "teal": [-1, 0, 0],
			"aqua": [-1, 1, 1], "navy": [-1, -1, 0], "blue": [-1, -1, 1],
			"purple": [0, -1 , 0], "fuchsia": [1, -1, 1]}
color_names = list(color_constants.keys())


# Revert time 0 background to time t to composite with time t foreground
@torch.no_grad()
def composite_rendered_view(scheduler, backgrounds, foregrounds, masks, t):
	composited_images = []
	for i, (background, foreground, mask) in enumerate(zip(backgrounds, foregrounds, masks)):
		if t > 0:
			alphas_cumprod = scheduler.alphas_cumprod[t]
			noise = torch.normal(0, 1, background.shape, device=background.device)
			background = (1-alphas_cumprod) * noise + alphas_cumprod * background
		composited = foreground * mask + background * (1-mask)
		composited_images.append(composited)
	composited_tensor = torch.stack(composited_images)
	return composited_tensor




class StableDiffusionSyncControlNetPipeline(StableDiffusionControlNetPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, 
            controlnet, scheduler, safety_checker, 
            feature_extractor, image_encoder, requires_safety_checker
        )

        self.scheduler = DDPMScheduler.from_config(self.scheduler.config)
        self.model_cpu_offload_seq = "vae->text_encoder->unet->vae"
        self.enable_model_cpu_offload()
        self.enable_vae_slicing()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def init_camera(self):
        elevs = [0, 30] # [0] # 0, 45, -15, 
        camera_azims = [0, 90, 180, 270]
        camera_poses = []
        for elev in elevs:
            for azim in camera_azims:
                if elev == 0:
                    camera_poses.append((0, azim))
                elif elev == 45:
                    camera_poses.append((0, azim+45))
                else:
                    camera_poses.append((elev, azim+45))
        logger.info("Finish camera pose")

        self.view_dirs = ['flat_view', 
                #   'flat_view_var', 
                'top_view', 'bottom_view'
                ]
        
        self.view_prompts = [
            "4 orthogonal flat views of ", 
            # "4 orthogonal flat side views of ", 
            "4 orthogonal top side views of ", 
            "4 orthogonal bottom side views of "
        ]
            
        return camera_poses
    
    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(join(self.output_dir, 'log.txt'), colorize=False, format=log_format)
    
    def initialize_pipeline(
			self,
            device=None,
			mesh_path=None,
			mesh_autouv=None,
            mesh_scale=None,
			latent_size=None,
			render_rgb_size=None,
			texture_size=None,
			texture_rgb_size=None,

			output_dir=None,
            grid=True,
		):
        # Make output dir
        self.output_dir = output_dir
        self.result_dir = f"{output_dir}/results"
        self.intermediate_dir = f"{output_dir}/intermediate"
        # Set logger
        self.init_logger()

        dirs = [output_dir, self.result_dir, self.intermediate_dir]
        for dir_ in dirs:
            if not os.path.isdir(dir_):
                os.mkdir(dir_)
        
        logger.info(f"Saving to {output_dir}")

        # Set camera
        self.camera_poses = self.init_camera()

        # Set up pytorch3D for projection between screen space and UV space
		# uvp is for latent and uvp_rgb for rgb color
        # Set mesh and render
        scale = ((mesh_scale, mesh_scale, mesh_scale),)

        self.uvp = UVP(texture_size=texture_size, render_size=latent_size, sampling_mode="nearest", channels=4, device=device)
        if mesh_path.lower().endswith(".obj"):
            self.uvp.load_mesh(mesh_path, scale_factor=1, autouv=mesh_autouv)
        elif mesh_path.lower().endswith(".glb"):
            self.uvp.load_glb_mesh(mesh_path, scale_factor=1, autouv=mesh_autouv)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."
        self.uvp.set_cameras_and_render_settings(self.camera_poses, centers=None, camera_distance=4.0, scale=scale, cal_cosmap=True)


        self.uvp_rgb = UVP(texture_size=texture_rgb_size, render_size=render_rgb_size, sampling_mode="nearest", channels=3, device=device)
        self.uvp_rgb.mesh = self.uvp.mesh.clone()
        self.uvp_rgb.set_cameras_and_render_settings(self.camera_poses, centers=None, camera_distance=4.0, scale=scale, cal_cosmap=True)
        # calculate cosine similarity between view->surface and surface normal.
        _,_,_,cos_maps,_, _ = self.uvp_rgb.render_geometry()
        # Bake screen-space cosine weights to UV space
        self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)

        self.uvp.to("cpu")
        self.uvp_rgb.to("cpu")

        color_images = torch.FloatTensor([color_constants[name] for name in color_names]).reshape(-1,3,1,1).to(dtype=self.text_encoder.dtype, device=device)
        # latent_size = 64
        if grid:
            size = latent_size*2*8
        else:
            size = latent_size*8
        color_images = torch.ones(
            (1, 1, size, size), 
            device=device, 
            dtype=self.text_encoder.dtype
        ) * color_images
        # set value = [0,1]
        color_images *= ((0.5*color_images)+0.5)
        color_latents = encode_latents(self.vae, color_images)

        # 准备好了每种颜色的隐纹理图
        self.color_latents = {color[0]:color[1] for color in zip(color_names, [latent for latent in color_latents])}
        self.vae = self.vae.to("cpu")

        logger.info("Done Initialization")

    # Append directions to the prompts
    def prepare_directional_prompt(self, prompt, negative_prompt):
        directional_prompt = [prompt  for i in range(self.batch_size)] # [f"{v}" + prompt   for v in self.view_prompts]
        negative_prompt = [negative_prompt for i in range(self.batch_size) ]
        return directional_prompt, negative_prompt

    def get_cond_images(self, cond_type, grid=True, device=None):
        # verts, normals, depths, cos_angles, texels, fragments
        self.uvp_rgb.to(device)
        _, normals, depths, _, _, _ = self.uvp_rgb.render_geometry() # depths: torch.Size([N, 512, 512, 2])

        # (N, 1, H, W) alpha channel
        masks = normals[...,3][:,None,...]

        # normals_transforms = Compose([
        #     Resize((output_size,)*2, interpolation=InterpolationMode.BILINEAR, antialias=True),
        #     GaussianBlur(blur_filter, blur_filter//3+1)])
        
        if cond_type == "depth":
            conds = self.uvp_rgb.decode_normalized_depth(depths) # torch.Size([4*4, H, W, 3])
            # conds = normals_transforms(depths)
        elif cond_type == "normal":
            conds = self.uvp_rgb.decode_view_normal(normals) # *2 - 1
            # conds = normals_transforms(normals)

        if not grid:
            for i, cond in enumerate(conds):
                save_img(cond, join(self.intermediate_dir, f"{cond_type}_{i}.jpg"))
                # raise ValueError("You need change here to fit the subsequent functions")
            return conds.permute(0, 3, 1, 2), masks
        else:
            cond_lists = [cond.permute(2,0,1) for cond in conds]
            mask_lists = [mask for mask in masks]
            cond_grids = []
            mask_grids = []
            for i in range(self.batch_size): # the num of grid
                cond = torch.stack(cond_lists[i*4:(i+1)*4])
                mask = torch.stack(mask_lists[i*4:(i+1)*4])
                cond_grid = make_grid(cond, nrow=2, padding=0).permute(1,2,0)
                mask_grid = make_grid(mask, nrow=2, padding=0).permute(1,2,0)
                save_img(cond_grid, join(self.intermediate_dir, f"{cond_type}_{self.view_dirs[i]}.jpg"))
                # save_img(mask_grid, join(self.intermediate_dir, f"mask_{self.view_dirs[i]}.jpg"))
                cond_grids.append(cond_grid)
                mask_grids.append(mask_grid)
        
            return torch.stack(cond_grids).permute(0, 3, 1, 2), torch.stack(mask_grids).permute(0, 3, 1, 2)


    @torch.no_grad()
    def __call__(
        self,
        device: int = 0,
        prompt: str = None,
        height: Optional[int] = None,
		width: Optional[int] = None,
        num_inference_steps: int = 20,  
        guidance_scale: float = 7.5,
        negative_prompt: str = None,
        
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
		callback_steps: int = 1,
        
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_guess_mode: bool = False,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.7,
        controlnet_conditioning_end_scale: Union[float, List[float]] = 0.9,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        multiview_diffusion_end=0.8,
        shuffle_background_change=0.4,
		shuffle_background_end=0.99,

        cond_type: str = None,
        grid_diffusion: bool = True,
        mesh_path: str = None,
        mesh_autouv = False,
        mesh_scale = 1.0, 
        
        render_rgb_size = 512,
        render_latent_size = 64,
        texture_size = 1536,
        texture_rgb_size = 1024,

        logging_config = None
    ):
        device = self._execution_device # torch.device(f"cuda:{device}" if torch.cuda.is_available() else 'cpu')

        # Setup pipeline settings
        self.initialize_pipeline(
            device=device,
            mesh_scale=mesh_scale,
            mesh_path=mesh_path,
            mesh_autouv=mesh_autouv,
            
            latent_size=render_latent_size, # latent view size = 64
            render_rgb_size=render_rgb_size,
            texture_size=texture_size,
            texture_rgb_size=texture_rgb_size,

            output_dir=logging_config["output_dir"],
            grid=grid_diffusion,
        )

        num_timesteps = self.scheduler.config.num_train_timesteps
        initial_controlnet_conditioning_scale = controlnet_conditioning_scale
        log_interval = logging_config.get("log_interval", 10)
		# default： True
        view_fast_preview = logging_config.get("view_fast_preview", True)
        tex_fast_preview = logging_config.get("tex_fast_preview", True)

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            # TODO: maybe need change
            # mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            # control_guidance_start, control_guidance_end = (
            #     mult * [control_guidance_start],
            #     mult * [control_guidance_end],
            # )
            mult = 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start], 
                mult * [control_guidance_end],
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            image=torch.zeros((1,3,height,width), device=device), # image,
            callback_steps=callback_steps,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
		    negative_prompt_embeds=None,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )

        # 2. Define call parameters        
        # TODO: change batch_size
        if grid_diffusion:
            batch_size = len(self.uvp.cameras) // 4 # len(self.uvp.cameras)
        else:
            batch_size = len(self.uvp.cameras)
        self.batch_size = batch_size

        do_classifier_free_guidance = guidance_scale > 1.0

        # if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        #     controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = controlnet_guess_mode or global_pool_conditions

        # 3. Encode input prompt
        prompt, negative_prompt = self.prepare_directional_prompt(prompt, negative_prompt)

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
			negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        # if self.do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        negative_prompt_embeds, positive_prompt_embeds = torch.chunk(prompt_embeds, 2)
        
        # 4. Prepare image
        cond_imgs, masks = self.get_cond_images(cond_type=cond_type, grid=grid_diffusion, device=device) #[4, H, W, c]
        cond_imgs = cond_imgs.type(positive_prompt_embeds.dtype) # [4, 1024, 1024, 3]
    
        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        if grid_diffusion:
            h = height * 2
            w = width * 2
        else:
            h = height
            w = width
        latents = self.prepare_latents(
            batch_size, # 4
            num_channels_latents,
            h, # height*2, # 512*2
            w, # width*2,
            positive_prompt_embeds.dtype,
            device,
            generator,
            None,
        )


        # torch.Size([4, 1536, 1536])
        latent_tex = self.uvp.set_noise_texture()
        # [torch.Size([5, 64, 64]) * 16]
        noise_views = self.uvp.render_textured_views()
        # TODO: need make grid
        if grid_diffusion:
            noise_grids = []
            for i in range(self.batch_size): # the num of grid
                noises = torch.stack(noise_views[i*4:(i+1)*4])
                noise_grid = make_grid(noises, nrow=2, padding=0).permute(1,2,0)
                noise_grids.append(noise_grid.permute(2, 0, 1))
            noise_views = torch.stack(noise_grids) # [4, 5, 128, 128]

        foregrounds = [view[:-1] for view in noise_views]
        masks = [view[-1:] for view in noise_views]
        # make all rendered view to a grid 
        # torch.Size([4, 4, 128, 128])
        composited_tensor = composite_rendered_view(self.scheduler, latents, foregrounds, masks, timesteps[0]+1)
        latents = composited_tensor.type(latents.dtype)

        self.uvp.to("cpu")

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        intermediate_results = []
        background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses))]
        dbres_sizes_list = []
        mbres_size_list = []
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latent_model_input = self.scheduler.scale_model_input(latents, t)

                '''
                    Use groups to manage prompt and results
                    Make sure negative and positive prompt does not perform attention together
                '''
                prompt_embeds_groups = {"positive": positive_prompt_embeds}
                result_groups = {}
                if do_classifier_free_guidance:
                    prompt_embeds_groups["negative"] = negative_prompt_embeds

                for prompt_tag, prompt_embeds in prompt_embeds_groups.items():
                    if prompt_tag == "positive" or not guess_mode:
                        # controlnet(s) inference
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds

                        if isinstance(controlnet_keep[i], list):
                            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                        else:
                            controlnet_cond_scale = controlnet_conditioning_scale
                            if isinstance(controlnet_cond_scale, list):
                                controlnet_cond_scale = controlnet_cond_scale[0]
                            cond_scale = controlnet_cond_scale * controlnet_keep[i]

                        # Split into micro-batches according to group meta info
                        # Ignore this feature for now
                        down_block_res_samples_list = []
                        mid_block_res_sample_list = []

                        # N = 4 
                        # [torch.Size([N, 4, 128, 128])] latent views
                        model_input_batches = [control_model_input]
                        # [torch.Size([N, 77, 768])] prompt embedding
                        prompt_embeds_batches = [controlnet_prompt_embeds]
                        # [torch.Size([N, 1024, 1024, 3])] condition image(render rgb)
                        conditioning_images_batches = [cond_imgs]

                        for model_input_batch ,prompt_embeds_batch, conditioning_images_batch \
                            in zip (model_input_batches, prompt_embeds_batches, conditioning_images_batches):

                            down_block_res_samples, mid_block_res_sample = self.controlnet(
                                model_input_batch,
                                t,
                                encoder_hidden_states=prompt_embeds_batch,
                                controlnet_cond=conditioning_images_batch,
                                conditioning_scale=cond_scale,
                                guess_mode=guess_mode,
                                return_dict=False,
                            )
                            
                            down_block_res_samples_list.append(down_block_res_samples) # [[torch.Size([10, 320, 96, 96]),*12]]
                            mid_block_res_sample_list.append(mid_block_res_sample) # [[torch.Size([1280, 12, 12]),*10]]

                        ''' For the ith element of down_block_res_samples, concat the ith element of all mini-batch result '''
                        model_input_batches = prompt_embeds_batches = conditioning_images_batches = None

                        if guess_mode:
                            for dbres in down_block_res_samples_list:
                                dbres_sizes = []
                                for res in dbres:
                                    dbres_sizes.append(res.shape)
                                dbres_sizes_list.append(dbres_sizes)

                            for mbres in mid_block_res_sample_list:
                                mbres_size_list.append(mbres.shape)

                    else:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        # We copy the tensor shapes from a conditional batch
                        down_block_res_samples_list = []
                        mid_block_res_sample_list = []
                        for dbres_sizes in dbres_sizes_list:
                            down_block_res_samples_list.append([torch.zeros(shape, device=device, dtype=latents.dtype) for shape in dbres_sizes])
                        for mbres in mbres_size_list:
                            mid_block_res_sample_list.append(torch.zeros(mbres, device=device, dtype=latents.dtype))
                        dbres_sizes_list = []
                        mbres_size_list = []

                    '''
						predict the noise residual, split into mini-batches.
						Downblock res samples has n samples, we split each sample into m batches
						and re group them into m lists of n mini batch samples.
					
					'''
                    noise_pred_list = []
                    # N = 4 
                    # [torch.Size([N, 4, 128, 128])] latent views
                    model_input_batches = [latent_model_input]
                    # [torch.Size([N, 77, 768])] prompt embedding
                    prompt_embeds_batches = [prompt_embeds]

                    for model_input_batch, prompt_embeds_batch, down_block_res_samples_batch, mid_block_res_sample_batch \
                        in zip(model_input_batches, prompt_embeds_batches, down_block_res_samples_list, mid_block_res_sample_list):

                        # TODO: change attention processor

                        # predict the noise residual
                        noise_pred = self.unet(
                            model_input_batch, # torch.Size([4, 4, 128, 128])
                            t,
                            encoder_hidden_states=prompt_embeds_batch, # torch.Size([1, 77, 768])
                            cross_attention_kwargs=cross_attention_kwargs, # None
                            down_block_additional_residuals=down_block_res_samples_batch,
                            mid_block_additional_residual=mid_block_res_sample_batch,
                            return_dict=False,
                        )[0]
                        noise_pred_list.append(noise_pred)

                    # noise_pred_list = [noise_pred[i] for i, noise_pred in enumerate(noise_pred_list)]
                    noise_pred = torch.cat(noise_pred_list, dim=0)                   
                    down_block_res_samples_list = None
                    mid_block_res_sample_list = None
                    noise_pred_list = None
                    model_input_batches = prompt_embeds_batches = down_block_res_samples_batches = mid_block_res_sample_batches = None

                    result_groups[prompt_tag] = noise_pred

                positive_noise_pred = result_groups["positive"]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred = result_groups["negative"] + guidance_scale * (positive_noise_pred - result_groups["negative"])

                self.uvp.to(device)
				# compute the previous noisy sample x_t -> x_t-1
				# Multi-View step or individual step
                if i%2==0 and i < multiview_diffusion_end*num_inference_steps: # t > (1-multiview_diffusion_end)*num_timesteps :
                    step_results = step_tex(
                        scheduler=self.scheduler, 
                        uvp=self.uvp, 
                        model_output=noise_pred,
                        grid=grid_diffusion, 
                        timestep=t, 
                        sample=latents,  # torch.Size([4, 4, 128, 128])
                        texture=latent_tex,
                        return_dict=True, 
                        main_views=[], 
                        exp=0,
                        **extra_step_kwargs
                    )
                    pred_original_sample = step_results["pred_original_sample"]  # x_{0}
                    latents = step_results["prev_sample"] # x_{t-1}
                    latent_tex = step_results["prev_tex"] # texture_{t-1}
                  
                    # Composit latent foreground with random color background
                    background_latents = [self.color_latents[color] for color in background_colors]
                    composited_tensor = composite_rendered_view(self.scheduler, background_latents, latents, masks, t)
                    latents = composited_tensor.type(latents.dtype)

                    intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))
                else:
                    # compute the previous noisy sample x_t -> x_t-1
                    step_results = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                    # TODO: need split to grid

                    pred_original_sample = step_results["pred_original_sample"]
                    latents = step_results["prev_sample"]
                    latent_tex = None

                    intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))

                del noise_pred, result_groups

                # Update pipeline settings after one step:
				# 1. Annealing ControlNet scale
                if (1-t/num_timesteps) < control_guidance_start[0]:
                    controlnet_conditioning_scale = initial_controlnet_conditioning_scale
                elif (1-t/num_timesteps) > control_guidance_end[0]:
                    controlnet_conditioning_scale = controlnet_conditioning_end_scale
                else:
                    alpha = ((1-t/num_timesteps) - control_guidance_start[0]) / (control_guidance_end[0] - control_guidance_start[0])
                    controlnet_conditioning_scale = alpha * initial_controlnet_conditioning_scale + (1-alpha) * controlnet_conditioning_end_scale

                # 2. Shuffle background colors;
                # Trick: only black and white used after certain timestep
                if (1-t/num_timesteps) < shuffle_background_change:
                    background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses))]
                elif (1-t/num_timesteps) < shuffle_background_end:
                    background_colors = [random.choice(["black","white"]) for i in range(len(self.camera_poses))]
                else:
                    background_colors = background_colors


                # Logging at "log_interval" intervals and last step
				# Choose to uses color approximation or vae decoding
                if i % log_interval == 0 or t == 1:
                    if view_fast_preview: # default: True
                        decoded_results = []
                        for latent_images in intermediate_results[-1]:
                            images = latent_preview(latent_images.to(device)) # transform latent texture to rgb
                            # TODO : change method
                            images = np.concatenate([img for img in images], axis=1)
                            decoded_results.append(images)
                        result_image = np.concatenate(decoded_results, axis=0)
                        # TODO: change method
                        numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.jpg") # [256, N*128, 3]
                    else:
                        decoded_results = []
                        for latent_images in intermediate_results[-1]:
                            images = decode_latents(self.vae, latent_images.to(device))

                            images = np.concatenate([img for img in images], axis=1)

                            decoded_results.append(images)
                        result_image = np.concatenate(decoded_results, axis=0)
                        numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.jpg")

                    if i < multiview_diffusion_end*num_inference_steps and i%2==0: # TODO : need check
                        if tex_fast_preview:
                            tex = latent_tex.clone()
                            texture_color = latent_preview(tex[None, ...])
                            numpy_to_pil(texture_color)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.jpg")
                        else:
                            self.uvp_rgb.to(device)
                            # decode render view,  bake to uv texture
                            result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, self.uvp_rgb, pred_original_sample)
                            numpy_to_pil(result_tex_rgb_output)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.png")
                            self.uvp_rgb.to("cpu")
                
                self.uvp.to("cpu")

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                # Signal the program to skip or end
                import select
                import sys
                if select.select([sys.stdin],[],[],0)[0]:
                    userInput = sys.stdin.readline().strip()
                    if userInput == "skip":
                        return None
                    elif userInput == "end":
                        exit(0)

        self.uvp.to(device)
        self.uvp_rgb.to(device)
        latent_views = [view for view in latents]
        if grid_diffusion:
            # TODO: split into 4 parts
            latent_views = split_grids(latent_views)
            latent_views = torch.stack(latent_views, dim=0)
            result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, self.uvp_rgb, latent_views)
        else:
            result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, self.uvp_rgb, latents)
        self.uvp.save_mesh(f"{self.result_dir}/textured.obj", result_tex_rgb.permute(1,2,0))


        self.uvp_rgb.set_texture_map(result_tex_rgb)
        textured_views = self.uvp_rgb.render_textured_views()
        textured_views_rgb = torch.cat(textured_views, axis=-1)[:-1,...]
        textured_views_rgb = textured_views_rgb.permute(1,2,0).cpu().numpy()[None,...]
        v = numpy_to_pil(textured_views_rgb)[0]
        v.save(f"{self.result_dir}/textured_views_rgb.jpg")

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        self.uvp.to("cpu")
        self.uvp_rgb.to("cpu")

        return result_tex_rgb, textured_views, v