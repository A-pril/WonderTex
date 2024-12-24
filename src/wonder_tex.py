import os
from os.path import join, abspath, dirname, basename, splitext
from datetime import datetime
from shutil import copy
from loguru import logger
from diffusers import (
    UNet2DConditionModel,
    StableDiffusionPipeline, 
    StableDiffusionInpaintPipeline, 
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel
    )
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
# Need: export PYTHONPATH=/path/to/WonderTex
from renderer.project import UVProjection as UVP
from utils import save_img, split_grid_img, make_inpaint_condition, fill_image, get_view
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm


class WonderTex:
    def __init__(self, opt):
        self.device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else 'cpu')
        self.diffusion_device = torch.device(f"cuda:{opt.diffusion_gpu}" if torch.cuda.is_available() else 'cpu')

        self.sd_model = opt.sd_model
        self.mvd_model = opt.mvd_model
        self.unet_step = opt.unet_step
        self.dep_ControlNet = opt.dep_ControlNet
        self.norm_ControlNet = opt.norm_ControlNet
        self.hd_ControlNet = opt.hd_ControlNet
        self.ipait_ControlNet = opt.ipait_ControlNet

        # get mesh path
        if opt.mesh_config_relative:
            mesh_path = join(os.path.dirname(opt.config), opt.mesh)
        else:
            mesh_path = abspath(opt.mesh)
        
        # set output path
        if opt.output:
            output_root = abspath(opt.output)
        else:
            output_root = dirname(opt.config)

        output_name_components = []
        # prefix
        if opt.prefix and opt.prefix != "":
            output_name_components.append(opt.prefix)
        # mesh name
        if opt.use_mesh_name:
            mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
            output_name_components.append(mesh_name)
        # add time-stamp
        if opt.timeformat != "":
            output_name_components.append(datetime.now().strftime(opt.timeformat))

        # join the name
        output_name = "_".join(output_name_components)
        # set output_dir dir+name
        output_dir = join(output_root, output_name)
        self.output_dir = output_dir
        self.result_dir = f"{output_dir}/results"
        self.intermediate_dir = f"{output_dir}/intermediate"
        # Set logger
        self.init_logger()
        
        dirs = [self.result_dir, self.intermediate_dir]
        for dir_ in dirs:
            if not os.path.isdir(dir_):
                os.mkdir(dir_)      
            else:
                logger.info(f"Results exist in the output directory, use time string to avoid name collision.")
                exit(0)

        logger.info(f"Saving to {output_dir}")

        copy(opt.config, join(output_dir, "config.yaml"))
        
        # Set camera
        self.camera_poses = self.init_camera(opt.init_elev)
        logger.info(f"Set camera pose: {self.camera_poses}")
        # Set mesh and render
        self.scale = ((opt.scale, opt.scale, opt.scale),)
        self.uvp_rgb = self.init_mesh(mesh_path=mesh_path, texture_size=opt.rgb_tex_size, render_size=opt.rgb_view_size, scale=self.scale, auto_uv=opt.mesh_autouv)
        # calculate all cameras iteratively
        self.get_all_cameras(diff_angle=opt.diff_angle, min_elev=opt.min_elev, max_elev=opt.max_elev, 
                             eps=opt.eps, samples=opt.samples, scale=self.scale)
        # Set pipeline
        self.cond_type = opt.cond_type
        self.grid_diffusion = opt.grid_diffusion
        self.seed = opt.seed
        self.pipe = self.init_pipeline(cond_type=opt.cond_type, grid=opt.grid_diffusion)

        
    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(join(self.output_dir, 'log.txt'), colorize=False, format=log_format)

    def init_camera(self, elev):
        # elevs = [0, 60, -15] # [0, 45, 30, -15]
        camera_azims = [0, 90, 180, 270]
        camera_poses = []
        self.view_dirs = ['flat_view', 'top_view', 'bottom_view'] # ['flat_view', 'flat_view_var', 'top_view', 'bottom_view']
        # for elev in elevs:
        for azim in camera_azims:
            camera_poses.append((elev, azim))
        logger.info("Finish camera pose")
        
        # camera_poses.append((0, 45))
        # camera_poses.append((0, 315))
        # camera_poses.append((30, 180))
        # camera_poses.append((-60, 180))

        return camera_poses

    def init_mesh(self, mesh_path, texture_size, render_size, scale, auto_uv):
        # Set up pytorch3D for projection between screen space and UV space
		# uvp is for latent and uvp_rgb for rgb color
        uvp_rgb = UVP(texture_size=texture_size, render_size=render_size, sampling_mode="nearest", channels=3, device=self.device)
        if mesh_path.lower().endswith(".obj"):
            uvp_rgb.load_mesh(mesh_path, scale_factor=1, autouv=auto_uv) # scale_factor no use, cause it's orthogonal projection
        elif mesh_path.lower().endswith(".glb"):
            uvp_rgb.load_glb_mesh(mesh_path, scale_factor=1, autouv=auto_uv)
        elif mesh_path.lower().endswith(".gltf"):
            uvp_rgb.load_gltf_mesh(mesh_path, scale_factor=1, autouv=auto_uv)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."
        
        uvp_rgb.set_cameras_and_render_settings(self.camera_poses, centers=None, camera_distance=4.0, scale=scale)
        # Save some VRAM
        # del _, cos_maps
        # uvp_rgb.to("cpu")
        logger.info("Finish mesh load")

        return uvp_rgb

    def get_all_cameras(self, diff_angle, min_elev, max_elev, eps, samples, 
					 	centers=None, camera_distance=2.7, scale=None):
        self.uvp_rgb.get_all_cameras(eps=eps, samples=samples,
                                      diff_angle=diff_angle, min_elev=min_elev, max_elev=max_elev,
                                        centers=centers, camera_distance=camera_distance, scale=scale)
        formatted_camera = [(round(x, 1), round(y, 1)) for x, y in self.uvp_rgb.camera_poses]
        logger.info(f"All camera poses: {formatted_camera}")
        self.camera_poses = self.uvp_rgb.camera_poses

    def get_cond_images(self, cond_type):
        # verts, normals, depths, cos_angles, texels, fragments
        self.uvp_rgb.to(self.device)
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

        cond_lists = [cond.permute(2,0,1) for cond in conds]
        mask_lists = [mask.repeat(3,1,1) for mask in masks]
        cond_grids = []
        mask_grids = []

        cond = torch.stack(cond_lists[0:4])
        mask = torch.stack(mask_lists[0:4])
        cond_grid = make_grid(cond, nrow=2, padding=0).permute(1,2,0)
        mask_grid = make_grid(mask, nrow=2, padding=0).permute(1,2,0)
        save_img(cond_grid, join(self.intermediate_dir, f"{int(self.camera_poses[0][0])}_{int(self.camera_poses[0][1])}_{cond_type}.jpg"))
        # save_img(mask_grid, join(self.intermediate_dir, f"{self.camera_poses[0][0]}_{self.camera_poses[0][1]}_mask.jpg"))
        cond_grids.append(cond_grid)
        mask_grids.append(mask_grid)
        for i in range(4, len(cond_lists)):
            cond = cond_lists[i].permute(1,2,0)
            mask = mask_lists[i].permute(1,2,0)
            save_img(cond, join(self.intermediate_dir, f"{int(self.camera_poses[i][0])}_{int(self.camera_poses[i][1])}_{cond_type}.jpg"))
            # save_img(mask, join(self.intermediate_dir, f"{self.camera_poses[i][0]}_{self.camera_poses[i][1]}_mask.jpg"))
            cond_grids.append(cond)
            mask_grids.append(mask)
        return cond_grids, mask_grids
        
    
    def get_conds(self):
        # verts, normals, depths, cos_angles, texels, fragments
        self.uvp_rgb.to(self.device)
        _, normals, depths, _, _, _ = self.uvp_rgb.render_geometry() # depths: torch.Size([N, 512, 512, 2])

        masks = normals[...,3][:,None,...] # (N, 1, H, W) alpha channel
        depths = self.uvp_rgb.decode_normalized_depth(depths) # torch.Size([4*4, H, W, 3])
        normals = self.uvp_rgb.decode_view_normal(normals) # *2 - 1

        dep_lists = [cond.permute(2,0,1) for cond in depths]
        norm_lists = [cond.permute(2,0,1) for cond in normals]
        mask_lists = [mask.repeat(3,1,1) for mask in masks]
        dep_grids = []
        norm_grids = []
        mask_grids = []

        dep = torch.stack(dep_lists[0:4])
        norm = torch.stack(norm_lists[0:4])
        mask = torch.stack(mask_lists[0:4])
        dep_grid = make_grid(dep, nrow=2, padding=0).permute(1,2,0)
        norm_grid = make_grid(norm, nrow=2, padding=0).permute(1,2,0)
        mask_grid = make_grid(mask, nrow=2, padding=0).permute(1,2,0)
        save_img(dep_grid, join(self.intermediate_dir, f"{int(self.camera_poses[0][0])}_{int(self.camera_poses[0][1])}_depth.jpg"))
        save_img(norm_grid, join(self.intermediate_dir, f"{int(self.camera_poses[0][0])}_{int(self.camera_poses[0][1])}_normal.jpg"))
        save_img(mask_grid, join(self.intermediate_dir, f"{int(self.camera_poses[0][0])}_{int(self.camera_poses[0][1])}_mask.jpg"))
        
        dep_grids.append(dep_grid)
        norm_grids.append(norm_grid)
        mask_grids.append(mask_grid)
        for i in range(4, len(mask_lists)):
            dep = dep_lists[i].permute(1,2,0)
            norm = norm_lists[i].permute(1,2,0)
            mask = mask_lists[i].permute(1,2,0)
            save_img(dep, join(self.intermediate_dir, f"{int(self.camera_poses[i][0])}_{int(self.camera_poses[i][1])}_depth.jpg"))
            save_img(norm, join(self.intermediate_dir, f"{int(self.camera_poses[i][0])}_{int(self.camera_poses[i][1])}_normal.jpg"))
            save_img(mask, join(self.intermediate_dir, f"{int(self.camera_poses[i][0])}_{int(self.camera_poses[i][1])}_mask.jpg"))
            dep_grids.append(dep)
            norm_grids.append(norm)
            mask_grids.append(mask)

        return dep_grids, norm_grids, mask_grids

    def init_pipeline(self, cond_type, grid):
        controlnet_list = []
       
        # load control net and stable diffusion v1-5
        if cond_type == "depth" or cond_type == "depth&normal":
            controlnet_depth = ControlNetModel.from_pretrained(
                self.dep_ControlNet, variant="fp16", torch_dtype=torch.float16
            )
            controlnet_list.append(controlnet_depth)
        if cond_type == "normal" or cond_type == "depth&normal":       
            controlnet_normal = ControlNetModel.from_pretrained(
                self.norm_ControlNet, ariant="fp16", torch_dtype=torch.float16
            )
            controlnet_list.append(controlnet_normal)

        if grid:
            unet = UNet2DConditionModel.from_pretrained(
                self.mvd_model, subfolder=f"unet-{self.unet_step}", torch_dtype=torch.float16
            )
            logger.info("load Multi-view Diffusion Model")
        else:
            unet = UNet2DConditionModel.from_pretrained(
                self.sd_model, subfolder="unet", torch_dtype=torch.float16
            )
            logger.info("load Stable Diffusion Model")
            # raise ValueError("Need load your unique StableDiffusionPipeline")

        # TODO: if safety problem happens, set 'safety_checker=None'
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.sd_model, unet=unet, controlnet=controlnet_list, safety_checker=None, torch_dtype=torch.float16
        )
        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload()
        pipe = pipe.to(self.diffusion_device)
        logger.info("Finish pipeline load")
        return pipe
    
    def init_HD_pipeline(self, cond_type, grid):
        controlnet_list = []

        # load control net and stable diffusion v1-5
        if cond_type == "depth" or cond_type == "depth&normal":
            controlnet_depth = ControlNetModel.from_pretrained(
                self.dep_ControlNet, variant="fp16", torch_dtype=torch.float16
            )
            controlnet_list.append(controlnet_depth)
        if cond_type == "normal" or cond_type == "depth&normal":       
            controlnet_normal = ControlNetModel.from_pretrained(
                self.norm_ControlNet, ariant="fp16", torch_dtype=torch.float16
            )
            controlnet_list.append(controlnet_normal)

        controlnet_HD = ControlNetModel.from_pretrained(
                self.hd_ControlNet, torch_dtype=torch.float16
            )
        controlnet_list.append(controlnet_HD)

        if grid:
            unet = UNet2DConditionModel.from_pretrained(
                self.mvd_model, subfolder=f"unet-{self.unet_step}", torch_dtype=torch.float16
            )
            logger.info("load Multi-view Diffusion Model")
        else:
            unet = UNet2DConditionModel.from_pretrained(
                self.sd_model, subfolder="unet", torch_dtype=torch.float16
            )
            logger.info("load Stable Diffusion Model")

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.sd_model, unet = unet, controlnet=controlnet_list, safety_checker=None, torch_dtype=torch.float16
        )

        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        pipe = pipe.to(self.diffusion_device)
        logger.info("Finish HD pipeline load")

        return pipe

        
    def init_inpaint_pipeline(self, cond_type):
        controlnet_list = []

        # load control net and stable diffusion v1-5
        if cond_type == "depth" or cond_type == "depth&normal":
            controlnet_depth = ControlNetModel.from_pretrained(
                self.dep_ControlNet, variant="fp16", torch_dtype=torch.float16
            )
            controlnet_list.append(controlnet_depth)
        if cond_type == "normal" or cond_type == "depth&normal":       
            controlnet_normal = ControlNetModel.from_pretrained(
                self.norm_ControlNet, ariant="fp16", torch_dtype=torch.float16
            )
            controlnet_list.append(controlnet_normal)

        controlnet_inpaint = ControlNetModel.from_pretrained(
                self.ipait_ControlNet, variant="fp16", torch_dtype=torch.float16
                )
        controlnet_list.append(controlnet_inpaint)
        
        unet = UNet2DConditionModel.from_pretrained(
            self.sd_model, subfolder="unet", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.sd_model, unet = unet, controlnet=controlnet_list, safety_checker=None, torch_dtype=torch.float16
        )

        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        pipe = pipe.to(self.diffusion_device)
        logger.info("Finish inpaint pipeline load")

        return pipe
        
    def render_tex_rgb(self, idx, mask=False, Isgrid=True):
        views = self.uvp_rgb.render_textured_views()
        textured_views_rgb = [image[:-1,...] for image in views] # TODO : maybe need check -1
        
        if Isgrid:
            imgs= torch.stack(textured_views_rgb[idx*4:(idx+1)*4])
            view = make_grid(imgs, nrow=2, padding=0).permute(1,2,0)
        else:
            view= textured_views_rgb[idx].permute(1,2,0)
        
        if not mask:
            save_img(view, join(self.intermediate_dir, f"{int(self.camera_poses[idx][0])}_{int(self.camera_poses[idx][1])}_rgb.jpg"))
            return view  # [h, w, c]
        else:
            save_img(1-view, join(self.intermediate_dir, f"{int(self.camera_poses[idx][0])}_{int(self.camera_poses[idx][1])}_mask.jpg"))
            return 1-view


    def inpaint(self, prompt, negative_prompt, steps, texture=None, no_HD=False, weight_limit=0.15):
        logger.info('Start InPainting ^_^')

        if self.cond_type == "depth&normal":
            dep_imgs, norm_imgs, _ = self.get_conds() #[4, H, W, c]
        else:
            cond_imgs, masks = self.get_cond_images(cond_type=self.cond_type) #[4, H, W, c]
        generator = torch.Generator(self.diffusion_device).manual_seed(self.seed)

        steps = len(self.camera_poses)
        if self.grid_diffusion:
            steps -= 3
        pbar = tqdm(total=steps, initial=0,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        views = []
        i = 0

        while i < steps:
            if i == 0:
                if self.cond_type == "depth&normal":
                    control_img = [dep_imgs[i].unsqueeze(0).cpu().numpy(), norm_imgs[i].unsqueeze(0).cpu().numpy()]
                else:
                    control_img = [cond_imgs[i].unsqueeze(0).cpu().numpy()]
                sd_output = self.pipe(
                    "4 orthogonal views of "+ prompt,
                    control_img, #  [1, H, W, c]
                    negative_prompt=negative_prompt,
                    generator=generator,
                    num_inference_steps=steps,
                )

                del self.pipe
                if not no_HD:
                    logger.info("Start HD.")
                    sd_output.images[0].save(join(self.intermediate_dir, f"{int(self.camera_poses[i][0])}_{int(self.camera_poses[i][1])}_raw-pipe-out.jpg"))
                    control_img.append(sd_output.images[0]) # .resize((2048, 2048))

                    # update the detail of 4 views
                    self.HD_pipe = self.init_HD_pipeline(cond_type=self.cond_type, grid=self.grid_diffusion)
                    if prompt.endswith("."):
                        HD_prompt = prompt[:-1] + ", more details, high quality, best quality."
                    elif prompt.endswith(","):
                        HD_prompt = prompt + " more details, high quality, best quality."
                    else:
                        HD_prompt = prompt + ", more details, high quality, best quality."
                    sd_output = self.HD_pipe(
                        prompt="4 orthogonal views of " + HD_prompt, 
                        negative_prompt=negative_prompt, 
                        image=sd_output.images[0], 
                        control_image=control_img, 
                        strength=1.0,
                        generator=generator,
                        num_inference_steps=steps,
                    )
                    # sd_grid = sd_output.images[0] # .resize((1024,1024))
                    del self.HD_pipe
                self.inpaint_pipe = self.init_inpaint_pipeline(cond_type=self.cond_type)  
            else:
                view_grid = self.render_tex_rgb(i+3, Isgrid=False).permute(2,0,1) # [C, H, W]

                self.uvp_rgb.set_texture_map(weight.to(torch.float32).permute(2,0,1)) 
                mask = self.render_tex_rgb(i+3, mask=True, Isgrid=False).permute(2,0,1)
                mask = mask[0,:,:]

                input_img = fill_image(view_grid, mask)
                # input_img.save(join(self.intermediate_dir, f"{self.camera_poses[i][0]}_{self.camera_poses[i][1]}_fill.jpg"))

                inpaint_image = make_inpaint_condition(input_img, mask) # [C, H, W]
                # save_img(inpaint_image.permute(1,2,0), join(self.intermediate_dir, f"{self.camera_poses[i][0]}_{self.camera_poses[i][1]}_inpaint-input.jpg"))
                
                # cond_img: cuda, inpaint_image: cpu
                if self.cond_type == "depth&normal":
                    control_img = [dep_imgs[i].permute(2,0,1).unsqueeze(0), norm_imgs[i].permute(2,0,1).unsqueeze(0), inpaint_image.unsqueeze(0)]
                else:
                    control_img = [cond_imgs[i].permute(2,0,1).unsqueeze(0), inpaint_image.unsqueeze(0)] # [1, 3, 1024, 1024] *2

                view_prompt = get_view(self.camera_poses[i][0], self.camera_poses[i][1]) + prompt
                sd_output = self.inpaint_pipe(
                    view_prompt, 
                    negative_prompt=negative_prompt,
                    image=input_img, # view_grid.unsqueeze(0).permute(0,2,3,1).cpu().numpy(), # [1, C, H, W]
                    mask_image=mask.cpu().numpy(),
                    control_image=control_img, # [C, H, W]*2
                    generator=generator,
                    num_inference_steps=steps,
                )

            sd_grid = sd_output.images[0]
            sd_grid.save(join(self.intermediate_dir, f"{int(self.camera_poses[i+3][0])}_{int(self.camera_poses[i+3][1])}_pipe-out.jpg"))

            # split views
            if i == 0:
                sd_views =  split_grid_img(sd_grid, device=self.device)
               
            else:
                sd_view = torch.tensor(np.array(sd_grid), dtype=torch.float32).to(self.device)
                sd_view /= 255.0
                sd_views = [sd_view.permute(2, 0, 1)]

            views = views+sd_views  

            sample_views, texture, weight = self.uvp_rgb.bake_texture(views=views, exp=1)
            # update texture
            self.uvp_rgb.set_texture_map(texture) # [c, h, w]
            
            # calculate invalid faces(weight * valid_faceids)
            # valid_pix2face = (weight.permute(1,2,0)[:,:,:1] > 0) * self.uvp_rgb.uv_pix2face[0]

            textured_views_rgb =  torch.cat(sample_views, axis=-1)[:-1,...] # views[0][:-1,...]
            textured_views_rgb = textured_views_rgb.permute(1,2,0)
            save_img(textured_views_rgb, join(self.intermediate_dir, f"multi-views-{i}.jpg"))          
            
            colored = (weight.permute(1,2,0)> 1e-5).int()
            save_img(colored, join(self.intermediate_dir, f"colored-{i}.jpg"))

            weight = (weight.permute(1,2,0)> weight_limit).int()
            save_img(weight, join(self.intermediate_dir, f"weight-{i}.jpg"))

            texture_img = texture.permute(1,2,0)
            save_img(texture_img, join(self.intermediate_dir, f"texture-{i}.jpg"))

            # # get new cameras
            # valid_faceids, invalid_verts = self.uvp_rgb.get_invalid_verts(valid_pix2face)
            # new_cameras = self.uvp_rgb.get_new_cameras(self.camera_poses, invalid_verts)
            # if len(new_cameras) == 0:
            #     break
            # # select new pose
            # R, T, camera_id = self.uvp_rgb.select_new_pose(new_cameras, all_valid_faceids=valid_faceids, centers=None, camera_distance=4.0, scale=self.scale)
            # # add camera
            # self.uvp_rgb.add_camera(R, T, scale=self.scale)
            # self.camera_poses.append(new_cameras[camera_id])  
            
            i += 1
            pbar.update(1)
        
        self.uvp_rgb.save_mesh(join(self.result_dir, "textured.obj"), texture.permute(1,2,0))

        self.uvp_rgb.set_texture_map(texture)

        views = self.uvp_rgb.render_textured_views()
        textured_views_rgb = torch.cat(views, axis=-1)[:-1,...]
        textured_views_rgb = textured_views_rgb.permute(1,2,0)
        save_img(textured_views_rgb, join(self.result_dir, "multi_views_result.jpg"))
        
        logger.info('Finish Painting ^_^')
