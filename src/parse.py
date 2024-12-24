import configargparse

def parse_config():
    parser = configargparse.ArgumentParser(
                        prog='WonderTex Diffusion',
                        description='Generate texture given mesh and texture prompt',
            )
    
    # File Config
    parser.add_argument('--config', type=str, default='./config/config.yaml', is_config_file=True) # required=True,
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--mesh_config_relative', action='store_true', help="Search mesh file relative to the config path instead of current working directory")
    parser.add_argument('--output', type=str, default=None, help="If not provided, use the parent directory of config file for output")
    parser.add_argument('--prefix', type=str, default='Tex', help="The prefix of output name")
    parser.add_argument('--use_mesh_name', action='store_true')
    parser.add_argument('--timeformat', type=str, default='%m%d-%H%M%S', help='Setting to None will not use time string in output directory')
    # parser.add_argument('--timeformat', type=str, default='%d%b%Y-%H%M%S', help='Setting to None will not use time string in output directory')
    
    # Run Config
    parser.add_argument('--gpu', type=int, default=0, help='Choose gpu id to run')
    parser.add_argument('--diffusion_gpu', type=int, default=0, help='Choose gpu id to load and run diffusion model')
    
    # Diffusion Config
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str, default='oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect.')
    parser.add_argument('--inference_steps', type=int, default=20, help='The inference steps of stable diffusion model')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grid_diffusion', action='store_true', help='Default, use finetuned grid-diffusion base model')
    
    # Model Config 
    parser.add_argument('--sd_model', type=str, default="runwayml/stable-diffusion-v1-5", help='The stable diffusion model id you use')
    # TODO: Upload hf
    parser.add_argument('--mvd_model', type=str, default="xxxx/4-view_diffusion", help='The unet of the 4-view image diffusion model id you use')
    parser.add_argument('--unet_step', type=int, default=10000, help='The training step of the fine-tuned unet you use')
    parser.add_argument('--dep_ControlNet', type=str, default="lllyasviel/control_v11f1p_sd15_depth", help='The depth ControlNet model id you use')
    parser.add_argument('--norm_ControlNet', type=str, default="lllyasviel/control_v11p_sd15_normalbae", help='The normal ControlNet model you use')
    parser.add_argument('--hd_ControlNet', type=str, default="lllyasviel/control_v11f1e_sd15_tile", help='The High-definition ControlNet model you use')
    parser.add_argument('--ipait_ControlNet', type=str, default="lllyasviel/control_v11p_sd15_inpaint", help='The inpaint ControlNet model you use')

    # ControlNet Config
    parser.add_argument('--cond_type', type=str, default='depth', help='Support depth and normal, less multi-face in normal mode, but some times less details')
    # parser.add_argument('--guess_mode', action='store_true')
    parser.add_argument('--no_HD', action='store_true', help='if set no_HD, do not use HD pipeline')

    # Camera Config
    # parser.add_argument('--cameras', type=int, default=[(0, 0), (0, 90), (0, 180), (0, 270), (45, 0), (-15, 45), (-15, 225)], action="append",help='The camera poses used for generating views')
    parser.add_argument('--init_elev', type=int, default=0, help='The initial elevation angel of camera pose used for 4-view SD')
    parser.add_argument('--min_elev', type=int, default=-30, help='The minimum elevation angle of cameras')
    parser.add_argument('--max_elev', type=int, default=60, help='The maximum elevation angle of cameras')
    parser.add_argument('--diff_angle', type=int, default=15, help='The smallest angle difference between two cameras')
    parser.add_argument('--eps', type=float, default=0.05, help='The eps of DBSCAN')
    parser.add_argument('--samples', type=int, default=10, help='The min_samples of DBSCAN')

    # Render Config
    parser.add_argument('--rgb_view_size', type=int, default=512)
    parser.add_argument('--rgb_tex_size', type=int, default=1024)

    parser.add_argument('--weight_limit', type=float, default=0.15, help='The number of weight visibity to determain the mask')
    parser.add_argument('--mesh_autouv', action='store_true', help='Use Xatlas to unwrap UV automatically')
    parser.add_argument('--scale', type=float, default=1.35, help='Set above 1 to enlarge object in camera views')
    

    options = parser.parse_args()
    return options
