import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid
from torchvision.transforms import Resize, InterpolationMode
import cv2


def save_img(img, file_path):
    img = img.cpu().numpy()
    img = (img * 255.0).round().astype(np.uint8)
    if img.shape[-1] == 1:
        # special case for grayscale (single channel) images
        img = Image.fromarray(img.squeeze(), mode="L")
    else:
        img = Image.fromarray(img)
    img.save(file_path)

def split_grid_img(grid_img, device):
    resolution = grid_img.size[0] // 2

    sub_grids = []

    for i in range(2):  
        for j in range(2): 
            left = j * resolution
            top = i * resolution
            right = left + resolution
            bottom = top + resolution
            cropped_image = grid_img.crop((left, top, right, bottom))
            sub_grid = torch.tensor(np.array(cropped_image), dtype=torch.float32).to(device)
            sub_grid /= 255.0
            sub_grid = sub_grid.permute(2, 0, 1) # (c, h, w)
            sub_grids.append(sub_grid)

    return sub_grids

def split_grids(grids: list, dir: str = None, name: str = None):
    views = []
    for grid in grids:
        sub_res = grid.shape[1] // 2
        for i in range(2):
            for j in range(2):
                left = j * sub_res
                top = i * sub_res
                right = left + sub_res
                bottom = top + sub_res
                view = grid[:, top:bottom, left:right]
                if dir is not None and i<1 and j<1:
                    import os
                    save_img(view[:-1].permute(1,2,0), os.path.join(dir, f"{name}_{i}_{j}.jpg"))
                views.append(view)

    return views

def make_grids(views: list, save_path: str = None):
    grids = []
    for i in range(len(views)//4): # the num of grid
        views_stack = torch.stack(views[i*4:(i+1)*4])
        grid = make_grid(views_stack, nrow=2, padding=0).permute(1,2,0)
        if save_path is not None:
            save_img(grid, save_path)
        grids.append(grid.permute(2, 0, 1))
    
    return grids

@torch.no_grad()
def encode_latents(vae, imgs):
	imgs = (imgs-0.5)*2
	latents = vae.encode(imgs).latent_dist.sample()
	latents = vae.config.scaling_factor * latents
	return latents

# A fast decoding method based on linear projection of latents to rgb
@torch.no_grad()
def latent_preview(x):
	# adapted from https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7
	v1_4_latent_rgb_factors = torch.tensor([
		#   R        G        B
		[0.298, 0.207, 0.208],  # L1
		[0.187, 0.286, 0.173],  # L2
		[-0.158, 0.189, 0.264],  # L3
		[-0.184, -0.271, -0.473],  # L4
	], dtype=x.dtype, device=x.device)
	image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
	image = (image / 2 + 0.5).clamp(0, 1)
	image = image.float()
	image = image.cpu()
	image = image.numpy()
	return image

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

@torch.no_grad()
def decode_latents(vae, latents):

	latents = 1 / vae.config.scaling_factor * latents

	image = vae.decode(latents, return_dict=False)[0]
	torch.cuda.current_stream().synchronize()

	image = (image / 2 + 0.5).clamp(0, 1)
	# we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
	image = image.permute(0, 2, 3, 1)
	image = image.float()
	image = image.cpu()
	image = image.numpy()
	
	return image


# Decode each view and bake them into a rgb texture
def get_rgb_texture(vae, uvp_rgb, latents):
    result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    resize = Resize((uvp_rgb.render_size,)*2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
    result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
    textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, main_views=[], exp=6, noisy=False)
    result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()[None,...]
    return result_tex_rgb, result_tex_rgb_output

def make_inpaint_condition(image, image_mask):
    """
    image: Image
    image_mask: [H, W]
    """
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = image_mask.cpu().numpy()
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image)

    return image

def fill_image(image, image_mask, inpaintRadius=1):
    image = image.permute(1,2,0).cpu().numpy() * 255
    image_mask = image_mask.cpu().numpy() * 255

    filled_image = cv2.inpaint(image.astype(np.uint8), image_mask.astype(np.uint8), inpaintRadius, cv2.INPAINT_TELEA)

    res_img = Image.fromarray(np.clip(filled_image, 0, 255).astype(np.uint8))

    return res_img

def get_view(elev, azith):
    descrip = ""
    if elev < -15:
        descrip = "bottom view of "
    elif elev > 30:
        descrip = "top view of "
    else:
        if azith > 60 and azith < 120:
            descrip = "left view of "
        elif  azith > 120 and azith < 240:
            descrip = "back view of "
        elif azith > 240 and azith < 300:
            descrip = "right view of "
    
    return descrip