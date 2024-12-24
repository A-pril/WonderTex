# Prepare the data

## Data from objaverse dataset
The 3D models we used for fine-tuning are from the LVIS subset of objaverse dataset, and we get the valid models based on the [lvis_uids_filter_by_vertex.json](./data_prepare/lvis_uids_filter_by_vertex.json) and [lvis_invalid_uids_nineviews.json](./data_prepare/lvis_invalid_uids_nineviews.json) from [Wonder3D](https://github.com/xxlong0/Wonder3D). 

Then according the [Cap3D](https://huggingface.co/datasets/tiange/Cap3D), we choose the high quality models which have the prompts without 3D word from the valid LVIS subset as our final data for fine-tuning, which are recorded in [ours_uids_nineviews.json](./data_prepare/ours_uids_highquality_no3Dword.json).


## Setup
The rendering codes are mainly based on [BlenderProc](https://github.com/DLR-RM/BlenderProc). Thanks for the great tool.
BlenderProc uses blender Cycle engine to render the images by default, which may meet long-time hanging problem in some specific GPUs.

```
pip install -r requirements.txt
```

## Render
Here we provide rendering script `blenderProc_ortho.py` which use **orthogonal** camera to render the objects, and get the rendered 2 $\times$ 2 grid normal, depth and rgb images of the objects.

### Single model mode
To render images of a single object, you can use this scripts.
```bash
bash render_single_ortho.sh
```

Or use `blenderProc_ortho.py` to render images of a single object, as follows.

```bash
CUDA_VISIBLE_DEVICES=0 \
blenderproc run --custom-blender-path /path/to/blender-3.3.0-linux-x64 \
blenderProc_ortho.py --object_path d0c0522e9a6d4684a2744b06143bf9a5.glb \
--output_folder ./out_renderings/  --object_uid d0c0522e9a6d4684a2744b06143bf9a5 \
--ortho_scale 1.00 --resolution 512 
```

Here `--ortho_scale` decides the scaling of rendered object in the image, `--object_uid` is the final output folder name.

### Multiple models mode
For distributed mode, please refer to `render_batch_ortho.sh`, as follows.
```bash
bash render_batch_ortho.sh
```

## Acknowledgement 💌
The rendering code is based on Wonder3D. Thanks to the great work.

For more information like perspective camera rendering, please refer to [Wonder3D](https://github.com/xxlong0/Wonder3D/tree/main/render_codes).


