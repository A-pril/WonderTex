CUDA_VISIBLE_DEVICES=0 \
blenderproc run --custom-blender-path /path/to/blender-3.3.0-linux-x64 \
blenderProc_ortho.py --object_path objaverse/3ed05d0386d745ed95ae6ec7f4ee86dd.glb \
--output_folder rendering --object_uid 3ed05d0386d745ed95ae6ec7f4ee86dd --ortho_scale 1.00 --resolution 512 
#  --reset_object_euler