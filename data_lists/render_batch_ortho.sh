 python distributed.py \
	--num_gpus 8 --gpu_list 0 1 2 3 4 5 6 7 --mode render_ortho    \
	--workers_per_gpu 10 --ortho_scale 1.00 \
	--start_i $1 --end_i $2  \
	--input_models_path data_prepare/ours_uids_highquality_no3Dword.json  \
	--objaverse_root /path/to/hf-objaverse-v1/glbs \
	--save_folder rendering \
	--custom_blender_path /path/to/blender-3.3.0-linux-x64

 python distributed.py \
	--num_gpus 2 --gpu_list 0 1 --mode render_ortho    \
	--workers_per_gpu 2 \
	--ortho_scale 1.00 \
	--input_models_path test.json  \
	--objaverse_root objaverse \
	--save_folder rendering \
	--custom_blender_path /path/to/blender-3.3.0-linux-x64