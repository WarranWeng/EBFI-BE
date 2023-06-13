############## synthetic data
CUDA_VISIBLE_DEVICES='0' \
    python infer_ours.py \
        --model_path /path/to/model \
        --data_list /path/to/test.txt \
        --output_path /path/to/output \
        --scale 2 \
        --ori_scale down2 \
        --time_bins 16 \
        --num_frame_per_period 16 \ # tune for different exposure assumptions 
        --num_frame_per_blurry 3 \ # tune for different exposure assumptions 
        --num_period_per_seq 2 \
        --sliding_window_seq 2 \
        --num_period_per_load 1 \
        --sliding_window_load 1 \
        --exposure_method Fixed \ # Auto/Fixed/Custom
        --noise_enabled 


############## real-world data: RealBlur-DAVIS
CUDA_VISIBLE_DEVICES='1' \
    python infer_ours.py \
        --model_path /path/to/model \
        --data_list /path/to/test.txt \
        --output_path /path/to/output \
        --scale 2 \
        --ori_scale down2 \
        --time_bins 16 \
        --interp_num 256 \ # define frame number for interpolation
        --num_period_per_seq 2 \
        --sliding_window_seq 2 \
        --num_period_per_load 1 \
        --sliding_window_load 1 \
        --noise_enabled \
        --real_blur









