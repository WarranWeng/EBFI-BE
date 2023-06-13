# first train exposure estimation
CUDA_VISIBLE_DEVICES='0' \
        python -m torch.distributed.launch --nproc_per_node 1 --use_env --master_port 355827 \
            train_ours_exposuredecision.py -c config\train_ours_exposuredecision.yml -id provide_id_name

# then train the whole model
CUDA_VISIBLE_DEVICES='1' \
        python -m torch.distributed.launch --nproc_per_node 1 --use_env --master_port 355829 \
            train_ours.py -c config/train_ours.yml  -id provide_id_name