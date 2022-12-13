#!/bin/sh

# eval_out_dir="./eval_results/"
for modelVer in $(seq 75 85)
do
python ../livecell_tracker/model_zoo/segmentation/eval_csn.py \
    --name="csn_model$modelVer" \
    --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" \
    --test_dir="./notebook_results/a549_ccp_vim/test_data_v4"\
    --ckpt="./lightning_logs/version_$modelVer/checkpoints/epoch=999-step=224000.ckpt"\
    --out_threshold=0.5 \
    --save_dir="./eval_results_v1/"\
    --wait_for_gpu_mem
done