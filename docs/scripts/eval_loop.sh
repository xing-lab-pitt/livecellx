#!/bin/sh

# eval_out_dir="./eval_results/"
out_threshold=1
save_dir="./eval_results_out="$out_threshold"_v1-viz_lab3/"

for modelVer in $(seq 88 100)
do
python ../livecellx/model_zoo/segmentation/eval_csn.py \
    --name="csn_model$modelVer" \
    --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" \
    --test_dir="./notebook_results/a549_ccp_vim/test_data_v4"\
    --ckpt="./lightning_logs/version_$modelVer/checkpoints/epoch=999-step=224000.ckpt"\
    --out_threshold=$out_threshold \
    --save_dir=$save_dir\
    --wait_for_gpu_mem\
    --viz_pred
done

# for modelVer in $(seq 76 87)
# do
# python ../livecellx/model_zoo/segmentation/eval_csn.py \
#     --name="csn_model$modelVer" \
#     --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" \
#     --test_dir="./notebook_results/a549_ccp_vim/test_data_v4"\
#     --ckpt="./lightning_logs/version_$modelVer/checkpoints/epoch=999-step=224000.ckpt"\
#     --out_threshold=$out_threshold \
#     --save_dir="./eval_results_out="$out_threshold"_v1-metrics_lab3/"\
#     --wait_for_gpu_mem
# done