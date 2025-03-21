#!/bin/sh

eval_out_dir="./csn_eval_results/"
mkdir -p $eval_out_dir
modelVer=v11_01-lr=0.00001
python ../livecellx/model_zoo/segmentation/eval_csn.py \
    --name="csn_model$modelVer" \
    --train_dir="./notebook_results/a549_ccp_vim/train_data_v11" \
    --test_dir="./notebook_results/a549_ccp_vim/test_data_v11"\
    --ckpt="./lightning_logs/version_$modelVer/checkpoints/epoch=999-global_step=0.ckpt"\
    --out_threshold=0.5 \
    --save_dir=$eval_out_dir\
    --wait_for_gpu_mem\
    >$eval_out_dir/eval_model$modelVer.out 2>&1 &

# modelVer=78
# python ../livecellx/model_zoo/segmentation/eval_csn.py \
#     --name="csn_model$modelVer" \
#     --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" \
#     --test_dir="./notebook_results/a549_ccp_vim/test_data_v4"\
#     --ckpt="./lightning_logs/version_$modelVer/checkpoints/epoch=999-step=224000.ckpt"\
#     --out_threshold=0.5 \
#     --save_dir="./eval_results/"\
#     --wait_for_gpu_mem\
#     >$eval_out_dir/eval_model$modelVer.out 2>&1 &

# for modelVer in $(seq 75 85)
# do
# python ../livecellx/model_zoo/segmentation/eval_csn.py \
#     --name="csn_model$modelVer" \
#     --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" \
#     --test_dir="./notebook_results/a549_ccp_vim/test_data_v4"\
#     --ckpt="./lightning_logs/version_$modelVer/checkpoints/epoch=999-step=224000.ckpt"\
#     --out_threshold=0.5 \
#     --save_dir="./eval_results/"\
#     --wait_for_gpu_mem\
#     >$eval_out_dir/eval_model$modelVer.out 2>&1 &
# done