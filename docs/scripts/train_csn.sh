#!/bin/sh

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v1" --kernel_size=7 --batch_size=2 >train_out_0.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v1" --kernel_size=1 --batch_size=2 >train_out_1.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v1" --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 >train_out_2.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v2" --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.5 >train_out_v2_aug.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v2" --kernel_size=1 --batch_size=2 --degrees=15 --translation=0.2 >train_out_v2_less_aug.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v2" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 >train_out_v2_no_aug.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v2" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" >train_out_v2_no_aug_no_scale.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v2" --kernel_size=1 --batch_size=2 --degrees=15 --translation=0.2 >train_out_v2_less_aug_with_resize_first.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v2" --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.2 --aug_scale="1,1" >train_out_v2_aug_no_scale.out 2>&1 &

# model 67-69
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v3" --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.2 --aug_scale="0.8,1.2" >train_out_v3_aug_medium.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v3" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" >train_out_v3_no_aug_no_scale.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v3" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" >train_out_v3_aug_large.out 2>&1 &

# model 70-71
# something wrong with pytorch_lightning: model71 model72 somehow output to the same folder...? outputs are not empty.
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.2 --aug_scale="0.8,1.2" >train_out_v4_aug_medium.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" >train_out_v4_no_aug_no_scale.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" >train_out_v4_aug_large.out 2>&1 &

# model 72-73 (retrain the previous three training...as there was something wrong with the workstation)
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" >train_out_v4_no_aug_no_scale_run1.out 2>&1 &

nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.2 --aug_scale="0.8,1.2" >train_out_v4_aug_medium_run1.out 2>&1 &

# model 75
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate >train_out_v4_model74.out 2>&1 &


# model 76
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE >train_out_v4_model76.out 2>&1 &

# model 77
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --loss=CE >train_out_v4_model77.out 2>&1 &

# model 78
# compare with 77: exclude bg
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --loss=CE --exclude_raw_input_bg>train_out_v4_model78.out 2>&1 &

# model 79
# compare with 76: exclude bg
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE --exclude_raw_input_bg>train_out_v4_model79.out 2>&1 &

# model 80
# compare with 75: exclude bg
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate >train_out_v4_model80.out --exclude_raw_input_bg 2>&1 &

# model 81
# compare with 79&76: use raw_duplicate instead of raw_aug_duplicate
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --model_ckpt="/home/ke/LiveCellTracker-dev/notebooks/lightning_logs/version_81/checkpoints/epoch=270-step=60704.ckpt">train_out_v4_model81.out 2>&1 &

# model 82
# compare with 75
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate --class-weights=1,2,2>train_out_v4_model82.out 2>&1 &
# model 83
# compare with 82
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate --class-weights=1,10,10>train_out_v4_model83.out 2>&1 &

# model 86 (forgot to run)
# compare with 82
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate --class-weights=1,20,20>train_out_v4_model86_resume.out 2>&1 &


# model 84
# compare with 81
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,2,2>train_out_v4_model84.out 2>&1 &

# model 85
# compare with 85
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,10,10>train_out_v4_model85.out 2>&1 &

# model 87 (not yet trained)
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,20,20>train_out_v4_model87.out 2>&1 &