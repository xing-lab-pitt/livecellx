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
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate >train_out_v4_model75.out 2>&1 &

# model 76
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE >train_out_v4_model76.out 2>&1 &

# model 77
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --loss=CE >train_out_v4_model77.out 2>&1 &

# model 78
# compare with 77: exclude bg
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --loss=CE --exclude_raw_input_bg >train_out_v4_model78.out 2>&1 &

# model 79
# compare with 76: exclude bg
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE --exclude_raw_input_bg >train_out_v4_model79.out 2>&1 &

# model 80
# compare with 75: exclude bg
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate --exclude_raw_input_bg >train_out_v4_model80.out 2>&1 &

# model 81
# compare with 79&76: use raw_duplicate instead of raw_aug_duplicate
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --model_ckpt="/home/ke/LiveCellTracker-dev/notebooks/lightning_logs/version_81/checkpoints/epoch=270-step=60704.ckpt" >train_out_v4_model81.out 2>&1 &

# model 82
# compare with 75
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate --class-weights=1,2,2 >train_out_v4_model82.out 2>&1 &
# model 83
# compare with 82
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate --class-weights=1,10,10 >train_out_v4_model83.out 2>&1 &

# model 86 (forgot to run)
# compare with 82
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_aug_duplicate --class-weights=1,20,20 >train_out_v4_model86_resume.out 2>&1 &

# model 84
# compare with 81
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,2,2 >train_out_v4_model84.out 2>&1 &

# model 85
# compare with 84
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,10,10 >train_out_v4_model85.out 2>&1 &

# model 87 (not yet trained)
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=0 --translation=0 --aug_scale="1,1" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,20,20 >train_out_v4_model87.out 2>&1 &

############################################################################################################
# train with augmentation: repeat 75-87
############################################################################################################

# model 88
# model 75 aug
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate >train_out_v4_model88.out 2>&1 &

# model 89
# model 76 aug
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE >train_out_v4_model89.out 2>&1 &

# model 90
# model 77 aug
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --loss=CE >train_out_v4_model90.out 2>&1 &

# model 91
# model 78 aug
# compare with 77: exclude bg
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --loss=CE --exclude_raw_input_bg >train_out_v4_model91.out 2>&1 &

# model 92
# model 79 aug
# compare with 76: exclude bg
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE --exclude_raw_input_bg >train_out_v4_model92.out 2>&1 &

# model 93
# model 80 aug
# compare with 75: exclude bg
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --exclude_raw_input_bg >train_out_v4_model93.out 2>&1 &

# model 94
# model 81 aug
# compare with 79&76: use raw_duplicate instead of raw_aug_duplicate
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --model_ckpt="/home/ke/LiveCellTracker-dev/notebooks/lightning_logs/version_81/checkpoints/epoch=270-step=60704.ckpt" >train_out_v4_model94.out 2>&1 &

# model 95
# model 82 aug
# compare with 75
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,2,2 >train_out_v4_model95.out 2>&1 &

# model 96
# model 83 aug
# compare with 82
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,10,10 >train_out_v4_model96.out 2>&1 &

# model 97
# aug model 86
# compare with 82
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,20,20 >train_out_v4_model97_resume.out 2>&1 &

# model 98
# model 84 aug
# compare with 81
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,2,2 >train_out_v4_model98.out 2>&1 &

# model 99
# model 85 aug
# compare with 84
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,10,10 >train_out_v4_model99.out 2>&1 &

# model 100
# model 87 aug (not yet trained)
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --kernel_size=1 --batch_size=2 --model_version=version_100 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,20,20 >train_out_v4_model100.out 2>&1 &

############################################################################################################
# for model 88-100, train another 1000 epochs
############################################################################################################

# model 101
# model 88 resume
# model 75 aug
model=101
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --model_ckpt="/home/ken67/LiveCellTracker-dev/notebooks/lightning_logs/version_88/checkpoints/epoch=999-step=224000.ckpt" >train_out_v4_model"$model"_resume.out 2>&1 &

# model 102
# model 89 resume
# model 76 aug
model=102
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE --model_ckpt="$HOME/LiveCellTracker-dev/notebooks/lightning_logs/version_89/checkpoints/epoch=999-step=224000.ckpt" >train_out_v4_model"$model"_resume.out 2>&1 &

# model 103
# model 90 resume
# model 77 aug
model=103
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --loss=CE --model_ckpt="$HOME/LiveCellTracker-dev/notebooks/lightning_logs/version_90/checkpoints/epoch=999-step=224000.ckpt" >train_out_v4_model"$model"_resume.out 2>&1 &

# model 104
# model 91
# model 78 aug
# compare with 77: exclude bg
model=104
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --loss=CE --exclude_raw_input_bg --model_ckpt="$HOME/LiveCellTracker-dev/notebooks/lightning_logs/version_91/checkpoints/epoch=999-step=224000.ckpt" >train_out_v4_model"$model"_resume.out 2>&1 &

# model 105
# model 92
# model 79 aug
# compare with 76: exclude bg
model=105
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE --exclude_raw_input_bg --model_ckpt="$HOME/LiveCellTracker-dev/notebooks/lightning_logs/version_92/checkpoints/epoch=999-step=224000.ckpt" >train_out_v4_model"$model"_resume.out 2>&1 &

# model 106
# model 93
# model 80 aug
# compare with 75: exclude bg
model=106
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --loss=CE --model_ckpt="$HOME/LiveCellTracker-dev/notebooks/lightning_logs/version_93/checkpoints/epoch=999-step=224000.ckpt" >train_out_v4_model"$model"_resume.out 2>&1 &

# model 107
# model 94
# model 81 aug
# compare with 79&76: use raw_duplicate instead of raw_aug_duplicate
model=107
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --model_ckpt="$HOME/LiveCellTracker-dev/notebooks/lightning_logs/version_94/checkpoints/epoch=999-step=224000.ckpt" >train_out_v4_model"$model"_resume.out 2>&1 &

# model 108 (train on slurm cluster)
# model 95
# model 82 aug
# compare with 75
model=108
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,2,2 --model_ckpt="$HOME/LiveCellTracker-dev/notebooks/lightning_logs/version_95/checkpoints/epoch=999-step=224000.ckpt" >train_out_v4_model"$model"_resume.out 2>&1 &

# model 109 (train on slurm cluster)
# model 96
# model 83 aug
# compare with 82
model=109
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,10,10 --model_ckpt="$HOME/LiveCellTracker-dev/notebooks/lightning_logs/version_96/checkpoints/epoch=999-step=224000.ckpt" >train_out_v4_model"$model"_resume.out 2>&1 &

# model 110 (train on slurm cluster)
# model 97
# aug model 86
# compare with 82
model=110
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,20,20 --model_ckpt="$HOME/LiveCellTracker-dev/notebooks/lightning_logs/version_97/checkpoints/epoch=999-step=224000.ckpt">train_out_v4_model"$model"_resume.out 2>&1 &

# model 111 (train on slurm cluster)
# model 98
# model 84 aug
# compare with 81
model=111
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,2,2 --model_ckpt="$HOME/LiveCellTracker-dev/notebooks/lightning_logs/version_98/checkpoints/epoch=999-step=224000.ckpt">train_out_v4_model"$model"_resume.out 2>&1 &

# model 112 (train on slurm cluster)
# model 99
# model 85 aug
# compare with 84
model=112
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,10,10 >train_out_v4_model"$model"_resume.out 2>&1 &

# model 113 (train on slurm cluster)
# model 100
# model 87 aug (not yet trained)
model=113
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,20,20 >train_out_v4_model"$model"_resume.out 2>&1 &

############################################################################################################
# start from 200: replace CE with BCE (retrain previous default arg models)
############################################################################################################

# model 201
# model 101
# model 88 resume
# model 75 aug
model=201
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py \
    --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" \
    --model_version=version_$model \
    --epochs=2000 \
    --kernel_size=1 \
    --batch_size=2 \
    --degrees=180 \
    --translation=0.5 \
    --aug_scale="0.5,2" \
    --input_type=raw_aug_duplicate \
    --loss=BCE \
    >train_out_v4_model"$model"_resume.out 2>&1 &

# model 203
# model 103
# model 90 resume
# model 77 aug
model=203
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --loss=BCE >train_out_v4_model"$model"_resume.out 2>&1 &


# model 204
# model 104
# model 91
# model 78 aug
# compare with 77: exclude bg
model=204
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_duplicate --loss=BCE --exclude_raw_input_bg >train_out_v4_model"$model"_resume.out 2>&1 &


# model 206
# model 106
# model 93
# model 80 aug
# compare with 75: exclude bg
model=206
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --loss=BCE  >train_out_v4_model"$model"_resume.out 2>&1 &


# model 208
# model 108 (train on slurm cluster)
# model 95
# model 82 aug
# compare with 75
model=208
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,2,2 --loss=BCE >train_out_v4_model"$model"_resume.out 2>&1 &

# model 209
# model 109 (train on slurm cluster)
# model 96
# model 83 aug
# compare with 82
model=209
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,10,10 --loss=BCE >train_out_v4_model"$model"_resume.out 2>&1 &

# model 210 (train on slurm cluster)
# model 97
# aug model 86
# compare with 82
model=210
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,20,20 --loss=BCE >train_out_v4_model"$model"_resume.out 2>&1 &

############################################################################################################
# draft below
############################################################################################################
