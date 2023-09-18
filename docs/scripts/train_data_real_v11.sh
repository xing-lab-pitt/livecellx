############################################################################################################
# syn underseg training 5300 series) with mild augmentation
############################################################################################################

# model 901
# model 88 resume
# model 75 aug
model=v11_01
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v11" --test_dir="./notebook_results/a549_ccp_vim/test_data_v11"  --source=all --model_version=version_$model --epochs=5000 --kernel_size=1 --batch_size=2 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --loss=BCE > train_out_model"$model"_resume.out 2>&1 &

# model 902
# model 89 resume
# model 76 aug
model=v11_02
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v11" --test_dir="./notebook_results/a549_ccp_vim/test_data_v11"  --source=all --model_version=version_$model --epochs=1000 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE \
        --model_ckpt=/net/capricorn/home/xing/ken67/LiveCellTracker-dev/notebooks/lightning_logs/version_v11_02/checkpoints/last.ckpt \
        >train_out_model"$model"_resume.out 2>&1 &