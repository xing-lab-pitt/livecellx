############################################################################################################
# syn underseg training 5300 series) with mild augmentation
############################################################################################################

# model 901
# model 88 resume
# model 75 aug
model=1001
nohup python ../livecellx/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v9" --test_dir="./notebook_results/a549_ccp_vim/test_data_v9"  --source=all --model_version=version_$model --epochs=1000 --kernel_size=1 --batch_size=2 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --loss=BCE > train_out_v9_model"$model"_resume.out 2>&1 &

# model 902
# model 89 resume
# model 76 aug
model=1002
nohup python ../livecellx/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v9" --test_dir="./notebook_results/a549_ccp_vim/test_data_v9"  --source=all --model_version=version_$model --epochs=1000 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE >train_out_v9_model"$model"_resume.out 2>&1 &