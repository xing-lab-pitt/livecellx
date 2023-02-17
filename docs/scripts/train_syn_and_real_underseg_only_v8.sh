############################################################################################################
# syn underseg training 5300 series) with mild augmentation
############################################################################################################

# model 901
# model 88 resume
# model 75 aug
model=901
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --loss=BCE > train_out_v7_model"$model"_resume.out 2>&1 &

# model 902
# model 89 resume
# model 76 aug
model=902
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE >train_out_v7_model"$model"_resume.out 2>&1 &

# model 903
# model 90 resume
# model 77 aug
model=903
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_duplicate --loss=BCE >train_out_v7_model"$model"_resume.out 2>&1 &

# model 904
# model 91
# model 78 aug
# compare with 77: exclude bg
model=904
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_duplicate --loss=BCE --exclude_raw_input_bg  >train_out_v7_model"$model"_resume.out 2>&1 &

# model 905
# model 92
# model 79 aug
# compare with 76: exclude bg
model=905
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --apply_gt_seg_edt --loss=MSE --exclude_raw_input_bg >train_out_v7_model"$model"_resume.out 2>&1 &

# model 906
# model 93
# model 80 aug
# compare with 75: exclude bg
model=906
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --loss=BCE  >train_out_v7_model"$model"_resume.out 2>&1 &

# model 907
# model 94
# model 81 aug
# compare with 79&76: use raw_duplicate instead of raw_aug_duplicate
model=907
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE  >train_out_v7_model"$model"_resume.out 2>&1 &

# model 908 (train on slurm cluster)
# model 95
# model 82 aug
# compare with 75
model=908
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,2,2  --loss=BCE >train_out_v7_model"$model"_resume.out 2>&1 &

# model 909 (train on slurm cluster)
# model 96
# model 83 aug
# compare with 82
model=909
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,10,10  --loss=BCE > train_out_v7_model"$model"_resume.out 2>&1 &

# model 910 (train on slurm cluster)
# model 97
# aug model 86
# compare with 82
model=910
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,20,20 --loss=BCE > train_out_v7_model"$model"_resume.out 2>&1 &

# model 911 (train on slurm cluster)
# model 98
# model 84 aug
# compare with 81
model=911
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,2,2 >train_out_v7_model"$model"_resume.out 2>&1 &

# model 912 (train on slurm cluster)
# model 99
# model 85 aug
# compare with 84
model=912
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,10,10 >train_out_v7_model"$model"_resume.out 2>&1 &

# model 913 (train on slurm cluster)
# model 100
# model 87 aug (not yet trained)
model=913
nohup python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v7" --test_dir="./notebook_results/a549_ccp_vim/test_data_v7"  --source=underseg-all --model_version=version_$model --epochs=50 --kernel_size=1 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_duplicate --apply_gt_seg_edt --loss=MSE --class-weights=1,20,20 >train_out_v7_model"$model"_resume.out 2>&1 &