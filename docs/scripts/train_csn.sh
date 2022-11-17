python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v1" --kernel_size=7 --batch_size=2 >train_out_0.out 2>&1 &

python ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v1" --kernel_size=1 --batch_size=2 >train_out_1.out 2>&1 &