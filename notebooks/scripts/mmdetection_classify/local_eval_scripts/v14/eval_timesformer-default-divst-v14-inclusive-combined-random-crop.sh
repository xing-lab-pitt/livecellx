
model="timesformer-default-divst-v14-inclusive-combined-random-crop"
model_dir="./work_dirs/"$model
video_dir="../../notebook_results/mmaction_train_data_v14-inclusive/videos"

test_data_meta_path="../../notebook_results/mmaction_train_data_v14-inclusive/CXA_normal_train_test_data_meta.txt"
# test_data_meta_path="../../notebook_results/mmaction_train_data_v14-inclusive/test_data.txt"


python classify_mmdetection_mitosis_eval.py \
    --model_dir=$model_dir \
    --out_dir="./work_dirs/eval_results/v14/cxa_normal_only/"$model \
    --checkpoint=$model_dir/"epoch_35.pth" \
    --video_dir=$video_dir \
    --mmaction_data_tsv=$mmaction_data_tsv \
    --test_data_meta_path=$test_data_meta_path \
    --add-random-crop \
    --device="cuda:1"


