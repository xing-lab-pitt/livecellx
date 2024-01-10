model="timesformer-default-divst-v13-inclusive-all-random-crop"
model_dir="./work_dirs/"$model
video_dir="../../notebook_results/mmaction_train_data_v13-inclusive-corrected/videos"

test_data_meta_path="../../notebook_results/mmaction_train_data_v14-inclusive/CXA_normal_train_test_data_meta.txt"

python classify_mmdetection_mitosis_eval.py \
    --model_dir=$model_dir \
    --out_dir="./work_dirs/eval_results/v13_on_v14_cxa_normal/"$model \
    --config=$model_dir/"config_train_timesformer_divst_v13-inclusive-all-random-crop.py" \
    --checkpoint=$model_dir/"epoch_30.pth" \
    --video_dir=$video_dir \
    --mmaction_data_tsv=$mmaction_data_tsv \
    --test_data_meta_path=$test_data_meta_path \
    --add-random-crop \
    --device="cuda:1"


