
model="tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-v13-drop-div-combined-clipLen=2-trainClipNum=3-valClipNum=3"
model_dir="./work_dirs/"$model
video_dir="../../notebook_results/mmaction_train_data_v13-inclusive-corrected/videos"

test_data_meta_path="../../notebook_results/mmaction_train_data_v13-inclusive-corrected/test_data.txt"

python classify_mmdetection_mitosis_eval.py \
    --model_dir=$model_dir \
    --out_dir="./work_dirs/eval_results/"$model \
    --config=$model_dir/"config_train_v13_div_cliplen=2_clipnum=3-combined.py" \
    --checkpoint=$model_dir/"best_acc_top1_epoch_27.pth" \
    --video_dir=$video_dir \
    --mmaction_data_tsv=$mmaction_data_tsv \
    --test_data_meta_path=$test_data_meta_path \
    --device="cuda:1" \
    --add-random-crop \
    --is-tsn \
    --no-wrong-video


