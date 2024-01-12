# python train_classify_ViT_classifier_v14_lightning.py \
#     --start_frame_idx=1 --end_frame_idx=5 \
#     --model_version=frame_1-5-combined \ 
#     --frame-type=combined


python train_classify_ViT_classifier_v14_lightning.py\
    --start_frame_idx=1 --end_frame_idx=5\
    --frame-type mask\
    --model_version "frame_1-5-mask"\ 
    