python train_classify_ViT_classifier_v14_lightning.py \
    --batch_size=128 \
    --start_frame_idx=1 --end_frame_idx=5 \
    --frame-type combined \
    --model_version frame_1-5-combined 
    


python train_classify_ViT_classifier_v14_lightning.py\
    --batch_size=128\
    --start_frame_idx=1 --end_frame_idx=5\
    --frame-type mask\
    --model_version "frame_1-5-mask"\ 
    