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


export CUDA_VISIBLE_DEVICES=1
python train_classify_ViT_classifier_v14_lightning.py\
    --batch_size=128\
    --start_frame_idx=0 --end_frame_idx=5\
    --frame-type mask\
    --model_version "frame_0-5-mask"\ 