python eval.py\
   --model_dir ./ViT_workdirs/ViT_lightning_logs/4090-frame_1-5-combined-batchsize-32\
   --frame-type combined\
   # --ckpt ./ViT_workdirs/ViT_lightning_logs/frame_1-5-combined/checkpoints/epoch=39-step=60319.ckpt\


python eval.py\
   --model_dir ./ViT_workdirs/ViT_lightning_logs/4090-frame_1-5-combined-batchsize-64\
   --frame-type combined

python eval.py\
   --model_dir ./ViT_workdirs/ViT_lightning_logs/4090-frame_1-5-combined-batchsize-128\
   --frame-type combined

python eval.py\
   --model_dir ./ViT_workdirs/ViT_lightning_logs/4090-frame_1-5-combined-batchsize-32\
   --frame-type combined\
   --affine-aug\
   # --ckpt ./ViT_workdirs/ViT_lightning_logs/frame_1-5-combined/checkpoints/epoch=39-step=60319.ckpt\


python eval.py\
   --model_dir ./ViT_workdirs/ViT_lightning_logs/4090-frame_1-5-combined-batchsize-64\
   --frame-type combined\
   --affine-aug\

python eval.py\
   --model_dir ./ViT_workdirs/ViT_lightning_logs/4090-frame_1-5-combined-batchsize-128\
   --frame-type combined\
   --affine-aug\

python eval.py\
   --model_dir ./ViT_workdirs/ViT_lightning_logs/4090-frame_1-5-combined-batchsize-128\
   --frame-type combined\
   --affine-aug\
   --save_dir_suffix="-batchsize-128"\
   --batch_size=128\

python eval.py\
   --model_dir ./ViT_workdirs/ViT_lightning_logs/resnet50-frame_1-5-combined\
   --frame-type combined\
   --batch_size=128\
   --model=resnet50\


python eval.py\
   --model_dir ./ViT_workdirs/ViT_lightning_logs/resnet50-frame_1-5-mask\
   --frame-type  mask\
   --batch_size=128\
   --model=resnet50\


python eval.py\
   --model_dir ./ViT_workdirs/ViT_lightning_logs/resnet50-frame_all-combined\
   --frame-type combined\
   --batch_size=128\
   --model=resnet50\


python eval.py\
   --model_dir ./ViT_workdirs/ViT_lightning_logs/resnet50-frame_all-mask\
   --frame-type mask\
   --batch_size=128\
   --model=resnet50\
