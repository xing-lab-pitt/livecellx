nvidia-smi -L
PYTHON=python
OUT_DIR=local_outs

export CUDA_VISIBLE_DEVICES=0
model="train_train_timesformer_divst_v10-corrected-combined-random-crop-sampling-sequantial"
nohup $PYTHON train.py "configs/config_train_timesformer_divst_v10-corrected-combined-random-crop-sampling-sequantial.py" --resume auto > $OUT_DIR/train_out_model"$model".out 2>&1&