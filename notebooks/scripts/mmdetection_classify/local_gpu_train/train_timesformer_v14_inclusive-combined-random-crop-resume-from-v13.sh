#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=train_mmaction

#SBATCH --gres=gpu:1
#SBATCH --exclude=g019,g102,g104,g122,g012,g013,g131
# #SBAATCH --nodelist=g004
#SBATCH --cpus-per-task=8

#SBATCH --mem=40G

# partition (queue) declaration
#SBATCH --partition=dept_gpu

# number of requested nodes
#SBATCH --nodes=1

# number of tasks
#SBATCH --ntasks=1

# standard output & error

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     User Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

nvidia-smi -L
PYTHON=python
OUT_DIR=local_outs

export CUDA_VISIBLE_DEVICES=0
model="timesformer-divst_v14_inclusive-combined-random-crop-resume-from-v13"
nohup $PYTHON train.py "configs/config_train_timesformer_divst_v14-inclusive-combined-random-crop-resume-from-v13.py" \
      --resume "/home/ken67/livecellx/notebooks/scripts/mmdetection_classify/work_dirs/timesformer-default-divst-v13-inclusive-with-mitosis-type-combined-random-crop/epoch_30.pth" \
      > $OUT_DIR/train_out_model"$model".out 2>&1&