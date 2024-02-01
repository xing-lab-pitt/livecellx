#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=train_csn

#SBATCH --gres=gpu:1
#SBATCH --exclude=g019,g102,g104,g122,g012,g013,g131,g011
#SBAATCH --nodelist=g104
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


# node
echo
echo $SLURM_JOB_NODELIST
echo

# check GPU
nvidia-smi -L
PYTHON=python

export CUDA_VISIBLE_DEVICES=1
nohup $PYTHON train_classify_ViT_classifier_v14_lightning.py\
    --batch_size=128\
    --start_frame_idx=1\
    --end_frame_idx=5\
    --frame-type combined\
    --model_version "4090-frame_1-5-combined-batchsize-128"\
    --max-epochs 100\
    1> outs/4090-frame_1-5-combined-batchsize-128.out 2>&1&\
    