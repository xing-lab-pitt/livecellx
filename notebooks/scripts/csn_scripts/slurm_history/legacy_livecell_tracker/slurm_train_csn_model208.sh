#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# job name
#SBATCH --job-name=train_csn

#SBATCH --gres=gpu:1
#SBATCH --exclude=g019,g102,g104,g122,g012,g013,g131
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
PYTHON=/net/capricorn/home/xing/ken67/.conda/envs/livecell-tracker/bin/python

# model 208
# model 108 (train on slurm cluster)
# model 95
# model 82 aug
# compare with 75
model=208
$PYTHON ../livecell_tracker/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v4" --model_version=version_$model --epochs=2000 --kernel_size=1 --batch_size=2 --degrees=180 --translation=0.5 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --class-weights=1,2,2 --loss=BCE