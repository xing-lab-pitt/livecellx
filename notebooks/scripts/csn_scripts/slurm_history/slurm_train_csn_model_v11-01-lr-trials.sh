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
PYTHON=/net/capricorn/home/xing/ken67/.conda/envs/livecell-tracker/bin/python

model=v11_01-lr=0.001
$PYTHON ../livecellx/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v11" --test_dir="./notebook_results/a549_ccp_vim/test_data_v11"  --source=all --model_version=version_$model --epochs=1000 --kernel_size=1 --batch_size=2 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --loss=BCE --lr=0.001> train_out_model_"$model"_resume.out 2>&1

model=v11_01-lr=0.0001
$PYTHON ../livecellx/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v11" --test_dir="./notebook_results/a549_ccp_vim/test_data_v11"  --source=all --model_version=version_$model --epochs=1000 --kernel_size=1 --batch_size=2 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --loss=BCE --lr=0.0001> train_out_model_"$model"_resume.out 2>&1

model=v11_01-lr=0.1
$PYTHON ../livecellx/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v11" --test_dir="./notebook_results/a549_ccp_vim/test_data_v11"  --source=all --model_version=version_$model --epochs=1000 --kernel_size=1 --batch_size=2 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --loss=BCE --lr=0.1> train_out_model_"$model"_resume.out 2>&1

model=v11_01-lr=0.01
$PYTHON ../livecellx/model_zoo/segmentation/train_csn.py --train_dir="./notebook_results/a549_ccp_vim/train_data_v11" --test_dir="./notebook_results/a549_ccp_vim/test_data_v11"  --source=all --model_version=version_$model --epochs=1000 --kernel_size=1 --batch_size=2 --batch_size=2 --degrees=30 --translation=0.3 --aug_scale="0.5,2" --input_type=raw_aug_duplicate --loss=BCE --lr=0.01> train_out_model_"$model"_resume.out 2>&1