#!/bin/bash
#
#SBATCH --job-name=ContinuingModel
#SBATCH --output=ContinuingModel.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH -p gpu

module purge
module load python/intel
module load cuda
module load cudnn
module load tensorflow/python2.7
module load h5py/intel/2.7.0rc2
module load scikit-image/intel/0.12.3
module load scikit-learn/intel/0.18.1




python -u train.py --data_set="flowers" --resume_model="Data/Models/latest_model_1_flowers_temp.ckpt"


