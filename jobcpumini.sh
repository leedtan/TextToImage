#!/bin/bash
#
#SBATCH --job-name=NewMini1
#SBATCH --output=NewMini1.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB

module purge
module load python/intel
module load tensorflow/python2.7
module load h5py/intel/2.7.0rc2
module load scikit-image/intel/0.12.3
module load scikit-learn/intel/0.18.1


python -u trainMiniBatchDisc.py --data_set="flowers"


