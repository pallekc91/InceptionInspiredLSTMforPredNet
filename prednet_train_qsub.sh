#!/bin/bash -l

#$ -l h_rt=48:00:00   # Specify the hard time limit for the job
#$ -N train_custom_prednet           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -pe omp 4 
#$-l gpus=0.5 
#$-l gpu_c=3.5 -V 
#$-P paralg

python prednet_train.py
