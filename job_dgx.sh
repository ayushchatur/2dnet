#!/bin/bash

nvidia-smi
export batch_size=1
export epochs=50
export retrain=3

export MASTER_ADDR=0.0.0.0
export MASTER_PORT=8882
export SLURM_PROCID=0
export append=$RANDOM
touch output_$append.out
python sparse_ddnet.py -n 1 -g 1 --batch $batch_size --epochs $epochs --retrain $retrain > output_$append.out &
