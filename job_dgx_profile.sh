#!/bin/bash

nvidia-smi
export batch_size=1
export epochs=50
export retrain=3

export MASTER_ADDR=0.0.0.0
export MASTER_PORT=8882
export SLURM_PROCID=0
export append=$RANDOM
touch output_${append}.out
echo "output_${append}.out"
date
time docker run --env MASTER_ADDR --env MASTER_PORT --env SLURM_PROCID --env append --rm --gpus 0,1 -w /projects/synergy_lab/garvit217/enhancement_data/2dnet -v ~/Documents/2dnet:/projects/synergy_lab/garvit217/enhancement_data/ -v ~/Documents/2dnet/2dnet:/code nvcr.io/nvidia/pytorch:21.12-py3 dlprof --output_path=amp_w16 --profile_name=amp_16 --dump_model_data=true --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn,cublas --cuda-memory-usage=true --gpuctxsw=true " -f true --reports=all --delay 60 --duration 60 python sparse_ddnet_pp.py -n 1 -g 1 --batch $batch_size --epochs $epochs --retrain $retrain > output_$append.out &
