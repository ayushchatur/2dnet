#!/bin/bash

nvidia-smi

export MASTER_ADDR=0.0.0.0
export MASTER_PORT=8882
export SLURM_PROCID=0
export WORLD_SIZE=1
export append=$RANDOM
touch output_${append}.out
echo "output_${append}.out"
export file="sparse_ddnet_old_dl.py"
export CMD="python sparse_ddnet_old_dl.py  --batch ${batch_size} --epochs ${epochs} --retrain ${retrain} --amp ${mp} --num_w $num_data_w --prune_amt $prune_amt --prune_t $prune_t  --wan $wandb --lr ${lr} --dr ${dr} --distback ${distback}"
date
time docker run --env MASTER_ADDR --env WORLD_SIZE --env MASTER_PORT --env SLURM_PROCID --env append --rm --gpus 3,4 -w /code -v /run/user/1001/2dnet:/projects/synergy_lab/garvit217/enhancement_data/ -v $PWD:/code nvcr.io/nvidia/pytorch:21.12-py3 dlprof --output_path=data_new_load_dense --profile_name=data_new_load_dense2 --dump_model_data=true --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn,cublas --cuda-memory-usage=true --gpuctxsw=true " -f true --reports=all --delay 120 --duration 60 ${CMD} > output_$append.out 2>&1 &
