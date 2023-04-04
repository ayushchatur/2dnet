#!/bin/bash

#export WORLD_SIZE=8

export numprofileranks=4
export min=0
export max=$(($WORLD_SIZE-1))
echo "max: $max"
export range=$(($max - $min + 1))
echo "range : $range"

export arr=$(shuf -i $min-$max -n $numprofileranks)


if [ "$enable_profile" = "true" ];then
  if [ "$new_load" = "true" ]; then
    export file="sparse_ddnet_pro.py"
  else
    export file="sparse_ddnet_old_dl_profile.py"
  fi
else
    if [ "$new_load" = "true" ]; then
    export file="sparse_ddnet.py"
  else
    export file="sparse_ddnet_old_dl.py"
  fi
fi


export CMD="python ${file} --batch ${batch_size} --epochs ${epochs} --retrain ${retrain} --out_dir $SLURM_JOBID --amp ${mp} --num_w $num_data_w --prune_amt $prune_amt --prune_t $prune_t  --wan $wandb --lr ${lr} --dr ${dr} --distback ${distback}"


# change base container image to graph is supported in pytorch 2.0
if [ "$pytor" = "ver1" ]; then
  export imagefile=/home/ayushchatur/ondemand/dev/pytorch_22.04.sif
else
  export imagefile=/home/ayushchatur/ondemand/dev/pytorch_2.sif
  export CMD="${CMD} --gr_mode $graph_mode --gr_backend $gr_back"
fi


export profile_prefix="nsys profile -o ${SLURM_JOBID}/profile_${SLURM_NODEID}_rank${SLURM_PROCID} --sample cpu -f true --export sqlite --trace=osrt,cuda,nvtx,cudnn,cublas,cusparse,cusparse-verbose,mpi --cuda-memory-usage=true --gpuctxsw=true --cudabacktrace=all --stats=true --stop-on-exit true --capture-range cudaProfilerApi --capture-range-end stop --kill none"


# profiling prefix

if [ "$enable_profile" = "true" ]; then
  export CMD="${profile_prefix} ${CMD}"
fi

echo "procid: ${SLURM_PROCID} cmd: $CMD"
$BASE exec --nv --writable-tmpfs --bind=/projects/synergy_lab/garvit217,/cm/shared:/cm/shared,$TMPFS $imagefile $CMD
