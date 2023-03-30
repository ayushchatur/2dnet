#!/bin/bash

#export WORLD_SIZE=8

export numprofileranks=4
export min=0
export max=$(($WORLD_SIZE-1))
echo "max: $max"
export range=$(($max - $min + 1))
echo "range : $range"

export arr=$(shuf -i $min-$max -n $numprofileranks)


# change file variable based on new data loader flag
if [ "$new_load" = "true" ];then
  export file="sparse_ddnet.py"
else
  export file="sparse_ddnet_old_dl.py"
fi

# change base container image to graph is supported in pytorch 2.0
if [ "$pytor" = "ver1" ]; then
  export imagefile=/home/ayushchatur/ondemand/dev/pytorch_22.04.sif
else
  export imagefile=/home/ayushchatur/ondemand/dev/pytorch_2.sif
  export CMD="${CMD} --gr_mode $graph_mode --gr_backend $gr_back"
fi

# profiling prefix
if [ "$enable_profile" = "true" ]; then
  # shellcheck disable=SC1072
  if [ "$WORLD_SIZE" -ge 4]; then
    for i in $arr
    do
      echo "enabling profiling global rank: $i"
      export CMD="${profile_prefix} ${CMD}"
    done
  else
    export CMD="${profile_prefix} ${CMD}"
  fi
else
  echo "no profiling"
fi


if [ "$enable_profile" = "true" ] && [ "${new_load}" == "true"]; then
  export file="sparse_ddnet_pro.py"
elif [ "$enable_profile" = "true" ]; then
  export file="sparse_ddnet_pp.py"
fi

export CMD="${CMD} python ${file} --batch ${batch_size} --epochs ${epochs} --retrain ${retrain} --out_dir $SLURM_JOBID --amp ${mp} --num_w $num_data_w --prune_amt $prune_amt --prune_t $prune_t  --wan $wandb --lr ${lr} --dr ${dr} --distback ${distback}"
echo "procid: ${SLURM_PROCID} cmd: $CMD"
$BASE exec --nv --writable-tmpfs --bind=/projects/synergy_lab/garvit217,/cm/shared:/cm/shared,/localscratch $imagefile $CMD
