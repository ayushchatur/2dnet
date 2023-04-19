#!/bin/bash

module reset
#module load numlib/cuDNN/8.4.1.50-CUDA-11.7.0
#module load system/CUDA/11.7.0
module load Anaconda3
module list
# change base conda env to nightly pytorch
if [ "$enable_gr" = "true" ]; then
  export cond_env="pytorch_night"
else
  export conda_env="py_13_1_cuda11_7"
fi


#export profile_prefix="dlprof --output_path=${SLURM_JOBID} --profile_name=dlpro_${SLURM_NODEID}_rank${SLURM_PROCID} --mode=pytorch -f true --reports=all -y 60 -d 120 --nsys_base_name=nsys_${SLURM_NODEID}_rank${SLURM_PROCID}  --nsys_opts=\"-t osrt,cuda,nvtx,cudnn,cublas\" "


#export profile_prefix="nsys profile -t cuda,nvtx,cudnn,cublas --show-output=true --force-overwrite=true --delay=60 --duration=220 --export=sqlite -o ${SLURM_JOBID}/profile_rank${SLURM_PROCID}_node_${SLURM_NODEID}"
#alias nsys=$CUDA_HOME/bin/nsys

export file="trainers.py"
export CMD="conda run -n ${cond_env} python ${file} --batch ${batch_size} --epochs ${epochs} --retrain ${retrain} --out_dir ${SLURM_JOBID} --amp ${mp} --num_w $num_data_w  --new_load ${new_load} --prune_amt $prune_amt --prune_t $prune_t  --wan $wandb --lr ${lr} --dr ${dr} --distback ${distback} --enable_profile ${enable_profile} --gr_mode ${gr_mode} --gr_backend ${gr_back} --enable_gr=${enable_gr} --schedtype ${schedtype}"
echo "CMD: ${CMD}"
if [ "$enable_profile" = "true" ];then
  module load CUDA/11.7.0
  echo "cuda home: ${CUDA_HOME}"
  conda run -n ${conda_env} dlprof --output_path=${SLURM_JOBID} --nsys_base_name=nsys_${SLURM_PROCID} --profile_name=dlpro_${SLURM_PROCID} --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn,cublas,cusparse,mpi, --cuda-memory-usage=true" -f true --reports=all --delay 60 --duration 120 ${CMD}
else
  $CMD
fi



