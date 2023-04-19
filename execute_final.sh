#!/bin/bash

module reset
module list
module load EasyBuild/4.6.1
module use $EASYBUILD_INSTALLPATH_MODULES

module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

if [[ "$SLURM_JOB_PARTITION" =~ "v100" ]]; then
  module load NCCL/2.10.3-GCCcore-11.2.0-CUDA-11.4.1
else
 module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
fi



alias nsys=$CUDA_HOME/bin/nsys
#module load numlib/cuDNN/8.4.1.50-CUDA-11.7.0
#module load system/CUDA/11.7.0

module load Anaconda3
conda init
source ~/.bashrc
conda activate tttt

# change base conda env to nightly pytorch
if [ "$enable_gr" = "true" ]; then
  conda activate pytorch_night
else
  conda activate py_13_1_cuda11_7
fi


#export profile_prefix="dlprof --output_path=${SLURM_JOBID} --profile_name=dlpro_${SLURM_NODEID}_rank${SLURM_PROCID} --mode=pytorch -f true --reports=all -y 60 -d 120 --nsys_base_name=nsys_${SLURM_NODEID}_rank${SLURM_PROCID}  --nsys_opts=\"-t osrt,cuda,nvtx,cudnn,cublas\" "


#export profile_prefix="nsys profile -t cuda,nvtx,cudnn,cublas --show-output=true --force-overwrite=true --delay=60 --duration=220 --export=sqlite -o ${SLURM_JOBID}/profile_rank${SLURM_PROCID}_node_${SLURM_NODEID}"
alias nsys=$CUDA_HOME/bin/nsys
if [ "$SLURM_PROCID" == "0" ];then
  echo "getting system info"
  conda info
  echo "cuda home: ${CUDA_HOME}"
  python --version
  nsys --version
  nvcc --version
fi

export file="trainers.py"
export CMD="python ${file} --batch ${batch_size} --epochs ${epochs} --retrain ${retrain} --out_dir ${SLURM_JOBID} --amp ${mp} --num_w $num_data_w  --new_load ${new_load} --prune_amt $prune_amt --prune_t $prune_t  --wan $wandb --lr ${lr} --dr ${dr} --distback ${distback} --enable_profile ${enable_profile} --gr_mode ${gr_mode} --gr_backend ${gr_back} --enable_gr=${enable_gr} --schedtype ${schedtype}"
echo "CMD: ${CMD}"
if [ "$enable_profile" = "true" ];then

  dlprof --output_path=${SLURM_JOBID} --nsys_base_name=nsys_${SLURM_PROCID} --profile_name=dlpro_${SLURM_PROCID} --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn,cublas,cusparse,mpi, --cuda-memory-usage=true --kill=none" -f true --reports=all --delay 60 --duration 120 ${CMD}
else
  $CMD
fi



