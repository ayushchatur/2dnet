#!/bin/bash

module reset
module list

#export profile_prefix="dlprof --output_path=${SLURM_JOBID} --profile_name=dlpro_${SLURM_NODEID}_rank${SLURM_PROCID} --mode=pytorch -f true --reports=all -y 60 -d 120 --nsys_base_name=nsys_${SLURM_NODEID}_rank${SLURM_PROCID}  --nsys_opts=\"-t osrt,cuda,nvtx,cudnn,cublas\" "


#export profile_prefix="nsys profile -t cuda,nvtx,cudnn,cublas --show-output=true --force-overwrite=true --delay=60 --duration=220 --export=sqlite -o ${SLURM_JOBID}/profile_rank${SLURM_PROCID}_node_${SLURM_NODEID}"
export file="trainers.py"
export CMD="python ${file} --batch ${batch_size} --epochs ${epochs} --retrain ${retrain} --out_dir ${SLURM_JOBID} --amp ${mp} --num_w $num_data_w  --new_load ${new_load} --prune_amt $prune_amt --prune_t $prune_t  --wan $wandb --lr ${lr} --dr ${dr} --distback ${distback} --enable_profile ${enable_profile} --gr_mode ${gr_mode} --gr_backend ${gr_back} --enable_gr=${enable_gr}"
echo "CMD: ${CMD}"

module load EasyBuild/4.6.1
module use $EASYBUILD_INSTALLPATH_MODULES
module load numlib/cuDNN/8.4.1.50-CUDA-11.7.0
#module load system/CUDA/11.7.0

module load Anaconda3
conda init
source ~/.bashrc
conda activate tttt
export infer_command="python ddnet_inference.py --filepath ${SLURM_JOBID} --batch ${batch_size} --epochs ${epochs} --out_dir ${SLURM_JOBID}"
# change base conda env to nightly pytorch
if [ "$enable_gr" = "true" ]; then
  conda activate pytorch_night
else
  conda activate py_13_1_cuda11_7
fi

if [ "$enable_profile" = "true" ];then
  dlprof --output_path=${SLURM_JOBID} --nsys_base_name=nsys_${SLURM_PROCID} --profile_name=dlpro_${SLURM_PROCID} --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn,cublas --cuda-memory-usage=true --kill=none" -f true --reports=all --delay 60 --duration 120 ${CMD}
else
  $CMD
  export MASTER_PORT=$(comm -23 <(seq 20000 65535) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[0-9]\{1,5\}" | sort | uniq) | shuf | head -n 1)
  echo "master port: $MASTER_PORT"
  $infer_command
fi



