#!/bin/bash
#SBATCH --job-name=ddnet
#SBATCH --nodes=4               # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node= gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16384                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             #GPU per node
#SBATCH --partition=a100_normal_q # slurm partition
#SBATCH --time=1:30:00          # time limit
#SBATCH -A HPCBIGDATA2           # account name

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$(comm -23 <(seq 20000 65535) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[0-9]\{1,5\}" | sort | uniq) | shuf | head -n 1)
echo "master port: $MASTER_PORT"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "slurm job: $SLURM_JOBID"
#expor job_id=$SLURM_JOBID
mkdir -p $SLURM_JOBID;cd $SLURM_JOBID
#cp ../sparse_ddnet.py .
#cp ../sparse_ddnet_pp.py .

mkdir -p ./loss/
mkdir -p ./reconstructed_images/
mkdir -p ./reconstructed_images/val
mkdir -p ./reconstructed_images/test
mkdir -p ./visualize
mkdir -p ./visualize/val/
mkdir -p ./visualize/val/mapped/
mkdir -p ./visualize/val/diff_target_out/
mkdir -p ./visualize/val/diff_target_in/
mkdir -p ./visualize/val/input/
mkdir -p ./visualize/val/target/
mkdir -p ./visualize/test/
mkdir -p ./visualize/test/mapped/
mkdir -p ./visualize/test/diff_target_out/
mkdir -p ./visualize/test/diff_target_in/
mkdir -p ./visualize/test/input/
mkdir -p ./visualize/test/target/
cd ..
echo "tmpfs for this job at $TMPDIR"
echo "Staging full data per node"

#cd $TMPDIR/tmpfs
export dest_dir=$TMPDIR/tmpfs
cp -r /projects/synergy_lab/garvit217/enhancement_data $dest_dir
echo "Staged full data per node on $dest_dir"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURMTMPDIR=$SLURMTMPDIR"

echo "working directory = "$SLURM_SUBMIT_DIR

# module load  apps  site/infer/easybuild/setup
# module load PyTorch/1.7.1-fosscuda-2020b
module reset
#module load Anaconda3 cuda-latest/toolkit/11.2.0 cuda-latest/nsight
module list
#nvidia-smi -L

# cd ~
#conda activate test
# cd -
#cd /projects/synergy_lab/garvit*/sc*/batch_16*

if [[ "$SLURM_JOB_PARTITION" == *"dgx"* ]]; then
  module load containers/apptainer
  export BASE="apptainer"
else
  module load containers/singularity
  export BASE="singularity"
fi


# export gpu=$(nvidia-smi -L | wc -l)

echo "current dir: $PWD"
chmod 755 * -R


export profile_prefix="dlprof --output_path=${SLURM_JOBID}_profile --profile_name=${SLURM_JOBID}_profile --dump_model_data=true --mode=pytorch --nsys_opts=\"-t osrt,cuda,nvtx,cudnn,cublas --cuda-memory-usage=true --gpuctxsw=true\" -f true --reports=all --delay 180 --duration 60"

if [ "$enable_profile" = "true" ]; then
#  if [ "" ]
  export file="sparse_ddnet_pro.py"
  export CMD="${profile_prefix} ${CMD}"
else
  if [ "$new_load" = "true" ];then
    export file="sparse_ddnet.py"
  else
    export file="sparse_ddnet_old_dl.py"
  fi

fi



export CMD="${CMD} python ${file} --batch ${batch_size} --epochs ${epochs} --retrain ${retrain} --out_dir $SLURM_JOBID --amp ${mp} --num_w $num_data_w --prune_amt $prune_amt --prune_t $prune_t  --wan $wandb --lr ${lr} --dr ${dr}"





if [ "$pytor" = "ver1" ]; then
  export imagefile=/home/ayushchatur/ondemand/dev/pytorch_22.04.sif
else
  export imagefile=/home/ayushchatur/ondemand/dev/pytorch_2.sif
  export CMD="${CMD} --gr_mode $graph_mode --gr_backend $gr_back"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
#BIND_CMD="./bind.sh --cpu=./cpu_bind.sh --mem=./cpu_bind.sh"

echo "cmd: $CMD"

if [ "$inferonly" = "false" ]; then

  srun $BASE exec --nv --writable-tmpfs --bind=/projects/synergy_lab/garvit217,/cm/shared:/cm/shared $imagefile $CMD
  $BASE exec --nv --writable-tmpfs --bind=/projects/synergy_lab/garvit217,/cm/shared:/cm/shared $imagefile python ddnet_inference.py --filepath $SLURM_JOBID --out_dir $SLURM_JOBID --epochs ${epochs} --batch ${batch_size}
else
  $BASE exec --nv --writable-tmpfs --bind=/projects/synergy_lab/garvit217,/cm/shared:/cm/shared $imagefile python ddnet_inference.py --filepath $1 --out_dir $1 --epochs ${epochs} --batch ${batch_size}

fi
#sgather $TMPDIR/myexec.out myexec.out
