#!/bin/bash
#SBATCH --job-name=ddnet
#SBATCH --nodes 1
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8384                # total memory per node (4 GB per cpu-core is default)
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1             #GPU per node
#SBATCH --partition=v100_normal_q # slurm partition
#SBATCH --time=24:30:00          # time limit
#SBATCH -A HPCBIGDATA2           # account name

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes


export MASTER_PORT=$(comm -23 <(seq 20000 65535) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[0-9]\{1,5\}" | sort | uniq) | shuf | head -n 1)
echo "master port: $MASTER_PORT"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE=$WORLD_SIZE"
echo "slurm job: $SLURM_JOBID"
#expor job_id=$SLURM_JOBID
mkdir -p $SLURM_JOBID;cd $SLURM_JOBID
set | grep SLURM | while read line; do echo "# $line"; done > slurm.txt
env | grep -i -v slurm | sort > env.txt
#cp ../sparse_ddnet.py .
#cp ../sparse_ddnet_pp.py .
export batch_size=4
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
#cp -r /projects/synergy_lab/garvit217/enhancement_data $dest_dir
echo "Staged full data per node on $dest_dir"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURMTMPDIR=$SLURMTMPDIR"

echo "working directory = "$SLURM_SUBMIT_DIR
echo "custom module path: $EASYBUILD_INSTALLPATH_MODULES"


echo "current dir: $PWD"
#chmod 755 * -R


: "${NEXP:=1}"

module reset
module load EasyBuild/4.6.1
module use $EASYBUILD_INSTALLPATH_MODULES


module load Anaconda3
if [ "$enable_gr" = "true" ]; then
  export conda_env="pytorch_night"
#  conda activate pytorch_night
else
  export conda_env="py_13_1_cuda11_7"
#  conda activate py_13_1_cuda11_7
fi


module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load NCCL/2.10.3-GCCcore-11.2.0-CUDA-11.4.1


module list

alias nsys=$CUDA_HOME/bin/nsys
#module load numlib/cuDNN/8.4.1.50-CUDA-11.7.0
#module load system/CUDA/11.7.0

echo "getting system info"
conda info
echo "cuda home: ${CUDA_HOME}"
python --version
nsys --version
nvcc --version
whereis nsys
whereis python
conda run -n ${conda_env} python -c "import torch;print(f'cuDNN version: {torch.backends.cudnn.version()}')"
conda run -n ${conda_env} python -c "import torch;print(f' NCCL version: {torch.cuda.nccl.version()}')"
export infer_command="conda run -n ${conda_env} python ddnet_inference.py --filepath ${SLURM_JOBID} --batch ${batch_size} --epochs ${epochs} --out_dir ${SLURM_JOBID}"


if [  "$inferonly"  == "true" ]; then
  export filepath=$1
  python ddnet_inference.py --filepath ${filepath} --batch ${batch_size} --epochs ${epochs} --out_dir ${filepath}
else
  for _experiment_index in $(seq 1 "${NEXP}"); do
    (
  	echo "Beginning trial ${_experiment_index} of ${NEXP}"
  	srun --wait=120 --kill-on-bad-exit=0 --cpu-bind=none ./execute_final.sh
    )
  done
  wait
  $infer_command
fi
