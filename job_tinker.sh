#!/bin/bash

#SBATCH --job-name=ddnet
#SBATCH --partition=a100_normal_q
#SBATCH --time=16:00:00
#SBATCH -A HPCBIGDATA2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --propagate=STACK
#SBATCH --dependency=259619
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=8888
#export WORLD_SIZE=4
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
# echo "${SLURM_NODELIST:7:1}"
# echo "${SLURM_NODELIST:8:3}"
# echo "MASTERs_ADDR="${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:3}
##only for tinkercliffs
if [ ${SLURM_NODELIST:6:1} == "[" ]; then
    echo "MASTER_ADDR="${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:3}
module reset
    export MASTER_ADDR=${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:3}
else
    echo "MASTER_ADDR="${SLURM_NODELIST}
    export MASTER_ADDR=${SLURM_NODELIST}
fi

mkdir -p ./loss
mkdir -p ./reconstructed_images
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


echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

# module load  apps  site/infer/easybuild/setup
# module load PyTorch/1.7.1-fosscuda-2020b
module reset
module load Anaconda3 cuda-latest/toolkit/11.2.0 cuda-latest/nsight
module list
nvidia-smi
export batch_size=1
export epochs=50
export retrain=1
echo "batch : $batch_size"

echo "retrain : $retrain"

echo "epochs : $epochs"
# cd ~
source activate test
# cd -
#cd /projects/synergy_lab/garvit*/sc*/batch_16*
imagefile=/projects/arcsingularity/AMD/ood-jupyter-pytorch_21.12.sif
module load containers/singularity
### the command to run
#nsys profile -t cuda,osrt,nvtx,cudnn,cublas -y 60 -d 300 -o baseline -f true -w true python train_main2_jy.py -n 1 -g 4 --batch $batch_size --epochs $epochs
#time python sparse_ddnet.py -n 1 -g 1 --batch $batch_size --epochs $epochs --retrain $retrain

singularity exec --nv --writable-tmpfs --bind=${TMPDIR},/cm/shared:/cm/shared,/projects:/projects $imagefile python sparse_ddnet.py -n 1 -g 1 --batch $batch_size --epochs $epochs --retrain $retrain
#sbatch --nodes=1 --ntasks-per-node=8 --gres=gpu:1 --partition=normal_q -t 1600:00 ./batch_job.sh
