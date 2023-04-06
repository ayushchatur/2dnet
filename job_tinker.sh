#!/bin/bash
#SBATCH --job-name=ddnet
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16384                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus-per-node 1             #GPU per node
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
set | grep SLURM | while read line; do echo "# $line"; done > slurm.txt
env | grep -i -v slurm | sort > env.txt
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

module reset
module list

if [[ "$SLURM_JOB_PARTITION" == *"dgx"* ]]; then
  module load containers/apptainer
  export BASE="apptainer"
else
  module load containers/singularity
  export BASE="singularity"
fi
echo "BASE: ${BASE}"

echo "current dir: $PWD"
chmod 755 * -R

#: "${NEXP:=1}"
#for _experiment_index in $(seq 1 "${NEXP}"); do
#    (
#        echo "Beginning trial ${_experiment_index} of ${NEXP}"
#        srun --wait=120 --kill-on-bad-exit=0 --cpu-bind=none --ntasks "${WORLD_SIZE}"  ./execute_final.sh
#    )
#done
#wait


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
  export imagefile=/home/ayushchatur/ondemand/dev/pytorch2.sif
  export CMD="${CMD} --gr_mode $graph_mode --gr_backend $gr_back"
fi


#export profile_prefix="dlprof --output_path=${SLURM_JOBID} --profile_name=dlpro_${SLURM_NODEID}_rank${SLURM_PROCID} --mode=pytorch -f true --reports=all -y 60 -d 120 --nsys_base_name=nsys_${SLURM_NODEID}_rank${SLURM_PROCID}  --nsys_opts=\"-t osrt,cuda,nvtx,cudnn,cublas\" "

#export profile_prefix="nsys profile -t cuda,nvtx,cudnn,cublas --show-output=true --force-overwrite=true --delay=60 --duration=220 --export=sqlite -o ${SLURM_JOBID}/profile_rank${SLURM_PROCID}_node_${SLURM_NODEID}"


echo "procid: ${SLURM_PROCID}"
#echo "cmd: $CMD"
echo "final command: $BASE exec --nv --writable-tmpfs --bind=/projects/synergy_lab/garvit217,/cm/shared:/cm/shared,$TMPFS $imagefile $profile_prefix $CMD"


if [ "$enable_profile" = "true" ]; then

  export imagefile=/home/ayushchatur/ondemand/dev/pytorch_21.12.sif
  $BASE exec --nv --writable-tmpfs --bind=/projects/synergy_lab/garvit217,/cm/shared:/cm/shared,$TMPFS $imagefile dlprof --output_path=${SLURM_JOBID} --profile_name=dlpro_{SLURM_PROCID} --dump_model_data=true --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn,cublas --cuda-memory-usage=true --gpuctxsw=true " -f true --reports=all --delay 120 --duration 60 ${CMD}

elif [ "$inferonly" = "false" ]; then

  $BASE exec --nv --writable-tmpfs --bind=/projects/synergy_lab/garvit217,/cm/shared:/cm/shared,$TMPFS $imagefile $profile_prefix $CMD
  $BASE exec --nv --writable-tmpfs --bind=/projects/synergy_lab/garvit217,/cm/shared:/cm/shared,$TMPFS $imagefile python ddnet_inference.py --filepath $SLURM_JOBID --out_dir $SLURM_JOBID --epochs ${epochs} --batch ${batch_size} --lr ${lr} --dr ${dr}
else
  $BASE exec --nv --writable-tmpfs --bind=/projects/synergy_lab/garvit217,/cm/shared:/cm/shared,$TMPFS $imagefile python ddnet_inference.py --filepath $1 --out_dir $1 --epochs ${epochs} --batch ${batch_size} --lr ${lr} --dr ${dr}
fi
