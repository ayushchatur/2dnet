# 2D-DDNet with VGG Loss and architecture aware sparse optimizations 
2D-DDNet with optimizations 

### instructions on running at ARC ( or similar system with Slurm and Singularity)
- Read Access to "/projects/synergy_lab/garvit217/enhancement_data/"
- Pytorch container [22.04](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)

#### Pulling container at ARC 

Run below commands: 
```sh
module load containers/singulartiy
singularity pull pytorch22.04.sif docker://nvcr.io/nvidia/pytorch:22.04-py3
```
> Other versions of the container can also be used, but driver compatibility needs to be checked with the driver version on GPU nodes at ARC. Carefully verify from [documentation](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)
the follow: 
- git checkout
- cd 2dnet
- source params.sh
- sbatch job_tinker.sh
- output will be generated in slurm-<jobid>.out
  
### Instructions on profiling 
- source params.sh
- export enable_profile="true"
- sbatch job_tinker.sh

> Read carefully the comments in [job_tinker.sh](./job_tinker.sh)

Provide access to all users to Python file 
``` sh 
  chmod 766 sparse_ddnet.py
  chmod 766 trainers.py
 ```

### DDNet baseline hyper-parameters
- batch size: 1
- learning rate: 0.0001
- epochs: 50
- decay rate: 0.95

### Enabling Architecture-Aware Optimizations: 
- Mixed precision:
```export mp=true```
- DoLL Data Loader for Small Datasets
```export new_load=true```
- Graph Optimizations:
```export enable_gr="false"
#to enable graph change pytorch version above otherwise, the below two parameters won't be respected

export gr_mode="reduce-overhead"
export gr_back="aot-eager"
```
### Enabling Sparse Optimizations: 
```
export retrain=0 # should be >0 options for prune_t (prune type) mag, l1_struc or random_unstru (default) will be set otherwise
export prune_t="random_unstru" 
export prune_amt=0.5 
```
### Enabling VGG loss 
```export model="ddnet" # choice ddnet, vgg16 (ddnet with vgg-16 based loss), vgg19 (ddnet with vgg-16 based loss)```

### Scaling 

Make the following changes in job_tinker.sh file 
```
export MASTER_PORT=<some unique value> 
```

And update below SLURM headers 
```
#SBATCH --ntasks-per-node P
#SBATCH --gpus-per-node G
#SBATCH --nodes N
```
P: number of parallel process 
G: Number of GPUs per node 
> G=P 

N: number of nodes
