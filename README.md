# 2dnet
2dnet

### instructions on running
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
  
### instrcutions on profiling 
- source params.sh
- export enable_profile="true"
- sbatch job_tinker.sh

> Read carefully the comments in [job_tinker.sh](./job_tinker.sh)

Provide access to all users to Python file 
``` sh 
  chmod 766 sparse_ddnet.py
  chmod 766 trainers.py
 ```
