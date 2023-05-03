# 2dnet
2dnet

### instructions on running
- Read Access to "/projects/synergy_lab/garvit217/enhancement_data/" 

the follow: 
- git checkout
- cd 2dnet
> - Provide executable to file in the folder before running 
``` sh 
  chmod -R 766 * 
 ```
- source params.sh
- update the number of nodes and gpus refer [this](https://docs.arc.vt.edu/resources.html) for cluster configuration.
- update the conda environment in the job_tinker.sh and job.infer.sh files. !! IMPORTANT !!
- sbatch job_tinker.sh for tinkercliffs cluster
- sbatch job_infer.sh for infer cluster
- output will be generated in slurm-xxx.out
### instrcutions on profiling 
> - note: profiling only works **without** containers so you have to update your job script and create a same conda environment. Profiling requires dlprof which can be installed using these [steps](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html)
- source params.sh
- export enable_profile="true"
- sbatch job_xx.sh

### To create module environment 
- first use an interact job 
- interact -p <partition> --gres=gpu:1 -A <account> --time 1:00:00
- module load EasyBuild/4.6.1
- module use $EASYBUILD_INSTALLPATH_MODULES
- module load Anaconda3/2022.05 
> - use only anaconda version as above; if not available use easy build to install it using below command 
> - eb -r Anaconda3/2022.05
- module load cuDNN/8.4.1.50-CUDA-11.7.0
- module load containers/singularity/3.8.5 
> - for tinkercliffs module load containers/singularity

The above loaded module can be saved and used inside job script (which has been done in job_tinker.sh) using the following command:
module save cu117 
module savelist (to verify)
then in the job file 
module restore cu117
module list (to verify module are loaded)


### containers

The code runs with containers: 
make sure you have read access to following containers
```
/projects/synergy_lab/ayush/containers/pytorch_22.04.sif
/projects/synergy_lab/ayush/containers/pytorch_2.0.sif # required for graph optimizations
```