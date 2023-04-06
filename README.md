# 2dnet
2dnet

### instructions on running
- Read Access to "/projects/synergy_lab/garvit217/enhancement_data/" 

the follow: 
- git checkout
- cd 2dnet
- source params.sh
- 
- sbatch job_tinker.sh
- output will be generated in slurm-<jobid>.out
### instrcutions on profiling 
- source params.sh
- export enable_profile="true"
- sbatch job_tinker.sh

Provide all access to all users to python file 
``` sh 
  chmod 766 sparse_ddnet.py
 ```
