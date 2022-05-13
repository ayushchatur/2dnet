# 2dnet
2dnet

### instructions on running
Make sure following are in the directory
- weights_50_1.pt
- Read Access to "/projects/synergy_lab/garvit217/enhancement_data/" 

the follow: 
- git checkout
- cd 2dnet
- sbatch job_tinker.sh
- output will be generated in slurm-<jobid>.out
### instrcutions on profiling 

Provide all access to all users to python file 
``` sh 
  chmod 766 sparse_ddnet.py
 ```
  
use the following command in job_tinker.sh file 
To profile the application we use DL prof which is available in the pytorch containers change the following [line](https://github.com/ayushchatur/2dnet/blob/7b9f3989dfa16faed1f7ef7f5f417abdee4866f9/sparse_ddnet.py#L737) in  job_tinker.sh file

``` sh 
singularity exec --nv --writable-tmpfs --bind=${TMPDIR},/cm/shared:/cm/shared,/projects:/projects $imagefile dlprof --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn" -f true --reports=summary,detail,iteration,kernel,tensor --delay 60 --duration 60 python sparse_ddnet.py -n 1 -g 1 --batch $batch_size --epochs $epochs --retrain $retrain

``` 
