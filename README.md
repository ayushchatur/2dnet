# 2dnet
2dnet

### instructions on running
Make sure following are in the directory
- weights_50_1.pt
- Read Access to "/projects/synergy_lab/garvit217/enhancement_data/" 

### instrcutions on profiling 


use the following command in job_tinker.sh file 
To profile the application we use DL prof which is available in the container files

``` sh 
export batch_size=1
export epochs=50
export retrain=1
echo "batch : $batch_size"

echo "retrain : $retrain"

echo "epochs : $epochs"
imagefile=/projects/arcsingularity/AMD/ood-jupyter-pytorch_21.12.sif
module load containers/singularity

singularity exec --nv --writable-tmpfs --bind=${TMPDIR},/cm/shared:/cm/shared,/projects:/projects $imagefile dlprof --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn" -f true --reports=summary,detail,iteration,kernel,tensor --delay 60 --duration 60 python sparse_ddnet.py -n 1 -g 1 --batch $batch_size --epochs $epochs --retrain $retrain

``` 
To enable baseline change the following [line](https://github.com/ayushchatur/2dnet/blob/7b9f3989dfa16faed1f7ef7f5f417abdee4866f9/sparse_ddnet.py#L737) 
parameter from O0 to O2 
- O0 => fp32 trainig
- O2 => mixed precision training
