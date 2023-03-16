#!/bin/bash 

#export image=nvcr.io/nvidia/pytorch:21.12-py3
#export image=nvcr.io/nvidia/pytorch:22.04-py3
export image=jupy_torch113_cu117
docker run  -p 8885:5555 -p 8888:8888  -w /code -v /run/user/1001/2dnet:/projects/synergy_lab/garvit217/enhancement_data/ -v ~/:/code $image jupyter-lab 
