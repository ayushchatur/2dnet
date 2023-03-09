#!/bin/bash

export pytor="ver1"
#to enable graph chane pytorch version above

export graph_mode="reduce-overhead"
export gr_back="incubator"

export enable_profile="true"
#graph_mode: 1-> 2-> 3-> max-autotune



export batch_size=32
export epochs=30
export retrain=5
export num_data_w=1
export wandb="true"
# options mag/l1_struc or random_unstru will be set otherwise
export prune_t="mag"
export prune_amt="0.5"
#options enable/disable
export mp="disable"
#pytorch version 1 or version 2(with new graph support)


echo "batch : $batch_size"
echo "retrain : $retrain"
echo "epochs : $epochs"