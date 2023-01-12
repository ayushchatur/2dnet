#!/usr/bin/env python
# coding: utf-8

# # testing dali pipeline

# In[1]:


import types
import collections
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import torch
from dali_pipe import NumpyExternalSource, ExternalSourcePipeline

batch_size = 16
epochs = 3


# In[2]:


dir_hq = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ"
dir_lq = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ"


# In[3]:


from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

eii = NumpyExternalSource(dir_hq,dir_lq, batch_size, 0, 1)
pipe = ExternalSourcePipeline(batch_size=batch_size, num_threads=2, device_id = 0,
                              external_data = eii)
output_str = ['HQ','LQ', 'max', 'min']
pii = PyTorchIterator(pipe,output_str, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)

for e in range(epochs):
    for i, data in enumerate(pii):
#         HQ_img, LQ_img, maxs, mins = batch_samples['HQ'], batch_samples['LQ'], \
#                                                     batch_samples['max'], batch_samples['min']
        print("epoch: {}, iter {}, real batch size: {}".format(e, i, len(data[0])))
#         print('t d[0]' , type(data[0]))
        for batch_samples in data:
            HQ_img, LQ_img, maxs, mins = batch_samples['HQ'], batch_samples['LQ'], \
                                                    batch_samples['max'], batch_samples['min']
        
            print('HQ_img' , type(HQ_img), HQ_img.shape)
            print('LQ_img' , type(LQ_img), LQ_img.shape)
            print('maxs' , type(maxs), maxs.shape)
            
            
            print('batch_samples' , len(batch_samples))
#         print(batch_samples)
#         print('file name:' , data[0].shape)
    pii.reset()


# In[ ]:




