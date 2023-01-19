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

batch_size = 4
epochs = 3


# In[2]:


dir_hq = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ"
dir_lq = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ"



# In[ ]:




