import os
import mmap
import io
import sys
import glob
import h5py as h5
import numpy as np
from os import path
from PIL import Image
import concurrent.futures as cf
import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

import cupy as cp

def read_correct_image(path):
    offset = 0
    ct_org = None
    with Image.open(path) as img:
        ct_org = np.float32(np.array(img))
        if 270 in img.tag.keys():
            for item in img.tag[270][0].split("\n"):
                if "c0=" in item:
                    loi = item.strip()
                    offset = re.findall(r"[-+]?\d*\.\d+|\d+", loi)
                    offset = (float(offset[1]))
    ct_org = ct_org + offset
    neg_val_index = ct_org < (-1024)
    ct_org[neg_val_index] = -1024
    return ct_org



class NumpyExternalSource(object):
    

    def __init__(self, hq_dir, lq_dir, batch_size, device_id, num_gpus):
#         self.images_dir = "../../data/images/"
        self.hq_dir = hq_dir + "/"
        self.lq_dir = hq_dir + "/"
        self.batch_size = batch_size
        self.img_list_all_l = os.listdir(self.lq_dir)
        self.img_list_all_h = os.listdir(self.hq_dir)
#         self.vgg_hq_img3 = os.listdir(self.data_root_h_vgg_3)
#         self.vgg_hq_img1 = os.listdir(self.data_root_h_vgg_1)

        self.img_list_all_l.sort()
        self.img_list_all_h.sort()
        
#         with open(self.images_dir + "file_list.txt", 'r') as f:
#             self.files = [line.rstrip() for line in f if line is not '']
        # whole data set size
        self.data_set_len = len(self.img_list_all_h) 
        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.img_list_h = self.img_list_all_h[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        self.img_list_l = self.img_list_all_l[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        
        self.n = len(self.img_list_l)
        self.rmin = 0
        self.rmax = 0

    def __iter__(self):
        self.i = 0
        shuffle(self.img_list_l)
        shuffle(self.img_list_h)
        return self

    def __next__(self):
#         rmin 
        hq_bl = []
        lq_bl = []
        max_bl = []
        min_bl = []
        vol_bl = []
#         max_i
#         min_i

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            inputs_np = None
            targets_np = None
            image_input = read_correct_image(self.lq_dir + self.img_list_l[self.i % self.n])
            image_target = read_correct_image(self.hq_dir + self.img_list_h[self.i % self.n])
            cmax1 = np.amax(image_target)
            cmin1 = np.amin(image_target)
            image_target = self.rmin + ((image_target - cmin1) / (cmax1 - cmin1) * (self.rmax - self.rmin))
            assert ((np.amin(image_target) >= 0) and (np.amax(image_target) <= 1))
            cmax2 = np.amax(image_input)
            cmin2 = np.amin(image_input)
            image_input = self.rmin + ((image_input - cmin2) / (cmax2 - cmin2) * (self.rmax - self.rmin))
            assert ((np.amin(image_input) >= 0) and (np.amax(image_input) <= 1))
            mins = ((cmin1 + cmin2) / 2)
            maxs = ((cmax1 + cmax2) / 2)
            image_target = image_target.reshape((1, 512, 512))
            image_input = image_input.reshape((1, 512, 512))
            inputs_np = image_input
            targets_np = image_target
    
            inputs = torch.from_numpy(inputs_np)
            targets = torch.from_numpy(targets_np)
#             print(targets)
    
#             inputs = inputs.type(torch.FloatTensor)
#             targets = targets.type(torch.FloatTensor)
            
#             jpeg_filename, label = self.files[self.i % self.n].split(' ')
            hq_bl.append(targets)  # we can use numpy
            lq_bl.append(inputs) # or PyTorch's native tensors
            vol_bl.append(str(self.img_list_l[self.i % self.n]))
            max_bl.append(maxs)
            min_bl.append(mins)
            print(vol_bl)
            
            self.i += 1
        return ( hq_bl,lq_bl,max_bl, min_bl)

    def __len__(self):
        return self.data_set_len

    next = __next__
    
class DaliLoaderCT(object):
    
    def get_pipeline(self, external_data):
        pipe = Pipeline(batch_size=self.batchsize, 
                            num_threads=self.num_threads, 
                            device_id=self.device.index,
                            seed = self.seed)
        with pipe:
#             with pipe:
            hq, lq, maxs , mins = fn.external_source(source=external_data, num_outputs=4)
#         images = fn.decoders.image(jpegs, device="mixed")
#         images = fn.resize(images, resize_x=240, resize_y=240)
#         output = fn.cast(images, dtype=types.UINT8)
            print('hq' , type(hq))
            hq = hq.gpu()
            lq = lq.gpu()
#         hq_ig = hq.to(device_id)
#         lq_ig = lq.to(device_id)
            pipe.set_outputs(hq,lq,maxs,mins)
    return pipe
            
    def __init__(self,hq_dir, lq_dir , batch_size, num_threads = 1, device = torch.device("cpu"), train = True):
        self.hq_dir = hq_dir
        self.lq_dir = lq_dir 
        self.num_threads = num_threads
        self.batchsize = batch_size
        self.device = device 
        self.pipeline = None
        self.iterator = None
        self.train = train
        self.init_file()
        self.pipeline = self.get_pipeline() 
        self.iterator = DALIGenericIterator([self.pipeline], ['data', 'label'], auto_reset = True,
                                            reader_name = "data",
                                            last_batch_policy = LastBatchPolicy.PARTIAL if self.train else LastBatchPolicy.DROP,
                                            prepare_first_batch = False)
        self.epoch_s = self.pipeline.epoch_size()
    @property
    def shapes(self):
        return self.data_shape
    
    def __iter__(self):
        for item in self.iterator:
            hq_img = item[0]['HQ']
            fname = item[0]['vol']
            yield hq_img, fname
        
        
    
class NumpyExternalSource(object):

    def init_files(self):
        # determine shard sizes etc:
        # shard the bulk first
        num_files = len(self.hq_list)
        num_files_per_shard = num_files // self.num_shards
        bulk_start = self.shard_id * num_files_per_shard
        bulk_end = bulk_start + num_files_per_shard
        
        # get the remainder now
        rem_start = self.num_shards * num_files_per_shard
        rem_end = num_files    

        # compute the chunked list
        self.hq_files_chunk = []
        self.lq_files_chunk = [] 
        for _ in range(self.oversampling_factor):
            
            # shuffle list
            perm = self.rng.permutation(range(num_files))
            hq_files = np.array(self.hq_list)[perm]
            lq_files = np.array(self.lq_list)[perm]

            # chunk: bulk
            hq_files_chunk = hq_files[bulk_start:bulk_end]
            lq_files_chunk = lq_files[bulk_start:bulk_end]

            # chunk: remainder
            hq_rem = hq_files[rem_start:rem_end]
            lq_rem = lq_files[rem_start:rem_end]
            if (self.shard_id < data_rem.shape[0]):
                np.append(hq_files_chunk, hq_rem[self.shard_id:self.shard_id+1], axis=0)
                np.append(lq_files_chunk, lq_rem[self.shard_id:self.shard_id+1], axis=0)

            self.hq_files_chunk.append(hq_files_chunk)
            self.lq_files_chunk.append(lq_files_chunk)

        return

            
    def start_prefetching(self):

        # we don't need to start prefetching again if it is already running
        # or if we do not cache data
        if self.prefetching_started or (not self.cache_data):
            return
        
        # flatten data lists: warning, unique sorts the data. We need to undo the sort
        # we need to make sure we return unique elements but maintain original ordering
        # data
        data_files = np.concatenate(self.data_files_chunks, axis = 0)
        data_files = data_files[np.sort(np.unique(data_files, return_index=True)[1])].tolist()
        # label
        label_files = np.concatenate(self.label_files_chunks, axis = 0)
        label_files = label_files[np.sort(np.unique(label_files, return_index=True)[1])].tolist()

        # get filesizes and fs blocksizes:
        # data
        fd = os.open(data_files[0], os.O_RDONLY)
        info = os.fstat(fd)
        self.data_filesize = info.st_size
        self.blocksize = info.st_blksize
        os.close(fd)
        # label
        fd = os.open(label_files[0], os.O_RDONLY)
        info = os.fstat(fd)
        self.label_filesize = info.st_size
        os.close(fd)
        
        # if zip does not work here, there is a bug
        for data_file, label_file in zip(data_files, label_files):
            # data and label are still unique, so just append to the queue
            self.prefetch_queue[data_file] = self.process_pool.submit(self._prefetch_sample, data_file, self.data_filesize, self.blocksize)
            self.prefetch_queue[label_file] = self.process_pool.submit(self._prefetch_sample, label_file, self.label_filesize, self.blocksize)

        # mark prefetching as started
        self.prefetching_started = True

        return


    def finalize_prefetching(self):
        if not self.prefetching_started or (not self.cache_data):
            return

        for task in cf.as_completed(self.prefetch_queue.values()):
            filename, data = task.result()
            self.file_cache[filename] = data
            
        return

    
    def _check_prefetching(self):
        # iterate over queue once:
        rmkeys = []
        for key in self.prefetch_queue.keys():
            task = self.prefetch_queue[key]
            if not task.done():
                continue
            
            filename, data = task.result()
            self.file_cache[filename] = data
            rmkeys.append(key)

        for key in rmkeys:
            self.prefetch_queue.pop(key)

        # check if queue is empty after this:
        if not self.prefetch_queue:
            self.cache_ready = True
        
        return
    
    
    def __init__(self, hq_list, lq_list, batch_size, last_batch_mode = "drop",
                 num_shards = 1, shard_id = 0, oversampling_factor = 1, shuffle = False,
                 cache_data = False, cache_device = "cpu", num_threads = 4, seed = 333):

        # important parameters
        self.hq_list = hq_list #list of file names only
        self.lq_list = lq_list
        self.batch_size = batch_size
        self.last_batch_mode = last_batch_mode

        # sharding info
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.oversampling_factor = oversampling_factor
        self.shuffle = shuffle
        self.cache_data = cache_data
        self.seed = seed
        self.rng = np.random.default_rng(seed = self.seed)

        # file cache relevant stuff
        self.prefetch_queue = {}
        self.file_cache = {}
        self.process_pool = cf.ThreadPoolExecutor(max_workers = num_threads)
        self.prefetching_started = False
        self.cache_ready = False
        self.use_direct_io = True

        # some file properties
        self.data_filesize = 0
        self.label_filesize = 0
        self.blocksize = 4096

        # running parameters
        self.chunk_idx = 0
        self.file_idx = 0

        # init file lists
        self.init_files()

        # some buffer for double buffering
        # determine shapes first, then preallocate
        hq_shape = self.hq_list[0]
        self.data_shape, self.data_dtype = hq_shape.shape, hq_shape.dtype
        # allocate buffers
        #self.data_batch = [ np.zeros(self.data_shape, dtype=self.data_dtype) for _ in range(self.batch_size) ]
        #self.label_batch = [ np.zeros(self.label_shape, dtype=self.label_dtype) for _ in range(self.batch_size) ] 

        
    def __iter__(self):
        self.chunk_idx = (self.chunk_idx + 1) % self.oversampling_factor
        self.file_idx = 0
        self.length = self.hq_files_chunks[self.chunk_idx].shape[0]

        # set new lists
        self.hq_files_current = self.hq_files_chunks[self.chunk_idx]
        self.lq_files_current = self.lq_files_chunks[self.chunk_idx]
        
        # shuffle chunk ONLY if prefetching is done
        if self.shuffle and self.cache_ready:
            perm = self.rng.permutation(range(self.length))
            self.hq_files_current = self.hq_files_current[perm]
            self.lq_files_current = self.lq_files_current[perm]

        if self.prefetching_started and not self.cache_ready:
            self._check_prefetching()
            
        self.get_sample_handle = self._get_sample_cache if self.cache_ready else self._get_sample_test
            
        return self

    
    def _get_sample_cache(self, data_filename, label_filename, batch_id=0):
        return self.file_cache[data_filename], self.file_cache[label_filename]


    def _load_sample(self, filename, filesize=0, blocksize=4096):
        
        torch.cuda.nvtx.range_push("NumpyExternalSource::_load_sample")
        
        if self.use_direct_io:
            ## open file
            #fd = os.open(filename, os.O_RDONLY | os.O_DIRECT)
            #
            ## check file size:
            #stat = os.fstat(fd)
            #readsize = stat.st_size
            #
            ## create mmap
            #mm = mmap.mmap(fd, readsize, offset=0, access=mmap.ACCESS_READ)
            #data = mm.read(readsize)
            #mm.close()
            
            ## close file
            #os.close(fd)
            data = ioh.load_file_direct(filename, blocksize=blocksize, filesize=filesize)

            # convert to numpy
            token = np.load(io.BytesIO(data))  
        else:
            token = np.load(filename)
            
        torch.cuda.nvtx.range_pop()
            
        return token
    

    def _get_sample_test(self, data_filename, label_filename, batch_id=0):

        torch.cuda.nvtx.range_push("NumpyExternalSource::_get_sample_test")
        
        #print(f"_get_sample_test for {data_filename}, {label_filename}")
        # fetch data
        if data_filename not in self.file_cache:
            #print(f"wait for {data_filename}")
            _, token = self.prefetch_queue[data_filename].result()
            self.file_cache[data_filename] = token
            self.prefetch_queue.pop(data_filename)
            
        data = self.file_cache[data_filename]

        # fetch label
        if label_filename not in self.file_cache:
            #print(f"wait for {label_filename}")
            _, token = self.prefetch_queue[label_filename].result()
            self.file_cache[label_filename] = token
            self.prefetch_queue.pop(label_filename)
                
        label = self.file_cache[label_filename]

        torch.cuda.nvtx.range_pop()
            
        return data, label
    
    
    def _prefetch_sample(self, filename, filesize, blocksize):
        data = self._load_sample(filename, filesize=filesize, blocksize=blocksize)
        return filename, data
    
    
    def __next__(self):
        torch.cuda.nvtx.range_push("NumpyExternalSource::__next__")
        # prepare empty batch
        fname = []
        hq = []
        lq = []
        max_s = []
        min_s = []
        
        # check if epoch ends here
        if self.file_idx >= self.length:
            raise StopIteration

        if ((self.file_idx + self.batch_size) >= self.length) and (self.last_batch_mode == "drop"):
            raise StopIteration
        elif (self.last_batch_mode == "partial"):
            batch_size_eff = min([self.length - self.file_idx, self.batch_size])
        else:
            batch_size_eff = self.batch_size

        #print("Get Batch")
            
        # fill batch
        for idb in range(batch_size_eff):

            hq_filename = self.hq_files_current[self.file_idx]
            lq_filename = self.lq_files_current[self.file_idx]
            hq_i, lq_i, maxs, mins = self.get_sample_handle(hq_filename,idb)
            fname.append(hq_filename)
            hq.append(hq_i)
            lq.append(lq_i)
            max_s.append(maxs)
            min_s.appen(mins)
            self.file_idx = self.file_idx + 1
            
        torch.cuda.nvtx.range_pop()

        #print("Done getting batch")
        
        return (fname, hq, lq, max_s,min_s)

class DDnetDaliLoader(object):
    
    def get_pipeline(self):
        self.data_size = 1152*768*16*4
        self.label_size = 1152*768*4
        
        pipeline = Pipeline(batch_size = self.batchsize, 
                            num_threads = self.num_threads, 
                            device_id = self.device.index,
                            seed = self.seed)
                                 
        with pipeline:
            # no_copy = True is only safe to use if data cache is enabled
            fname,hq_img,lq_img,mins,maxs = fn.external_source(source = self.extsource,
                                             num_outputs = 5,
                                             cycle = "raise",
                                             no_copy = self.cache_data,
                                             parallel = False)
            hq_img = hq_img.gpu()
            lq_img = lq_img.gpu()
                                       
            # normalize data
#             data = fn.normalize(data, 
#                                 device = "gpu",
#                                 mean = self.data_mean,
#                                 stddev = self.data_stddev,
#                                 scale = 1.,
#                                 bytes_per_sample_hint = self.data_size)
            
            # cast label to long
#             label = fn.cast(label,
#                             device = "gpu",
#                             dtype = types.DALIDataType.INT64,
#                             bytes_per_sample_hint = self.label_size)

#             if self.transpose:
#                 data = fn.transpose(data,
#                                     device = "gpu",
#                                     perm = [2, 0, 1],
#                                     bytes_per_sample_hint = self.data_size)

            pipeline.set_outputs(fname, hq_img, lq_img, mins, maxs)
            
        return pipeline
    
    def read_correct_image(path):
    offset = 0
    ct_org = None
    with Image.open(path) as img:
        ct_org = np.float32(np.array(img))
        if 270 in img.tag.keys():
            for item in img.tag[270][0].split("\n"):
                if "c0=" in item:
                    loi = item.strip()
                    offset = re.findall(r"[-+]?\d*\.\d+|\d+", loi)
                    offset = (float(offset[1]))
    ct_org = ct_org + offset
    neg_val_index = ct_org < (-1024)
    ct_org[neg_val_index] = -1024
    return ct_org

    def init_files(self, hq_dir, lq_dir, leng):
        self.leng = leng
        self.hq_dir = hq_dir
        self.lq_dir = lq_dir
        self.data_root_l = root_dir_l + "/"
        self.data_root_h = root_dir_h + "/"
        
        self.img_list_l = os.listdir(self.lq_dir)
        self.img_list_h = os.listdir(self.hq_dir)
        
        self.img_list_l.sort()
        self.img_list_h.sort()
#         self.vgg_hq_img3.sort()
#         self.vgg_hq_img1.sort()

        self.img_list_l = self.img_list_l[0:length]
        self.img_list_h = self.img_list_h[0:length]
        self.leng = len(self.img_list_l)


        # get shapes
        self.data_shape = read_correct_image(self.data_root_h  + self.img_list_h[0]).shape
#         self.label_shape = np.load(self.label_files[0]).shape

        # open statsfile
#         with h5.File(statsfile, "r") as f:
#             data_mean = f["climate"]["minval"][...]
#             data_stddev = (f["climate"]["maxval"][...] - data_mean)
            
        #reshape into broadcastable shape: channels first
#         self.data_mean = np.reshape( data_mean, (1, 1, data_mean.shape[0]) ).astype(np.float32)
#         self.data_stddev = np.reshape( data_stddev, (1, 1, data_stddev.shape[0]) ).astype(np.float32)

        # clean up old iterator
        if self.iterator is not None:
            del(self.iterator)
            self.iterator = None
        
        # clean up old pipeline
        if self.pipeline is not None:
            del(self.pipeline)
            self.pipeline = None

        # io devices
        self.io_device = "gpu" if self.read_gpu else "cpu"

        # create ES
        self.extsource = NumpyExternalSource(self.img_list_h , self.img_list_l , self.batchsize,
                                             last_batch_mode = "partial" if self.is_validation else "drop",
                                             num_shards = self.num_shards, shard_id = self.shard_id,
                                             oversampling_factor = self.oversampling_factor, shuffle = self.shuffle,
                                             cache_data = self.cache_data, num_threads = self.num_es_threads, seed = self.seed)
        
        # set up pipeline
        self.pipeline = self.get_pipeline()
       
        # build pipes
        self.global_size = len(self.data_files)
        self.pipeline.build()
    def start_prefetching(self):
        self.extsource.start_prefetching()

        
    def finalize_prefetching(self):
        self.extsource.finalize_prefetching()
    def __init__(self,hq_files, lq_files, leng , batch_size, device = torch.device("cpu"), is_validation = False, last_batch_mode = "drop",num_threads=1
                 num_shards = 1, shard_id = 0, oversampling_factor = 1, shuffle = False,lazy_init = False,
                 use_mmap = True, read_gpu = False, seed = 333):

        # important parameters
        self.hq_files = hq_files
        self.lq_files = lq_files
        self.batchsize = batchsize
        self.num_threads = num_threads
        self.num_es_threads = 2
        self.device = device
        self.io_device = "gpu" if read_gpu else "cpu"
        self.use_mmap = use_mmap
        self.shuffle_mode = shuffle_mode
        self.read_gpu = read_gpu
        self.pipeline = None
        self.iterator = None
        self.lazy_init = lazy_init
#         self.transpose = transpose
#         self.augmentations = augmentations
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.is_validation = is_validation
        self.seed = seed
        self.epoch_size = 0
        self.leng = leng

        if self.shuffle_mode is not None:
            self.shuffle = True
            self.stick_to_shard = False
        else:
            self.shuffle = False
            self.stick_to_shard = True
        self.cache_data = True
        self.oversampling_factor = oversampling_factor
        self.rng = np.random.default_rng(self.seed)

        # init file lists
        self.init_files(hq_files, lq_files, leng)
        
        self.iterator = DALIGenericIterator([self.pipeline], ['vol', 'HQ', 'LQ' , 'mins' , 'maxs'], auto_reset = True,
                                            size = -1,
                                            last_batch_policy = LastBatchPolicy.PARTIAL if self.is_validation else LastBatchPolicy.DROP,
                                            prepare_first_batch = False)
        self.epoch_size = self.pipeline.epoch_size()

        # some buffer for double buffering
#         # determine shapes first, then preallocate
#         data = np.load(self.data_files_chunks[0][0])
#         self.data_shape, self.data_dtype = data.shape, data.dtype
#         label = np.load(self.label_files_chunks[0][0])
#         self.label_shape, self.label_dtype = label.shape, label.dtype
#         # allocate buffers
        #self.data_batch = [ np.zeros(self.data_shape, dtype=self.data_dtype) for _ in range(self.batch_size) ]
        #self.label_batch = [ np.zeros(self.label_shape, dtype=self.label_dtype) for _ in range(self.batch_size) ] 
    @property
    def shapes(self):
        return self.data_shape
        
        

    def __iter__(self):
        #self.iterator.reset()
        for token in self.iterator:
            fname = token[0]['vol']
            hq_img = token[0]['HQ']
            lq_img = token[0]['LQ']
            mins = token[0]['mins']
            maxs = token[0]['maxs']
            
            yield fname,hq_img,lq_img,mins,maxs, ""
