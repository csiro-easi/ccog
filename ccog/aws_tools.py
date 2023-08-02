# Implementation notes:
# fsspec provids a nice neat way for working with s3.
# while using fsspec these funtions are really tied to using s3
# However with some effort some could be adapted to other file specs.
# eg multipart uploads 
# could be adapted for other file systems that expose multipart upload capabilities (eg i think google cloud storage does)
# for other file systems could write multiple files and then concatenate them and then cleanup

# while fsspec is neat it uses async under the hood. s3fs and fsspec give nice little wrappers to make those async calls available in sync code.
# fs.call_s3 is nice to use as in handles errors and retries.
# fs.call_s3 doesnt seem to know about everything (eg generate_presigned_url).
# Im not sure why. There is obviously stuff going on that i dont understand.
# fsspec.asyn.sync works fine for generate_presigned_url and may be a path to not tiing this implementation to s3 so tightly.
#from fsspec.asyn import sync
    #return sync(
    #    fs.loop,
    #    fs.s3.generate_presigned_url,
    #    ClientMethod="get_object",
    #    Params={"Bucket": bucket, "Key": key},
    #    ExpiresIn=expiration,
    #)

from collections import deque
import fsspec
from osgeo import gdal
from rasterio._path import _parse_path, _vsi_path
import dask

def _resolve_store(store,storage_options=None):
    '''
    handle stores supplied as strings with storage options
    
    storage_options is ignored if store is an fsspec mapping
    '''
    storage_options = {} if storage_options is None else storage_options
    if not isinstance(store,fsspec.mapping.FSMap):
        store = fsspec.get_mapper(store, **storage_options)
    assert 's3' in store.fs.protocol, 'this tool only works with the s3 file system at this stage'
    return store

def s3_to_vrt(store,vrtpath= 's3.vrt', expiration=8 * 60 * 60,storage_options = None):
    '''makes a vrt file that points to a presigned url on s3
    
    useful for legacy GIS (arcmap) that cant open data from a url
    
    store is an s3 file path or fsspec mapping
    vrtpath - output filename for a .vrt file
    expiration (int, optional) - The number of seconds that the url is valid for
    storage_options (dict, optional) – Any additional parameters for the storage backend 
    '''
    storage_options = {} if storage_options is None else storage_options
    store = _resolve_store(store,storage_options)
    url = store.fs.url(store.root,expiration)
    #todo: rasterio is currently added vrt building functionality - switch to use that when its ready.
    ds = gdal.BuildVRT(vrtpath,_vsi_path(_parse_path(url)))
    ds.FlushCache()
    ds = None

    
def presigned_url(store, expiration=8 * 60 * 60,storage_options = None):
    """
    generating presigned urls so files can be used elsewhere.
    
    store is an s3 file path or fsspec mapping
    expiration (int, optional) - The number of seconds that the url is valid for
    storage_options (dict, optional) – Any additional parameters for the storage backend 
    """
    storage_options = {} if storage_options is None else storage_options
    store = _resolve_store(store,storage_options)
    return store.fs.url(store.root)

class Mpu:
    def __init__(self,store,storage_options =None, mpu_options=None):
        """
        starts a multipart upload
        Note: Dont use this normally as s3fs automatically does multipart uploads when needed
        This is for use if you want more control over the upload process such as uploading the parts from distributed workers

        store is an s3 file path or fsspec mapping
        storage_options (dict, optional) – Any additional parameters for the storage backend 
        mpu_options (dict, optional) – Any additional parameters for the s3 mpu call

        returns an fsspec mapping with an mpu attribute added

        """
        storage_options = {} if storage_options is None else storage_options
        mpu_options = {} if mpu_options is None else mpu_options
        self.finalised = False
        if store is None:
            #making this whole class a no op - useful for testing
            self.finalised = True
        if self.finalised:
            return
        self.store = _resolve_store(store,storage_options)
        self.bucket, self.key, _ = self.store.fs.split_path(self.store.root)
        #start the mpy abd store the mpu details with the store
        self.mpu = self.store.fs.call_s3("create_multipart_upload", Bucket=self.bucket, Key=self.key, **mpu_options)

    
    def upload_part_mpu(self, PartNumber, data):
        """Uploads a part to a multipart upload.

        Args:
            part_number: The number of the part to upload.
            data: The data to upload.

        Returns:
            A dictionary containing the part information.
        """
        if self.finalised:
            return
        part_mpu = self.store.fs.call_s3(
            "upload_part",
            Bucket=self.mpu["Bucket"],
            PartNumber=PartNumber,
            UploadId=self.mpu["UploadId"],
            Body=data,
            Key=self.mpu["Key"],
        )
        return {"PartNumber": PartNumber, "ETag": part_mpu["ETag"]}

    def upload_parts_mpu(self, write_parts):
        """Uploads multiple part to a multipart upload.

        Args:
            an iterable of tuples where the first value is the
            part_number: The number of the part to upload.
            and the second value is the
            data: The data to upload.

        Returns:
            A list of dictionaries containing the part information.
        """
        return [self.upload_part_mpu(i, part_data) for i,part_data in write_parts]

    def complete_mpu(self, mpu_parts):
        """
        finishes a multipart upload

        store is an fsspec mapping with an mpu attribute added by create_mpu
        mpu_parts = is an iterable of part information as returned from upload_part_mpu
        """
        if self.finalised:
            return
        #make sure parts are sorted and filter out any skipped parts
        mpu_parts = [part for part in mpu_parts if part is not None]
        mpu_parts = sorted(mpu_parts, key= lambda y: y["PartNumber"]) 
        try:
            _ = self.store.fs.call_s3(
                "complete_multipart_upload",
                Bucket=self.mpu["Bucket"],
                Key=self.mpu["Key"],
                UploadId=self.mpu["UploadId"],
                MultipartUpload={"Parts": mpu_parts},
            )
        except:
            print("multipart upload failed")
            _ = self.store.fs.call_s3(
                "abort_multipart_upload",
                Bucket=self.mpu["Bucket"],
                Key=self.mpu["Key"],
                UploadId=self.mpu["UploadId"],
            )
        #block any ongoing changes to the MPU
        self.finalised = True

        
def mpu_write_planner(buffer_parts,part_bytes,end_partition=False,merge_partition_buffer=None,end_final_partition = False,user_limit_bytes = 2 * 1024 * 1024 * 1024):
    '''
        the main place for the logic for handling partitioned writes to ranges of a mpu
        
        Designed to quickly group data into the parts for uploading.
        The uploading in handled in another function so that this one is fast to run and
        the uploading can happen in parallel.
        
    '''
    #partitions hold the first write back so enable the previous partiation to make a write if it doesnt get over the 5MiB limit.
    #this is unavoidable unless its ok to add padding into the file at the end of a partition.
    #this is also easier then keeping a 5MiB buffer active while writing a partitation (and such a buffer doesnt work if there is 
    #a partition with less then 5MiB of data to write.)
    
    #TODO: revisit being carefulof about memory usage as the data buffer might be many GB and avoiding duplicating this much data is worthwhile
    #TODO: the below logic works but its asthetically clunky. revisit with fresh eyes
    #TODO: ensure user limit bytes is in the valid s3 range

    
    #trigger the first write to have an upper limit at smaller size to avoid having to keep a first data part that is say 4GiB in memory when 5MiB will do
    #shouldnt usually hit this limit unless there is a large byte string near the start
    #100MiB - max it could be set at is (s3_max_part_bytes - s3_min_part_bytes)
    first_buffer_max = 100*1024*1024 

    partition_start_id,partition_end_id,next_id,queue = buffer_parts
    #add in the new list of bytes
    #this list of bytes may be the 'firsts' from the previous partition so filter out the Nones
    part_bytes = [item for item in part_bytes if item is not None]
    queue.extend(part_bytes)
    
    buffer_merged = False
    if merge_partition_buffer is not None and len(part_bytes) == 0:
        buffer_merged = True
        #in this case the call is to merge a partition
        #and the second partition didnt have enough bytes to produce a 'first write'
        # can also check merge_partition_buffer[0] == merge_partition_buffer[2] 
        #if this is the case need to merge the 2 buffers
        partition_end_id = merge_partition_buffer[1]
        queue.extend(merge_partition_buffer[3])
        end_partition = True
    
    
    current_length = 0
    current_queue = deque()
    
    first_write = None
    writes = []
    
    while queue:
        b = queue.popleft()
        current_length += len(b)
        current_queue.append(b)
        if current_length >= s3_min_part_bytes:
            if next_id == partition_start_id and partition_start_id != 1 and not end_partition and not end_final_partition:
                #first lot of data in the partition (but not the first partition) need to keep a small amount for joining partitions
                joined_buffer = b''.join(current_queue)
                #deal with if the data is too long (happens if a very large object is in the buffer)
                #first write needs to have space for a small write from the previous partition if needed
                if current_length > first_buffer_max:
                    #do something to split up the data
                    #generally want to avoid spliting the chunks of data that come in but unavoidable due to large part and s3_max_part_bytes
                    queue.appendleft(joined_buffer[s3_min_part_bytes:])
                    joined_buffer = joined_buffer[:s3_min_part_bytes]
                #store the first write
                first_write = [joined_buffer]
                current_queue.clear()
                current_length = 0
                next_id += 1

            elif current_length > user_limit_bytes:
                joined_buffer = b''.join(current_queue)
                if current_length > s3_max_part_bytes:
                    if (current_length - len(b)) >= s3_min_part_bytes:
                        #put the last one back
                        queue.appendleft(current_queue.pop())
                        joined_buffer = b''.join(current_queue)
                    else:
                        #cant put the last one back
                        #generally want to avoid spliting the chunks of data that come in but unavoidable due to large part and s3_max_part_bytes
                        joined_buffer = b''.join(current_queue)
                        queue.appendleft(joined_buffer[user_limit_bytes:])
                        joined_buffer = joined_buffer[:user_limit_bytes]
                        
                        
                writes.append ([next_id,joined_buffer])
                assert next_id <= partition_end_id, f'too many parts written in partition {partition_start_id} to {partition_end_id}' #todo: check this logic is correct
                current_queue.clear()
                current_length = 0
                next_id += 1
    
    #all the queue is now in current_queue now and should be shorter then user_limit_bytes
    if end_final_partition or ( end_partition and current_length >= s3_min_part_bytes):
        joined_buffer = b''.join(current_queue)
        writes.append ([next_id,joined_buffer])
        current_queue.clear()
        current_length = 0
        next_id += 1
    
    if not buffer_merged and merge_partition_buffer is not None:
        buffer_parts = merge_partition_buffer
        assert len(current_queue) == 0, 'buffer part management issue'
    else:
        buffer_parts = [partition_start_id,partition_end_id,next_id,current_queue]
    return first_write,writes,buffer_parts

def mpu_upload_dask_partitioned(partitions,store,storage_options=None):
    '''
    writes delayed lists of bytes to s3 with a mpu using partitions to allow writes out of order
    
    The aim is to move most of the data to S3 as soon as possible to avoid large caches
    Small caches are used at the boundaries of partitions to avoid issues due to the s3 mpu minimum part size.
    
    simplified example 
    partition_specs =  {'header':{'start':1,'end':2,'data':[dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),]},
                        'last_overviews':{'start':2,'end':3,'data':[dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),]},
                        6:{'start':3,'end':5,'data':[dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),]},
                        5:{'start':5,'end':12,'data':[dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),]},
                        4:{'start':12,'end':42,'data':[dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),]},
                        3:{'start':42,'end':159,'data':[dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),]},
                        2:{'start':159,'end':627,'data':[dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),]},
                        1:{'start':627,'end':2502,'data':[dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),]},
                        0:{'start':2502,'end':10000,'data':[dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),dask.delayed([b'',b'',b'']),]},
                       }
    partitions =partition_specs.values()
    dd = ccog.aws_tools.mpu_upload_dask_partitioned(partitions,'s3:\test\test.tif')
    #%lprun -f ccog.aws_tools.mpu_upload_dask_partitioned ccog.aws_tools.mpu_upload_dask_partitioned(partitions,'test')
    dd.visualize(optimize_graph=True)
    
    '''
    #TODO: ensure partitions are provided sorted first to last ( or sort them)
    
    storage_options = {} if storage_options is None else storage_options
    mpu_store = dask.delayed(Mpu)(store,storage_options=storage_options)
    
    mpu_write_planner_func = dask.delayed(mpu_write_planner, nout=3)
    mpu_writer_func = dask.delayed(mpu_store.upload_parts_mpu, nout=1)
    mpu_complete_func = dask.delayed(mpu_store.complete_mpu)
    
    #now deal with the writing to mpu
    mpu_parts = []
    previous_buffer_parts = None
    for partition in partitions:
        first_parts = []
        buffer_parts = [partition['start'],partition['end'],partition['start'],deque(),]
        last_flag = (len(partition['data'])-1) * [False] + [True]
        for part_bytes,last in zip(partition['data'],last_flag):
            first_part,write_parts,buffer_parts = mpu_write_planner_func(buffer_parts,part_bytes,end_partition = last)
            first_parts.append(first_part)
            mpu_parts.append(mpu_writer_func(write_parts))
        
        #Resolve partition boundaries
        #Note these boundaries are resolved first to last. Other ordering is possible and may reduce caching load
        #however as long as the amount of data cached to deal with the boundares is small its unlikely to be problematic
        if previous_buffer_parts is not None:
            first_part,write_parts,buffer_parts = mpu_write_planner_func(previous_buffer_parts,first_parts,merge_partition_buffer = buffer_parts)
            first_parts.append(first_part) #todo: shouldnt be any first parts here - throw error if there is
            mpu_parts.append(mpu_writer_func(write_parts))
        previous_buffer_parts = buffer_parts
        
    #write the final mpu
    first_part,write_parts,buffer_parts = mpu_write_planner_func(buffer_parts,end_final_partition = True)
    #todo: shouldnt be any first parts or buffer_parts here - throw error if there is
    mpu_parts.append(mpu_writer_func(write_parts))
    
    delayed_graph = mpu_complete_func(mpu_parts)
    return delayed_graph