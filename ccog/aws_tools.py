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
    
import fsspec
from osgeo import gdal
from rasterio._path import _parse_path, _vsi_path

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
        """
        uploads a part to a multipart upload

        store is an fsspec mapping with an mpu attribute added by create_mpu
        data is what to write to the part - note some size limitation from s3

        returns mpu part information
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
        