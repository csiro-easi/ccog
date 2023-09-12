# Implementation notes:
# fsspec provids a nice neat way for working with s3.
# while using fsspec these funtions are really tied to using s3
# However with some effort some could be adapted to other file specs.
# eg multipart uploads
# could be adapted for other file systems that expose multipart upload capabilities (eg i think google cloud storage does)
# for other file systems could write multiple files and then concatenate them and then cleanup

# while fsspec is neat it uses async under the hood. s3fs and fsspec give nice little wrappers to make those async calls available in sync code.
# fs.call_s3 is nice to use as in handles errors and retries.


from itertools import zip_longest

import dask
import fsspec
import numpy as np
from more_itertools import collapse

# s3 part limit docs https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html
# actually 1 to 10,000 (inclusive)
s3_part_limit = 10000
# 5 MiB - There is no minimum size limit on the last part of your multipart upload.
s3_min_part_bytes = 5 * 1024 * 1024
# 5 GiB
s3_max_part_bytes = 5 * 1024 * 1024 * 1024
# 5 TiB
s3_max_total_bytes = 5 * 1024 * 1024 * 1024 * 1024


def _resolve_store(store, storage_options=None):
    """
    handle stores supplied as strings with storage options

    storage_options is ignored if store is an fsspec mapping
    """
    storage_options = {} if storage_options is None else storage_options
    if not isinstance(store, fsspec.mapping.FSMap):
        store = fsspec.get_mapper(store, **storage_options)
    assert (
        "s3" in store.fs.protocol
    ), "this tool only works with the s3 file system at this stage"
    return store


def presigned_url(store, expiration=8 * 60 * 60, storage_options=None):
    """
    generating presigned urls so files can be used elsewhere.

    store is an s3 file path or fsspec mapping
    expiration (int, optional) - The number of seconds that the url is valid for
    storage_options (dict, optional) – Any additional parameters for the storage backend
    """
    storage_options = {} if storage_options is None else storage_options
    store = _resolve_store(store, storage_options)
    return store.fs.url(store.root, expiration)


class Mpu:
    def __init__(self, store, storage_options=None, mpu_options=None):
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
            # making this whole class a no op - useful for testing
            self.finalised = True
        if self.finalised:
            return
        self.store = _resolve_store(store, storage_options)
        self.bucket, self.key, _ = self.store.fs.split_path(self.store.root)
        # start the mpy abd store the mpu details with the store
        self.mpu = self.store.fs.call_s3(
            "create_multipart_upload", Bucket=self.bucket, Key=self.key, **mpu_options
        )

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
        data = _flatten_bytes(data)

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
        return [
            self.upload_part_mpu(i, part_data)
            for i, part_data in write_parts
            if i is not None
        ]

    def complete_mpu(self, mpu_parts):
        """
        finishes a multipart upload

        store is an fsspec mapping with an mpu attribute added by create_mpu
        mpu_parts = is an iterable of part information as returned from upload_part_mpu

        return the result of the complete_multipart_upload call
        """
        if self.finalised:
            return
        # flatten and make sure parts are sorted and filter out any skipped parts
        parts = []
        for part in mpu_parts:
            if isinstance(part, list):
                parts.extend(part)
            elif part is not None:
                parts.append(part)
        parts = sorted(parts, key=lambda y: y["PartNumber"])
        # print (parts)
        try:
            result = self.store.fs.call_s3(
                "complete_multipart_upload",
                Bucket=self.mpu["Bucket"],
                Key=self.mpu["Key"],
                UploadId=self.mpu["UploadId"],
                MultipartUpload={"Parts": parts},
            )
        except:
            print("multipart upload failed")
            _ = self.store.fs.call_s3(
                "abort_multipart_upload",
                Bucket=self.mpu["Bucket"],
                Key=self.mpu["Key"],
                UploadId=self.mpu["UploadId"],
            )
        # block any ongoing changes to the MPU
        self.finalised = True
        return result

def _flatten_bytes(*data):
    """
    takes arbitarily nestes list/tubles of bytes/bytesarray and flattens to bytes
    """
    if data is None:
        return

    # being flexible to accept (nested) iterables of bytes)
    if not isinstance(data, (bytes, bytearray, str)):
        bytes_arr = bytearray()
        for p in collapse(data, base_type=bytearray):
            bytes_arr.extend(p)
        data = bytes_arr

    if isinstance(data, bytearray):
        # need to cast to bytes - unfortunatly doubles memory usage
        data = bytes(data)

    if len(data) == 0:
        return
    return data
    

def mpu_upload_dask_partitioned(ordered_parts, store, storage_options=None):
    """
    takes an iterable of bytes data and uses dask to write to s3 with a multipart upload
    
    """
    storage_options = {} if storage_options is None else storage_options
    mpu_store = dask.delayed(Mpu)(store, storage_options=storage_options)

    mpu_write_planner_func = dask.delayed(mpu_write_planner, nout=2)
    mpu_writer_func = dask.delayed(mpu_store.upload_parts_mpu)
    mpu_complete_func = dask.delayed(mpu_store.complete_mpu)

    mpu_parts = []

    parts = [
        [[id, list(x)], [None, []]]
        for id, x in enumerate(np.array_split(ordered_parts, 10000), 1)
        if len(x)
    ]

    while parts:
        if len(parts) == 1:
            # deal with the last part
            mpu_parts.append(mpu_writer_func(parts.pop()))
            break

        parts_temp = []
        # access parts in pairs
        for part1, part2 in zip_longest(parts[0::2], parts[1::2], fillvalue=None):
            if part2 is None:
                # not a pair - send it straight up to the next level
                parts_temp.append(part1)
            else:
                buffers, writes = mpu_write_planner_func(part1, part2)
                parts_temp.append(buffers)
                mpu_parts.append(mpu_writer_func(writes))

        parts = parts_temp

    delayed_graph = mpu_complete_func(mpu_parts)
    return delayed_graph


def mpu_write_planner(part1, part2):
    [id_A, parts_A], [id_B, parts_B] = part1
    [id_C, parts_C], [id_D, parts_D] = part2

    # if part 1 is shorter then the lower limit then that part had no writes after it in previous step
    if sum(len(x) for x in collapse(parts_A)) < s3_min_part_bytes:
        assert not parts_B, "mmm data where it shouldnt be"
        parts_A.extend(parts_C)
        return [[id_A, parts_A], [id_D, parts_D]], []
    else:
        parts_B.extend(parts_C)
        # if second part makes it to here but it is too small to write it becomes the third part.
        # otherwise it stays as the second part and gets written
        if sum(len(x) for x in collapse(parts_B)) < s3_min_part_bytes:
            if id_A + 1 == id_C:
                #there are no writes already between these so we can merge them and write most of the data
                data = _flatten_bytes(parts_A,parts_B)
                if len(data) < s3_min_part_bytes:
                    return [[id_A, [data]], [None, []]], []
                
                split_at = s3_min_part_bytes
                if len(data) >= (s3_max_part_bytes + s3_min_part_bytes):
                    split_at = len(data)-s3_max_part_bytes
                return [[id_A, [data[:split_at]]], [None, []]], [id_C, [data[split_at:]]]

                    
                    
            
            return [[id_A, parts_A], [id_B, parts_B]], []
        else:
            return [[id_A, parts_A], [id_D, parts_D]], [[id_B, parts_B]]
