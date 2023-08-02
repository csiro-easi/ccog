__all__ = ['write_ccog']

from contextlib import suppress
from itertools import count
from collections import deque
import math
import io
import struct
import xml.etree.ElementTree as ET
import numpy as np
import xarray
import dask
from distributed import get_client
from dask.base import tokenize
import rasterio
from rasterio import shutil #cant see it ifs not imported apparently
#from rasterio.rio.overview import get_maximum_overview_level
from affine import Affine

import tifffile

from . import aws_tools

# s3 part limit docs https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html
# actually 1 to 10,000 (inclusive) 
s3_part_limit = 10000  
# leaving the first part for the headers
s3_start_part = 2  
# 5 MiB - There is no minimum size limit on the last part of your multipart upload.
s3_min_part_bytes = 5 * 1024 * 1024  
# 5 GiB
s3_max_part_bytes = 5 * 1024 * 1024 * 1024  
# 5 TiB
s3_max_total_bytes = 5 * 1024 * 1024 * 1024 * 1024 

required_creation_options = dict(
    sparse_ok=True,
    driver="COG",
    bigtiff="YES",
    num_threads="all_cpus", #TODO: test what difference this makes if any on a dask cluster
    overviews="AUTO",
    jpegtablesmode = 0, #not tested but will be needed when I get to multiband compressed data- does it work with the GDAL COG driver? 
)

default_creation_options = dict(
    geotiff_version= 1.1, #why not use the latest by default
    blocksize = 512,
    overview_resampling = rasterio.enums.Resampling.nearest,

)

def get_maximum_overview_level(width, height, minsize=256,overview_count=None):
    """
    Calculate the maximum overview level of a dataset at which
    the smallest overview is smaller than `minsize`.
    
    Based on rasterio.rio.overview.get_maximum_overview_level
    modified to match the behaviour of the gdal COG driver
    so that the smallest overview is smaller than `minsize` in BOTH width and height.
    
    modified to have different behaviour to rasterio and GDAL
    wont produce more overview levels once one dimension has reduced to 1 pixel.
    Attributes
    ----------
    width : int
        Width of the dataset.
    height : int
        Height of the dataset.
    minsize : int (default: 256)
        Minimum overview size.
    Returns
    -------
    overview_level: int
        overview level.
    """
    #note GDAL seems to have a limit of an overview level of 30
    
    #GDAL will keep producing overviews for single pixel strands of data longer then blocksize.
    #this is a bit odd because the resampling doesnt make sense
    #here limit overviews when min dimension is 1.

    overview_level = 0
    overview_factor = 1
    if overview_count is not None:
        while overview_count > overview_level and min(width // overview_factor, height // overview_factor) > 1:
            overview_factor *= 2
            overview_level += 1 
            print (width // overview_factor, height // overview_factor)
    else:
        while max(width // overview_factor, height // overview_factor) > minsize and min(width // overview_factor, height // overview_factor) > 1:
            overview_factor *= 2
            overview_level += 1

    return overview_level


def xarray_to_profile(x_arr):
    profile = dict(        
        dtype = x_arr.dtype.name,
        width = x_arr.rio.width,
        height = x_arr.rio.height,
        transform = x_arr.rio.transform(),
        crs = x_arr.rio.crs,
        nodata = x_arr.rio.nodata,
        count = x_arr.rio.count,
    )
    return profile

def empty_single_band_vrt(**profile):
    """Make a VRT XML document.

    This is a very simple vrt with no datasets.

    Its used as a starting point to create an empty COG file

    Parameters
    ----------
    Returns
    -------
    str
        An XML text string.
    """
    #         Implementation notes:
    #     A COG can only be created with the gdal COG driver by copying an existing dataset
    #     Note that rasterio seems to hide this in the background but also seems to 
    #     break the handling of nodata in sparse COGs if you never write to them.
    #     The method used here is also faster than getting rasterio to produce an empty COG from scratch.

    #     The existing dataset doesnt need to be a VRT but experimentation has shown that this VRT
    #     is faster then the other empty datasets Ive tried as starting points.
    #     Of particular benefit is the inclusion of an OverviewList
    #     It seems that building overviews, even if empty and sparse is the slow part of making a COG.

    #     An alternative approach to produce the vrt xml written in this code
    #     is to use a dummy dataset and BuildVRT and then
    #     build_overviews with VRT_VIRTUAL_OVERVIEWS = 'YES'and then editing the VRT to remove the dummy dataset
    #     The code below produces an output identical to the above steps but this implementation 
    #     is prefered as it is more direct and hopefully clearer
    
    overviews = " ".join([str(2**j) for j in range(1, profile['overview_count'] + 1)])
    
    # based on code in rasterio https://github.com/rasterio/rasterio/blob/main/rasterio/vrt.py
    vrtdataset = ET.Element("VRTDataset")
    vrtdataset.attrib["rasterXSize"] = str(profile['width'])
    vrtdataset.attrib["rasterYSize"] = str(profile['height'])
    srs = ET.SubElement(vrtdataset, "SRS")
    srs.text = profile['crs'].wkt if 'crs' in profile else ""
    geotransform = ET.SubElement(vrtdataset, "GeoTransform")
    geotransform.text = ",".join([str(v) for v in profile['transform'].to_gdal()])
    
    vrtrasterband = ET.SubElement(vrtdataset, "VRTRasterBand")
    vrtrasterband.attrib["dataType"] = rasterio.dtypes._gdal_typename(profile['dtype'])
    vrtrasterband.attrib["band"] = str(1)
    vrtrasterband.attrib["blockXSize"] = str(profile['blocksize'])
    vrtrasterband.attrib["blockYSize"] = str(profile['blocksize'])
    if 'nodata' in profile:
        nodatavalue = ET.SubElement(vrtrasterband, "NoDataValue")
        nodatavalue.text = str(profile['nodata'])
    OverviewList = ET.SubElement(vrtdataset, "OverviewList")
    OverviewList.attrib["resampling"] = profile['overview_resampling'].name.upper()
    OverviewList.text = overviews
    return ET.tostring(vrtdataset).decode("ascii")

def empty_single_band_COG(profile,rasterio_env_options=None):
    """
    makes an empty sparse COG in memory

    used as a reference to look at the structure of a COG file
    and as the starting point for some file format fiddling.

    returns bytes
    """
    rasterio_env_options = {} if rasterio_env_options is None else rasterio_env_options
    #this is applied elsewhere but apply here for when testing
    profile.update(required_creation_options)
       
    vrt = empty_single_band_vrt(**profile)

    with rasterio.Env(**rasterio_env_options):
        with rasterio.io.MemoryFile() as memfile:
            memfile = rasterio.io.MemoryFile()
            rasterio.shutil.copy(
                vrt,
                memfile.name,
                **profile
            )
            memfile.seek(0)
            data = memfile.read()
    return data

def delete_COG_ghost_header(memfile):
    '''If messing with the gdal COG block order optimisations its likely a good idea to let GDAL know about it in the ghost header
    its a little annoying to have to fully switch off the optimisations as its still 'mostly' got all the optimisations in place
    Ive not tested what GDAL does if if I dont switch off this optimisation. 
    GDAL complains if I only modify the header so im removing it fully. Leaving the space as its only a few hundred bytes.
    '''
    assert not memfile.closed, "memory file was closed before writing"
    memfile.seek(16+30)
    GDAL_STRUCTURAL_METADATA_SIZE = int(memfile.read(6))
    memfile.seek(16)
    memfile.write((43+GDAL_STRUCTURAL_METADATA_SIZE)*b"\0")
    memfile.seek(0)
    return

def partial_COG_maker(
    arr,
    profile,
    rasterio_env_options,
):
    """
    writes arr to an in memory COG file and the pulls the COG file apart and returns it in parts that are useful for
    reconstituting it differently.
    Gets all the advantages of rasterio/GDALs work writing the file so dont need to do that work from scratch.
    """
    profile = profile.copy() #careful not to update in place
    profile['height']= arr.shape[0]
    profile['width']= arr.shape[1]
    #not using the transform in profile as its going to be wrong - make it obviously wrong avoids rasterio warnings
    profile['transform']= Affine.scale(2)
    #the overview is only used for resampling its not stored
    profile["overview_count"] = 1
    profile["overview_compress"] = 'NONE'
    
    print (profile)
    shp = arr.shape
    with rasterio.Env(**rasterio_env_options):
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as src:
                src.write(arr, 1)
                del arr  # reducing mem footprint
            
            #need to stop gdal producing overviews that should shrink to nothing.
            #however note that GDAL wont do this for a whole image so there is an edge case inconsistency here
            overview_arr = None
            if min(shp)>1:
                with memfile.open(overview_level=0) as src:
                    # get the required overview back as an array
                    overview_arr = src.read(1)

            #removing the gdal ghost leaders
            part_bytes = []
            memfile.seek(0)
            #return (memfile.read()) #used for testing
            with tifffile.TiffFile(memfile) as tif:
                page = tif.pages[0]
            for offset, bytecount in zip(page.dataoffsets, page.databytecounts):
                _ = memfile.seek(offset)
                part_bytes.append(memfile.read(bytecount))
    
    # data will be more flexible later as 2d numpy arrays
    tile_col_count = math.ceil(page.imagewidth/page.tilewidth)
    #part_bytes = np.array(part_bytes,dtype=object).reshape(tile_col_count,-1) #not currently needed - leave as may be useful if rearranging tiles in a later step in the future

    #not convinced this is the correct place to produce this data - could easily be made from the part_bytes in a later step
    #eg use np.vectorize(len)(part_bytes) to generate databytecounts
    #this would give flexability for rearranging the data
    databytecounts = np.array(page.databytecounts,dtype=np.int64).reshape(tile_col_count,-1)
    #at some later stage sparse tiles (bytecount=0) need to have their offset set to zero before writing.
    databyteoffsets = np.cumsum([0,*page.databytecounts[0:-1]],dtype=np.int64).reshape(tile_col_count,-1)
    part_info = (databyteoffsets, databytecounts)
    return overview_arr,part_bytes,part_info

def adjust_compression(profile):
    '''
    When writing blocks that are actually all overviews gdal will still think the first level is not overviews
    so adjust the profile to account for this
    '''
    if 'overview_compress' in profile:
        if profile.get('overview_compress', None) != profile.get('compress', None):
            for k in ['quality','predictor']:
                if k in profile:
                    del profile[k] 
    for k1,k2 in [('overview_compress','compress'),('overview_quality','quality'),('overview_predictor','predictor'),]:
        if k1 in profile:
            profile[k2] = profile[k1]

def COG_graph_builder(da,store,profile,rasterio_env_options,storage_options=None):
    '''
    makes a dask delayed graph that when computed writes a COG file to S3
    '''
    tif_part_maker_func = dask.delayed(partial_COG_maker, nout=3)

    #ensure the overview count is set
    profile['overview_count'] = get_maximum_overview_level(profile['width'], profile['height'], minsize=profile['blocksize'],overview_count=profile.get('overview_count',None))
    profile = profile.copy()
    overview_profile = profile.copy()
    adjust_compression(overview_profile)

    del_profile = dask.delayed(profile,traverse=False)
    del_overview_profile = dask.delayed(overview_profile,traverse=False)
    del_rasterio_env_options = dask.delayed(rasterio_env_options,traverse=False)
    current_level_profile = del_profile
    parts_info = {}
    parts_bytes = {}

    for level in range(profile['overview_count']+1):
        parts_info[level] = []
        parts_bytes[level] = []
        chunk_heights, chunk_widths = da.chunks
        da_del = da.to_delayed(optimize_graph=True)
        res_arr = np.ndarray(da_del.shape, dtype=object)

        for blk in da_del.ravel():
            ind = blk.key[1:]
            overview_arr,part_bytes,part_info = tif_part_maker_func(blk,current_level_profile,del_rasterio_env_options,)
            parts_info[level].append((ind,part_info))
            parts_bytes[level].append(part_bytes) #((ind,part_bytes))
            blk_final_shape = (chunk_heights[ind[0]] // 2,chunk_widths[ind[1]] // 2)
            res_arr[ind] = dask.array.from_delayed(overview_arr, shape=blk_final_shape, dtype=da.dtype)

        #if level == profile['overview_count']: #last
        #    break
        da = dask.array.block(res_arr.tolist())
        #copy the chunking from the initial data - assuming the first chunk is indicative of appropriate chunking to use for overviews
        da = da.rechunk(chunks= (chunk_heights[0], chunk_widths[0]))
        current_level_profile = del_overview_profile

    #empty_single_band_COG is slow for large COGs, its seperate here so dask can run it early
    header_bytes = dask.delayed(empty_single_band_COG)(del_profile, rasterio_env_options= del_rasterio_env_options)
    header_bytes_final = dask.delayed(prep_tiff_header)(header_bytes, parts_info)
           
    #set how many mpu parts are used in each partition. 
    #these numbers have been determined from an analysis of the relative size of each overview 
    #every overview after 6 goes in 1 partition
    partition_specs =  {'header':{'start':1,'end':2,'data':[]},
                        'last_overviews':{'start':2,'end':3,'data':[]},
                        6:{'start':3,'end':5,'data':[]},
                        5:{'start':5,'end':12,'data':[]},
                        4:{'start':12,'end':42,'data':[]},
                        3:{'start':42,'end':159,'data':[]},
                        2:{'start':159,'end':627,'data':[]},
                        1:{'start':627,'end':2502,'data':[]},
                        0:{'start':2502,'end':10000,'data':[]},
                       }
                        
    #rearrange the delayed data into the write partitions
    partition_specs['header']['data']=[header_bytes_final]
    for level in sorted(parts_bytes, reverse=True):#revese so that when levels share a partition they need to be in this order
        partition_specs.get('level',partition_specs['last_overviews'])['data'].extend(parts_bytes[level])
    
    delayed_graph = mpu_upload_dask_partitioned(partition_specs.values(),store,storage_options=storage_options)
    return delayed_graph

def ifd_updater(memfile,block_data):
    #memfile is a bytesIO object
    assert not memfile.closed, "memory file was closed before writing"
    block_offsets,block_counts = block_data
    with tifffile.TiffFile(memfile) as tif:
        for page,offs,cnts in zip(tif.pages,block_offsets,block_counts):
            assert len(page.tags['TileOffsets'].value) == len(offs) == len(cnts), 'oh no data is muddled'
            page.tags['TileOffsets'].overwrite(offs)
            page.tags['TileByteCounts'].overwrite(cnts)
        
def ifd_offset_adjustments(header_length,parts_info):
    '''
    merging of ifd data into the right order needed for the concatenated COG tif header
    also applies part offsets to the image data offsets
    '''
    block_offsets = []
    block_counts = []
    #apply cumulative offsets to tile offset data
    offset = header_length
    #the order produced is right except the levels need reversing - should match the order data is written to file
    for level in sorted(parts_info, reverse=True):
        shp = [xy+1 for xy in parts_info[level][-1][0]]
        TileOffsets_arr = np.ndarray(shp, dtype=object)
        TileByteCounts_arr = np.ndarray(shp, dtype=object)
        for ind,(TileOffsets, TileByteCounts) in parts_info[level]:
            #note this modifies in place!
            TileOffsets[TileByteCounts > 0] += offset
            TileOffsets[TileByteCounts == 0] = 0 #sparse tif data
            offset = TileOffsets[-1,-1] + TileByteCounts[-1,-1]
            TileOffsets_arr[ind] = TileOffsets
            TileByteCounts_arr[ind] = TileByteCounts
        #make block arrays into arrays and then flatten them
        block_offsets.append(np.block(TileOffsets_arr.tolist()).ravel())
        block_counts.append(np.block(TileByteCounts_arr.tolist()).ravel())

    return (block_offsets, block_counts)
                              
def prep_tiff_header(header_bytes,parts_info):
    '''update the header
    '''                             
    #adjust the block data
    block_data = ifd_offset_adjustments(len(header_bytes),parts_info)
    #write the block data into the header
    with io.BytesIO(header_bytes) as memfile:
        ifd_updater(memfile,block_data)
        delete_COG_ghost_header(memfile)
        header_data = memfile.getvalue()
    return header_data
    
def write_ccog(x_arr,store, COG_creation_options = None, rasterio_env_options = None, storage_options = None):
    '''writes a concatenated COG to S3
    x_arr an xarray array.
        notes on chunking:
        ccog is picky about chunking and will tell you about it.
        chunks should have xa nd y dimensions of multiples of the blocksize
        the last chunk in the x  and y dimensions do not have this limitation.
    store is an s3 file path or fsspec mapping
    storage_options (dict, optional) – Any additional parameters for the storage backend
    COG_creation_options (dict, optional) – options as described here https://gdal.org/drivers/raster/cog.html#creation-options
        unlike gdal resampling defaults to nearest unless specified
        not all creation options are available at this stage.
    rasterio_env_options (dict, optional) –  options as described here https://rasterio.readthedocs.io/en/latest/topics/image_options.html#configuration-options
        gdal exposes many configuration options how (or if) any particular option works with ccog is untested
        ccog does all its work in memory so many of the commonly used gdal config options are unlikely to be useful, but the option is there if you want to try.
        They may be better used with however you open data to make your xarray. 
    
    returns a dask.delayed object
    '''
    #TODO: shortcut route if a small array can just be handled as a single part
    #TODO: work with GDAL config settings (environment)
    
    COG_creation_options = {} if COG_creation_options is None else COG_creation_options
    rasterio_env_options = {} if rasterio_env_options is None else rasterio_env_options
    storage_options = {} if storage_options is None else storage_options
    
    print ('warning ccog is only a proof of concept at this stage.... enjoy')
    
    if not isinstance(x_arr,xarray.core.dataarray.DataArray):
        raise TypeError (' x_arr must be an instance of xarray.core.dataarray.DataArray')
    
    #normalise keys to lower case for ease of referencing
    user_creation_options = {key.lower():val for key,val in COG_creation_options.items() }
    
    #throw error as these options have not been tested and likely wont work as expected
    exclude_opts = [opt for opt in ['warp_resampling','target_srs','tiling_scheme'] if opt in user_creation_options]
    if exclude_opts:
        raise ValueError (f'ccog cant work with COG_creation_options of {exclude_opts}')
                
    if 'overview_resampling' not in user_creation_options and 'resampling' in user_creation_options:
        user_creation_options['overview_resampling'] = user_creation_options['resampling']
        del user_creation_options['resampling']
    
    #todo: raise error if required_creation_options are going to change user_creation_options

    #build the profile from layering profile sources sequentially making sure most important end up with precendence.
    profile = default_creation_options.copy()
    profile.update(xarray_to_profile(x_arr))
    profile.update(user_creation_options)
    profile.update(required_creation_options)
    
    if int(profile['blocksize']) % 16: 
        #the tiff rule is 16 but that is a very small block, resulting in a large header. consider a 256 minimum?
        raise ValueError (f'blocksize must be multiples of 16')

    #check chunking.
    if any([dim%profile['blocksize'] for dim in x_arr.data.chunks[0][:-1]]) or any([dim%profile['blocksize'] for dim in x_arr.data.chunks[1][:-1]]):
        raise ValueError ('chunking needs to be multiples of the blocksize (except the last in any dimension)')

    #building the delayed graph
    delayed_graph = COG_graph_builder(x_arr.data,store,profile=profile,rasterio_env_options=rasterio_env_options,storage_options=storage_options)
    return delayed_graph

