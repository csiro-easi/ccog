__all__ = ['write_ccog']

from contextlib import suppress
from itertools import zip_longest
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



required_creation_options = dict(
    sparse_ok=True,
    driver="COG",
    bigtiff="YES",
    num_threads="all_cpus", #TODO: test what difference this makes if any on a dask cluster
    overviews="AUTO",
)

default_creation_options = dict(
    geotiff_version= 1.1, #why not use the latest by default
    blocksize = 512,
    overview_resampling = rasterio.enums.Resampling.nearest,
    compress = None,
    overview_compress = None,
    COG_ghost_data = False,

)

def get_maximum_overview_level(width, height, minsize=256,overview_count=None):
    """
    Calculate the maximum overview level of a dataset at which
    the smallest overview is smaller than `minsize`.
    
    Based on rasterio.rio.overview.get_maximum_overview_level
    modified to match the behaviour of the gdal COG driver
    so that the smallest overview is smaller than `minsize` in BOTH width and height.
    
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


    overview_level = 0
    overview_factor = 1
    if overview_count is not None:
        while overview_count > overview_level and max(width // overview_factor, height // overview_factor) > 1:
            overview_factor *= 2
            overview_level += 1 
            #print (width // overview_factor, height // overview_factor)
    else:
        while max(width // overview_factor, height // overview_factor) > minsize:
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


def empty_COG(profile,rasterio_env_options=None,mask=False):
    """
    makes an empty sparse COG in memory

    used as a reference to look at the structure of a COG file
    and as the starting point for some file format fiddling.
    
    faster then doing this directly with rasterio

    returns bytes
    """
    #This simply works by calling gdal with rasterio and not writing any data
    #however for large datasets this can be quite slow.
    #so the size of the dataset is reduced so that it still creates all the required overview
    #then tifffile is used to massage the file to the correct dimensions.

    rasterio_env_options = {} if rasterio_env_options is None else rasterio_env_options
    #this is applied elsewhere but apply here for when testing
    profile.update(required_creation_options)
   
    prof = profile.copy()
    #if overview count isnt preset use  max([profile['height'],profile['width']])
    #however note that gdal has an issue that throws an error if dim is over blocksize and overview_count is zero
    prof['height'] = 2**profile['overview_count']
    prof['width'] = 1
    
    with rasterio.Env(**rasterio_env_options):
        with rasterio.io.MemoryFile() as memfile:
            #print (prof)
            with memfile.open(**prof) as src:
                if "colormap" in profile:
                    src.write_colormap(1, profile["colormap"])
                if mask == True:
                    src.write_mask(False)
                #todo include tags,scale,offset,desctription,units etc
            memfile.seek(0)
            data = memfile.read()

            with io.BytesIO(data) as memfileio:
                with tifffile.TiffFile(memfileio) as tif:
                    main = []
                    mask=[]
                    trash_offsets = []
                    for p in tif.pages:
                        if p.tags['TileOffsets'].valueoffset > p.tags['TileOffsets'].offset + 12:
                            trash_offsets.append(p.tags['TileOffsets'].valueoffset)
                        #adjust for gdal cog ghost leader
                        trash_offsets.extend(v-4 for v in p.tags['TileOffsets'].value if v != 0)
                        if p.tags['TileByteCounts'].valueoffset > p.tags['TileByteCounts'].offset + 12:
                            trash_offsets.append(p.tags['TileByteCounts'].valueoffset)
                            
                        tif.pages[p.index].tags['ImageWidth'].overwrite(1)
                        tif.pages[p.index].tags['ImageLength'].overwrite(1)
                        tif.pages[p.index].tags['TileByteCounts'].overwrite([0])
                        tif.pages[p.index].tags['TileOffsets'].overwrite([0])
                                                
                        if 'MASK' in str(p.tags.get('NewSubfileType','')):
                            mask.append(p)
                        else:
                            main.append(p)
                #truncate file to get rid of old offsets and counts data
                if trash_offsets:
                    memfileio.truncate(min(trash_offsets))         
                h = profile['height']
                w = profile['width']
                for main_and_mask in zip_longest(main,mask):
                    num_tiles = math.ceil(h/profile["blocksize"])*math.ceil(w/profile["blocksize"])
                    for p in main_and_mask:
                        #interleaving the main and mask offsets/bytecounts
                        if p is None:
                            #no mask
                            continue
                        #tifffile gave warnings if editing offsets and bytes at once. so opening and closing the file to avoid this
                        memfileio.seek(0)
                        with tifffile.TiffFile(memfileio) as tif:
                            tif.pages[p.index].tags['ImageWidth'].overwrite(w,dtype='I')
                            tif.pages[p.index].tags['ImageLength'].overwrite(h,dtype='I')
                            tif.pages[p.index].tags['TileOffsets'].overwrite([0]*num_tiles)
                        memfileio.seek(0)
                        with tifffile.TiffFile(memfileio) as tif:
                            tif.pages[p.index].tags['TileByteCounts'].overwrite([0]*num_tiles)
                    h =max(1,h//2)
                    w =max(1,w//2)
                memfileio.seek(0)
                if not profile["COG_ghost_data"]:
                    delete_COG_ghost_header(memfileio)
                memfileio.seek(0)
                data_fixed = memfileio.read()
    return data_fixed

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

def test_jpegtables(JPEGTables,profile,rasterio_env_options=None):
    '''tests if jpegtables matches that for an empty tiff
    
    throws an error if there is a difference
    
    used because its not entirely clear what changes the contents of this tag
    stop using this once comfortable that jpegtables doesnt change
    '''
    profile = profile.copy()
    profile['height']= 512
    profile['width']= 512
    with rasterio.Env(**rasterio_env_options):
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as src:
                pass
            memfile.seek(0)
            with tifffile.TiffFile(memfile) as tif:
                empty_JPEGTables = tif.pages[0].tags['JPEGTables'].value
    if empty_JPEGTables != JPEGTables:
        raise ValueError('different JPEGTables')
    

def partial_COG_maker(
    arr,
    mask=None,
    *,
    profile,
    rasterio_env_options,
):
    """
    writes arr to an in memory COG file and the pulls the COG file apart and returns it in parts that are useful for
    reconstituting it differently.
    Gets all the advantages of rasterio/GDALs work writing the file so dont need to do that work from scratch.
    """
    profile = profile.copy() #careful not to update in place
    profile['height']= arr.shape[1]
    profile['width']= arr.shape[2]
    #not using the transform in profile as its going to be wrong - make it obviously wrong avoids rasterio warnings
    profile['transform']= Affine.scale(2)
    #the overview is only used for resampling its not stored
    profile["overview_compress"] = 'NONE'
    profile["overview_count"] = 1
    overview_arr = None
    mask_overview_arr = None
    if profile['height']==1 and profile['width']==1:
        #stops gdal throwing an error
        #in the case of a 1x1 pixel input return it as the overview
        profile["overview_count"] = 0
        overview_arr = arr
        mask_overview_arr = mask
        
    with rasterio.Env(**rasterio_env_options):
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as src:
                src.write(arr)
                del arr  # reducing mem footprint
                if mask:
                    src.write_mask(mask)
                    del mask
            
            if profile["overview_count"]:
                with memfile.open(overview_level=0) as src:
                    # get the required overview back as an array
                    overview_arr = src.read()
                    if mask:
                        #mask is the same for all bands
                        mask_overview_arr = src.read_masks(1)

            #removing the gdal ghost leaders
            part_bytes = []
            memfile.seek(0)
            with tifffile.TiffFile(memfile) as tif:
                page = tif.pages[0]
                if profile['compress'] == 'jpeg':
                    test_jpegtables(page.tags['JPEGTables'].value,profile,rasterio_env_options)
            for offset, bytecount in zip(page.dataoffsets, page.databytecounts):
                _ = memfile.seek(offset)
                part_bytes.append(memfile.read(bytecount))
    
    todo - extract bytes and offsets/counts for masks
    
    
    # data will be more flexible later as 2d numpy arrays
    tile_dims_count = (math.ceil(page.imagelength/page.tilelength),math.ceil(page.imagewidth/page.tilewidth))
   
    databytecounts = np.array(page.databytecounts,dtype=np.int64).reshape(*tile_dims_count)
    #note: at a later stage sparse tiles (bytecount=0) have their offset set to zero before writing.
    databyteoffsets = np.cumsum([0,*page.databytecounts[0:-1]],dtype=np.int64).reshape(*tile_dims_count)
    part_info = (databyteoffsets, databytecounts)
    return overview_arr,mask_overview_arr,part_bytes,part_info

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

def COG_graph_builder(da,mask,profile,rasterio_env_options):
    '''
    makes a dask delayed graph that when computed writes a COG file to S3
    '''
    
    tif_part_maker_func = dask.delayed(partial_COG_maker, nout=4)

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
        chunk_depth, chunk_heights, chunk_widths = da.chunks
        #matching gdal behaviour
        if len(chunk_heights)>1 and chunk_heights[-1] ==1:
            chunk_heights = chunk_heights[:-1]
        if len(chunk_widths)>1 and chunk_widths[-1] ==1:
            chunk_widths = chunk_widths[:-1]
        
        #this is the slowest line by far. if not optimize_graph its faster but slower overall.
        #if optimize_graph=True then the graph ends up reading in the source data many times
        #if optimize_graph=False then the data is read more optimally but the building of the graph gets slower.
        data_del = da.to_delayed(optimize_graph=False)
        res_arr = np.ndarray((len(chunk_heights),len(chunk_widths)), dtype=object)
        
        if mask:
            mask_del = mask.to_delayed(optimize_graph=False)
            data_del = zip(data_del.ravel(),mask_del.ravel())
            mask_res_arr = np.ndarray((len(chunk_heights),len(chunk_widths)), dtype=object)

        for blk in data_del:
            ind = blk[0].key[1:][-2:]
            overview_arr,mask_arr,part_bytes,part_info = tif_part_maker_func(*blk, profile=current_level_profile,rasterio_env_options=del_rasterio_env_options)
            parts_info[level].append((ind,part_info))
            parts_bytes[level].append(part_bytes)
            
            #checks if the index falls within the array
            #throwing away overviews that are an artifact of the way gdal produces overviews when there is a dimension of 1
            if len([1 for s,i in zip(res_arr.shape[-2:],ind) if i<s])==2:
                blk_final_shape = (chunk_depth[0],max(1,chunk_heights[ind[0]] // 2),max(1,chunk_widths[ind[1]] // 2))
                res_arr[ind] = dask.array.from_delayed(overview_arr, shape=blk_final_shape, dtype=da.dtype)
                if mask:
                    mask_res_arr[ind] = dask.array.from_delayed(mask_arr, shape=blk_final_shape, dtype=mask.dtype)

        da = dask.array.block(res_arr.tolist())
        #copy the chunking from the initial data - assuming the first chunk is indicative of appropriate chunking to use for overviews
        da = da.rechunk(chunks= (chunk_depth[0],chunk_heights[0], chunk_widths[0]))
        
        if mask:
            mask = dask.array.block(mask_res_arr.tolist())
            #copy the chunking from the initial data - assuming the first chunk is indicative of appropriate chunking to use for overviews
            mask = mask.rechunk(chunks= (chunk_heights[0], chunk_widths[0]))
        
        current_level_profile = del_overview_profile

    header_bytes_final = dask.delayed(prep_tiff_header)(parts_info,del_profile,del_rasterio_env_options)
    #sum as a shorthand for flattening a list in the right order
    delayed_parts = sum([v for k,v in sorted(parts_bytes.items(), reverse=True)],[header_bytes_final])
    return delayed_parts


def ifd_updater(memfile,block_data):
    #memfile is a bytesIO object
    assert not memfile.closed, "memory file was closed before writing"
    block_offsets,block_counts = block_data
    with tifffile.TiffFile(memfile) as tif:
        for page,offs,cnts in zip(tif.pages,block_offsets,block_counts):
            #print (page.tags)
            #print (offs)
            #print (cnts)
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
            
            #reset the offset for the next level
            offset += np.sum(TileByteCounts)
            
            TileOffsets_arr[ind] = TileOffsets
            TileByteCounts_arr[ind] = TileByteCounts
            
#         print ( level)    
#         print (TileOffsets_arr)
#         print (TileByteCounts_arr)
        #make block arrays into arrays and then flatten them
        block_offsets.append(np.block(TileOffsets_arr.tolist()).ravel())
        block_counts.append(np.block(TileByteCounts_arr.tolist()).ravel())

    #reverse as the header pages are in reverse order
    return (block_offsets[::-1], block_counts[::-1])
                              
def prep_tiff_header(parts_info,del_profile,rasterio_env_options):
    '''update the header
    '''                  
    #empty_COG is slow for large COGs, its seperate here so dask can run it early
    header_bytes = empty_COG(del_profile, rasterio_env_options=rasterio_env_options)
    #adjust the block data
    #print (parts_info)
    block_data = ifd_offset_adjustments(len(header_bytes),parts_info)

    #write the block data into the header
    with io.BytesIO(header_bytes) as memfile:
        ifd_updater(memfile,block_data)
        memfile.seek(0)
        header_data = memfile.getvalue()
    return header_data
    
def write_ccog(x_arr,store, mask=None, COG_creation_options = None, rasterio_env_options = None, storage_options = None):
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
    #todo:
    #In addition to the gdal option an optional colormap is accepted. This only works with a single int8 band.
    #eg colormap = {0:(4,0,0,4),255:(0,0,4,4)}
    #for more info see https://rasterio.readthedocs.io/en/stable/topics/color.html
    
    #TODO: shortcut route if a small array can just be handled as a single part
    #TODO: work with GDAL config settings (environment)
    
    COG_creation_options = {} if COG_creation_options is None else COG_creation_options
    rasterio_env_options = {} if rasterio_env_options is None else rasterio_env_options
    storage_options = {} if storage_options is None else storage_options
    
    print ('warning ccog is only a proof of concept at this stage.... enjoy')
    
    if not isinstance(x_arr,xarray.core.dataarray.DataArray):
        raise TypeError (' x_arr must be an instance of xarray.core.dataarray.DataArray')
        

    
    #normalise keys and string values to lower case for ease of use
    user_creation_options = {key.lower():val.lower() if isinstance(val,str) else val for key,val in COG_creation_options.items() }
    
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


    #handle single band data the same as multiband data
    if len(x_arr.dims) == 2:
        x_arr = x_arr.expand_dims('band')
    #check chunking.
    if any([dim%profile['blocksize'] for dim in x_arr.data.chunks[-2][:-1]]) or any([dim%profile['blocksize'] for dim in x_arr.data.chunks[-1][:-1]]):
        raise ValueError ('chunking needs to be multiples of the blocksize (except the last in any spatial dimension)')
    if len(x_arr.data.chunks) ==3 and len(x_arr.data.chunks[0]) > 1:
        raise ValueError ('non spatial dimension chunking needs to be a single chunk')
        
    #mask - check
    if mask is not None:
        if not isinstance(mask,xarray.core.dataarray.DataArray):
            raise TypeError (' mask must be an instance of xarray.core.dataarray.DataArray')
        if x_arr.data.chunks[-2] != mask.data.chunks:
            raise ValueError ('mask spatial chunks needs to match those of x_arr')
        #todo: also check CRS and transform match
        mask = mask.data
    
    #building the delayed graph
    delayed_parts = COG_graph_builder(x_arr.data,mask,profile=profile,rasterio_env_options=rasterio_env_options)
    delayed_graph = aws_tools.mpu_upload_dask_partitioned(delayed_parts,store,storage_options=storage_options)
    return delayed_graph

