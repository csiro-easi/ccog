__all__ = ['write_ccog']

from contextlib import suppress
from itertools import count
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
    num_threads="all_cpus",
    overviews="AUTO",
)

default_creation_options = dict(
    geotiff_version= 1.1, #why not use the latest by default
    blocksize = 512,
    overview_resampling = rasterio.enums.Resampling.nearest,

)

#thoughts around how to
# break a COG down to good sized chunks
#
# its a balance between:
# s3 multipart upload limits minimum size of 5MiB per part (note total combined maximum size of 5TiB)
# and maximum of 10,000 parts
# bigger parts get more overviews calculated and therefore less data needs to be transfured to do courser overview creation
# spreading the load between dask workers/threads - ideally 1 or a few parts per thread.
# not overloading worker memory
# working with source data chunks - cant really adjust for this as the COG output tile/overview structure is very strict.
# chunks MUST be aligned to one of the overview levels
# Therefore, I think the goal is big chunks that dont overload memory.
# in the place of automatically analysing the specs of the cluster tune for a typical cluster worker
# For a small array on a big cluster this may leave some workers idle. so be it.


#related links/projects
#cog standard https://portal.ogc.org/files/102116?utm_source=phplist879&utm_medium=email&utm_content=HTML&utm_campaign=OGC+seeks+public+comment+on+Cloud+Optimized+GeoTIFF+%28COG%29+Standard
#https://pypi.org/project/large-image-converter/
#interesting tequnique for partially updateing a s3 key https://stackoverflow.com/questions/38069985/replacing-bytes-of-an-uploaded-file-in-amazon-s3
#cogger https://github.com/airbusgeo/cogger
#rioxarray
#https://pypi.org/project/tifftools/
#https://github.com/cgohlke/tifffile
#a timely warning about s3 costs https://pancho.dev/posts/watch_out_cloud_storage/
#many others

#waiting on gdal 3.6
#it allows my to force creation of overview levels beyond what is needed for normal users
#it fixed a writing bug i identified for long thin COGs
#however for now as long as the input data fills full sized chunks at the level used then ccog should work fine

#not tested with
# more then 1 band
# with masks
# dtypes other then float32
#different copressions schemes - shouldnt be any issue
# a range of resample types - some resampling types may not transistion well between the finer levels and the course levels.
#various nodata values - handling of nodata is a bit haphazard - should work with np.nan
#different compression for different tiff pages

#interesting things to consider
#  multi multi part upload !! https://en.wikipedia.org/wiki/Epizeuxis
#- multipart upload to s3 is limited to 10000 parts. to get around this parts can be put into multiple temp files on s3 and then
#     a new multipart file is produced and s3.upload_part_copy can be used to assemble the temp files
# - a similar approach could be used to rearrange a ccog file into a stricter cog format. as a post processign step.
#    care would need to be taken to not cross the min/max part size limits
#    max could be avoided jsut by splitting a part
#    min could be avoided by downloading enough small bits locally to be able to provide a 50MiB part and then upload_part.

def get_maximum_overview_level(width, height, minsize=256):
    """
    Based on rasterio.rio.overview.get_maximum_overview_level
    modified to match the behaviour of the gdal COG driver
    so that the smallest overview is smaller than `minsize` in BOTH width and height.
    
    
    Calculate the maximum overview level of a dataset at which
    the smallest overview is smaller than `minsize`.
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
    overview_level = 0
    overview_factor = 1
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
    #     Of particular benefit is the inclusion of an OverviewList and then using the COG driver
    #     option of OVERVIEWS = 'FORCE_USE_EXISTING'
    #     It seems that building overviews, even if empty and sparse is the slow part of making a COG.

    #     An alternative approach to produce the vrt xml written in this code
    #     is to use a dummy dataset and BuildVRT and then
    #     build_overviews with VRT_VIRTUAL_OVERVIEWS = 'YES'and then editing the VRT to remove the dummy dataset
    #     The code below produces an output identical to the above steps but this implementation 
    #     is prefered as it is more direct and hopefully clearer
    
    overview_level = get_maximum_overview_level(profile['width'], profile['height'], minsize=profile['blocksize'])
    overviews = " ".join([str(2**j) for j in range(1, overview_level + 1)])

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

def partial_COG_maker(
    arr,
    writeLastOverview,
    profile,
    rasterio_env_options,
):
    """
    writes arr to a in memory COG file and the pulls the COG file apart and returns it in parts that are useful for
    reconstituting it differently.
    Gets all the advantages of GDALs work writing the file so dont need to do that work from scratch.
    writeLastOverview If False the last overview is not included in imagedata ,TileOffsets, TileByteCounts

    """
    profile = profile.copy() #careful not to update in place
    profile['height']= arr.shape[0]
    profile['width']= arr.shape[1]
    #not using the transform in profile as its going to be wrong - make it obviously wrong avoids rasterio warnings
    profile['transform']= Affine.scale(2)

    with rasterio.Env(**rasterio_env_options):
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as src:
                src.write(arr, 1)
                del arr  # reducing mem footprint
                
                #gdal <3.6 doesnt understand "overview_count" so 
                #forcing overview building for the number of levels needed
                #next  2 lines can likely be removed if support is limited to gdal >= 3.6
                factors = [2**j for j in range(1, profile['overview_count'] + 1)]
                src.build_overviews(factors, profile['overview_resampling'])

            with memfile.open(overview_level=profile['overview_count'] - 1) as src:
                # get the required overview back as an array
                outarr = src.read(1)
 
            memfile.seek(0)

            #gdal/rasterio dont make it easy to access and manipulate offsets and counts tifffile does :-)
            lim = None if writeLastOverview else profile['overview_count']
            with tifffile.TiffFile(memfile) as tif:
                TileOffsets = [page.tags['TileOffsets'].value for page in tif.pages[0:lim]]
                TileByteCounts = [page.tags['TileByteCounts'].value for page in tif.pages[0:lim]]

            # assume the very strict COG layout from GDAL get the image data for all except the highest overview
            # keep the gdal ghost leader - not sure its going to be much use but all the other tiles will have it so keep it for the first one to be consistant
            #keeping it also maintains the offset of '0' as having special meaning
            gdal_ghost_offset = 4
            # find the offset to the first tile of image data
            pos_offsets = [i for l in TileOffsets for i in l if i > 0]  
            imagedata = b''
            if len(pos_offsets) > 0:
                # its not all nodata
                image_offset = min(pos_offsets) - gdal_ghost_offset
                TileOffsets = [[v - image_offset if v else 0 for v in l] for l in TileOffsets]
                memfile.seek(image_offset)
                imagedata = memfile.read()

            # offsets and counts will be more useful as 2d numpy arrays
            # level -1 is the full res data - also accessed with overview_level=None
            for ov in range(len(TileOffsets)):
                with memfile.open(overview_level=ov - 1) as src:
                    last_window_indicies = list(src.block_windows(-1))[-1][0]
                    blocks_shp = (last_window_indicies[0]+1,last_window_indicies[1]+1)
                    TileOffsets[ov] = np.array(TileOffsets[ov], dtype=np.int64).reshape(blocks_shp)
                    TileByteCounts[ov] = np.array(TileByteCounts[ov], dtype=np.int64).reshape(blocks_shp)

    return (outarr, TileOffsets, TileByteCounts, imagedata)

def tif_part_writer(
    block, i, mpu_store, writeLastOverview, pad, profile, rasterio_env_options,
):
    
    outarr, TileOffsets, TileByteCounts, imagedata = partial_COG_maker(
        block, writeLastOverview=writeLastOverview, profile=profile, rasterio_env_options=rasterio_env_options,
    )
    #only make a part if there is data in the part
    part = None
    if len(imagedata):
        if pad:
            # extend the file so that imagedata is at least pad bytes long due to s3 minimum file size
            imagedata = imagedata.ljust(pad, b"\0")
        assert len(imagedata) <= s3_max_part_bytes , 'part too big for s3 part upload'
        part = mpu_store.upload_part_mpu(i, imagedata)

    data_len = len(imagedata)
    #print (i,writeLastOverview,profile)
    return outarr, (part, data_len, TileOffsets, TileByteCounts)

def modify_COG_ghost_header(memfile):
    '''If messing with the gdal COG block order optimisations its likely a good idea to let GDAL know about it in the ghost header
    its a little annoying to have to fully switch off the optimisations as its still 'mostly' got all the optimisations in place
    Ive not tested what GDAL does if if I dont switch off this optimisation. 
    I would hope that it has some sanity checks that stop to deal with blocks that are out of order and doesnt try and read massive amounts of data for getting a small tile?
    quick and dirty way to do it but assumes this structure isnt likely to change and if it does and breaks this code then likely other assumptions need fixing.
    todo: ask gdal maintainers about doing this and its impacts.
    '''
    assert not memfile.closed, "memory file was closed before writing"
    memfile.seek(83)
    assert memfile.read(21) == b'BLOCK_ORDER=ROW_MAJOR' , 'oh no gdal COG format changed'
    memfile.seek(83)
    memfile.write(b'BLOCK_ORDER=CCOG     ')#trailing space intentional - ive just made CCOG up - wonder if GDAL will even notice
    memfile.seek(168)
    assert memfile.read(31) == b'KNOWN_INCOMPATIBLE_EDITION=NO\n ' , 'oh no gdal COG format changed'
    memfile.seek(168)
    memfile.write(b'KNOWN_INCOMPATIBLE_EDITION=YES\n') #trailing space removed just like gdal would - NOTE: This change will result in GDAL giving warnings
    memfile.seek(0)
    return

def chunk_dim(blockSize,chunk_exp):
    return blockSize * 2 ** chunk_exp

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

def chunk_adjuster(dask_arr,blockSize,chunk_exp = 0):
    '''
    analyse chunks and combine edge chunks in some limited cases
    '''
    chunk_heights,chunk_widths = dask_arr.chunks
    if chunk_exp<1:
        chunk_exp = get_maximum_overview_level(chunk_widths[0], chunk_heights[0], minsize=blockSize)
    #assert chunk_exp >= 1, 'chunk_exp too small'
    #if chunk_exp == 2:
    #    print('consider using larger chunks') #shift to logging system
    #get here if there is more then a single chunk
    #very small last chunks trigger different handling in gdal overview creation
    #merge the second to last chunks if too small
    if chunk_heights[-1] < 2**chunk_exp:
        chunk_heights = (*chunk_heights[:-2],sum(chunk_heights[-2:]))
    if chunk_widths[-1] < 2**chunk_exp:
        chunk_widths = (*chunk_widths[:-2],sum(chunk_widths[-2:]))
    dask_arr = dask_arr.rechunk(chunks=(chunk_heights,chunk_widths))
    return(dask_arr,chunk_exp)

def nested_graph_builder(da,mpu_store,profile,rasterio_env_options):
    profile = profile.copy()

    parts_info = []
    # part ids in reverse so higher resolution data is towards the end of the cog
    mpu_part_id_generator = count(10000,-1)
    for level in count():
        if level > 0:
            adjust_compression(profile)
        da, chunk_exp = chunk_adjuster(da, profile['blocksize'],chunk_exp = 0)
        profile["overview_count"] = chunk_exp

        chunk_heights, chunk_widths = da.chunks
        da_del = da.to_delayed(optimize_graph=False)
        last = get_maximum_overview_level(chunk_widths[0], chunk_heights[0], minsize=profile['blocksize']) == 0
        res_arr = np.ndarray(da_del.shape, dtype=object)
        #flipping as the mpu_part_id_generator decrements the counter
        for mpu_part_id, blk in zip(mpu_part_id_generator, np.flip(da_del.ravel())):
            ind = blk.key[1:]
            blk_final_shape = (
                chunk_heights[ind[0]] // 2**chunk_exp,
                chunk_widths[ind[1]] // 2**chunk_exp,
            )

            part_writer_func = dask.delayed(tif_part_writer, nout=2)
            outarr, part_info = part_writer_func(
                blk,
                i=mpu_part_id,
                mpu_store = mpu_store,
                writeLastOverview=last, 
                pad=s3_min_part_bytes,
                profile = profile,
                rasterio_env_options = rasterio_env_options,
                #dask_key_name=("tif_part_writer",level,ind,mpu_part_id,tokenize(ind,level,mpu_part_id,mpu_store.mpu))
            )
            res_arr[ind] = dask.array.from_delayed(outarr, shape=blk_final_shape, dtype=da.dtype)
            parts_info.append((level,ind,mpu_part_id,part_info))
        if last:
            break
        da = dask.array.block(res_arr.tolist())
        #copy the chunking from the initial data
        da = da.rechunk(chunks= (chunk_heights[0], chunk_widths[0]))

    #print (mpu_part_id)
    if mpu_part_id < s3_start_part:
        raise ValueError ('too many parts - try starting with larger chunks')
    return parts_info

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
    #apply cumulative offsets to tile offset data   
    offset = header_length
    #sorted by mpu_part_id
    for level,ind,mpu_part_id,(part, data_len, TileOffsets, TileByteCounts) in sorted(parts_info, key= lambda y: y[2]):
        for arr in TileOffsets:
            #note this modifies in place!
            arr[arr > 0] += offset
        offset += data_len

    #rearrange info arrays into a format that suits.
    rearranged_info = {}
    for level,ind,mpu_part_id,(part, data_len, TileOffsets, TileByteCounts) in sorted(parts_info):
    #for level,ind,_,TileOffsets,TileByteCounts in sorted(parts_info.values()):
        for level2,(to,tb) in enumerate(zip(TileOffsets,TileByteCounts)):
            rearranged_info.setdefault((level,level2), []).append((ind,to,tb))

    #second stage of rearranging into the final arrangement that is needed for the tif headers
    block_offsets = []
    block_counts = []
    for v in rearranged_info.values():
        #rely on things being correctly sorted
        #get the last 2d index and  convert it to a 2d shape
        shp = (v[-1][0][0]+1,v[-1][0][1]+1)
        #place all the blocks into 2d arrays
        TileOffsets_arr = np.ndarray(shp, dtype=object)
        TileByteCounts_arr = np.ndarray(shp, dtype=object)
        for ind,TileOffsets,TileByteCounts in v:
            TileOffsets_arr[ind] = TileOffsets
            TileByteCounts_arr[ind] = TileByteCounts
        #make block arrays into arrays and then flatten them
        block_offsets.append(np.block(TileOffsets_arr.tolist()).ravel())
        block_counts.append(np.block(TileByteCounts_arr.tolist()).ravel())

    return (block_offsets, block_counts)
                              
def write_tiff_header(mpu_store,header_bytes,parts_info):
    '''update the header and finish writing to the mpu
    '''                             
    #extend so at least s3_min_part_bytes long due to s3 minimum file size
    header_bytes = header_bytes.ljust(s3_min_part_bytes, b"\0")
    #adjust the block data
    block_data = ifd_offset_adjustments(len(header_bytes),parts_info)
    #write the block data into the header
    with io.BytesIO(header_bytes) as memfile:
        ifd_updater(memfile,block_data)
        modify_COG_ghost_header(memfile)
        header_data = memfile.getvalue()

    assert len(header_data) <= s3_max_part_bytes , 'part too big for s3 part upload'
    parts = [part for level,ind,mpu_part_id,(part, data_len, TileOffsets, TileByteCounts) in parts_info]
    #write the header to the first (1) mpu part
    parts.append(mpu_store.upload_part_mpu(1, header_data))
    mpu_store.complete_mpu(parts)
    
def chunky_checker(dask_arr,blockSize):
    '''ccog has specific chunking requirements
    
    this checks that they are met
    '''
    valid_chunk_dims = [blockSize* 2**e for e in range(30)]#use a range bigger then ever expected
    if len(dask_arr.chunks) != 2:
        raise ValueError ('currently only works with 2d arrays')
    chunk_heights,chunk_widths = dask_arr.chunks
    if len(set(chunk_heights[:-1])) > 1 or len(set(chunk_widths[:-1]))  > 1:
        raise ValueError ('chunking needs to be consistant (except the last in any dimension)')
    if len(chunk_heights)>1 and len(chunk_widths)>1:
        if chunk_heights[0] != chunk_widths[0]:
            raise ValueError ('chunking needs to be square')
    if len(chunk_heights)>1 :
        if chunk_heights[-1] >= 2 * chunk_heights[0]:
            raise ValueError ('last chunk too large')
        if chunk_heights[0] not in valid_chunk_dims[1:] :
            raise ValueError ('chunk size needs to be in the power of 2 series starting after blockSize')
    if len(chunk_widths)>1 :
        if chunk_widths[-1] >= 2 * chunk_widths[0]:
            raise ValueError ('last chunk too large')
        if chunk_widths[0] not in valid_chunk_dims[1:] :
            raise ValueError ('chunk size needs to be in the power of 2 series starting after blockSize')

def write_ccog(x_arr,store, COG_creation_options = None, rasterio_env_options = None, storage_options = None):
    '''writes a concatenated COG to S3
    x_arr an xarray array.
        notes on chunking:
        ccog is picky about chunking and will tell you about it.
        most chunks should be square and with a dimension powers of 2 of block size 
        so if your block size is 512 then chunks of one of 1024,2048,4096,8192,16384....etc
        larger chunks are better - limited to your available ram
        small chunks may result in some bloat in the final file or trigger some errors due to limitations in s3 uploads.
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
    
    TODO: shortcut route if a small array can just be handled as a single part
    TODO: work with GDAL config settings (environment)
    '''
    COG_creation_options = {} if COG_creation_options is None else COG_creation_options
    rasterio_env_options = {} if rasterio_env_options is None else rasterio_env_options
    storage_options = {} if storage_options is None else storage_options
    
    print ('warning ccog is only a proof of concept at this stage.... enjoy')
    
    if not isinstance(x_arr,xarray.core.dataarray.DataArray):
        raise TypeError (' x_arr must be an instance of xarray.core.dataarray.DataArray')
    
    #normalise keys to lower case for ease of referencing
    user_creation_options = {key.lower():val for key,val in COG_creation_options.items() }
    
    #throw error as these options not been tested and may not work
    #overview_count is used internally
    exclude_opts = [opt for opt in ['warp_resampling','target_srs','tiling_scheme','overview_count'] if opt in user_creation_options]
    if exclude_opts:
        raise ValueError (f'ccog cant work with COG_creation_options of {exclude_opts}')
                
    if 'overview_resampling' not in user_creation_options and 'resampling' in user_creation_options:
        user_creation_options['overview_resampling'] = user_creation_options['resampling']
        del user_creation_options['resampling']
    
    #todo: error if required_creation_options are goign to change user_creation_options

    #build the profile from layering profile sources sequentially making sure most important end up with precendence.
    profile = default_creation_options.copy()
    profile.update(xarray_to_profile(x_arr))
    profile.update(user_creation_options)
    profile.update(required_creation_options)
    
    if int(profile['blocksize']) % 256: 
        #really the tiff rule is 16 but that is a very small block
        raise ValueError (f'blocksize must be multiples of 256')

    #check chunking.
    chunky_checker(x_arr.data,profile['blocksize'])
    
    #building the delayed graph
    mpu_store = dask.delayed(aws_tools.Mpu)(store,storage_options=storage_options)
    parts_info = nested_graph_builder(x_arr.data,mpu_store,profile=profile,rasterio_env_options=rasterio_env_options)

    #empty_single_band_COG is slow for large COGs
    #if its not part of write_tiff_header it runs concurrently with other jobs.
    header_bytes = dask.delayed(empty_single_band_COG)(profile, rasterio_env_options= rasterio_env_options)
    delayed_graph = dask.delayed(write_tiff_header)(mpu_store, header_bytes, parts_info)
    return delayed_graph

