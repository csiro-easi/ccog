__all__ = ["write_ccog"]

import io
import math
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple, Union

import dask
import dask.array as da
import numpy as np
import rasterio
import rioxarray
import tifffile
import xarray

# from rasterio.rio.overview import get_maximum_overview_level
from affine import Affine
from dask.delayed import Delayed

from . import aws_tools

required_rasterio_env_options = dict(gdal_tiff_internal_mask=True)  # only needed if switching to use the Gtiff driver

required_creation_options = dict(
    driver="cog",
    bigtiff="yes",
    # num_threads="all_cpus",  # TODO: test what difference this makes if any on a dask cluster
    tiled="yes",  # only needed if switching to use the Gtiff driver
    overviews="auto",
    sharing = False,
)

default_creation_options = dict(
    sparse_ok=True,
    geotiff_version=1.1,  # why not use the latest by default
    blocksize=512,
    cog_ghost_data=False,
)

# these must be even
# based on testing with GDAL
resample_overlaps = {
    "nearest": 0,
    "average": 0,
    "mode": 0,
    "rms": 0,
    "bilinear": 2,
    "cubic": 4,
    "cubicspline": 4,
    "lanczos": 6,
    "gauss":2,
}


def _get_maximum_overview_level(
    width: int, height: int, minsize: int = 256, overview_count: Optional[int] = None
) -> int:
    """
    Calculate the maximum overview level of a dataset at which
    the smallest overview is smaller than `minsize`.

    Based on rasterio.rio.overview.get_maximum_overview_level
    modified to match the behavior of the gdal COG driver
    so that the smallest overview is smaller than `minsize` in BOTH width and height.

    Parameters:
    ----------
    width : int
        Width of the dataset.
    height : int
        Height of the dataset.
    minsize : int (default: 256)
        Minimum overview size.
    overview_count: Optional[int] (default: None)
        The maximum overview level to generate.

    Returns:
    -------
    overview_level: int
        Overview level.
    """
    # note GDAL seems to have a limit of an overview level of 30

    # GDAL will keep producing overviews for single pixel strands of data longer then blocksize.
    # this is a bit odd because the resampling doesnt make sense

    overview_level = 0
    overview_factor = 1
    if overview_count is not None:
        while overview_count > overview_level and max(width // overview_factor, height // overview_factor) > 1:
            overview_factor *= 2
            overview_level += 1
            # print (width // overview_factor, height // overview_factor)
    else:
        while max(width // overview_factor, height // overview_factor) > minsize:
            overview_factor *= 2
            overview_level += 1

    return overview_level


def _empty_COG(profile: dict, rasterio_env_options: Optional[Dict] = None, mask: Optional[bool] = False) -> bytes:
    """
    Makes an empty sparse COG in memory.

    Used as a reference to look at the structure of a COG file
    and as the starting point for some file format fiddling.

    Faster than doing this directly with rasterio.

    Output should match the output from this code:

    with rasterio.Env(**rasterio_env_options):
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as src:
                pass
            memfile.seek(0)
            data_rio = memfile.read()

    Note: it very slightly differs in some cases (where the ifd length isn't divisible by 2 - jpeg compressing seems to have this issue)
    where tifffile and gdal layout 1 byte differently.

    Parameters:
    ----------
    profile : dict
        The profile for the COG.
    rasterio_env_options : Optional[Dict], (default: None)
        Additional options for the rasterio environment.
    mask : bool, optional
        Whether to use a mask, by default False.

    Returns:
    -------
    bytes
        The bytes representing the COG.
    """
    # This simply works by calling gdal with rasterio and not writing any data
    # however for large datasets this can be quite slow.
    # so the size of the dataset is reduced so that it still creates all the required overview
    # then tifffile is used to massage the file to the correct dimensions.

    rasterio_env_options = {} if rasterio_env_options is None else rasterio_env_options

    prof = profile.copy()
    # Note that gdal has an issue that throws an error if dim is over blocksize and overview_count is zero
    prof["height"] = 2 ** profile["overview_count"]
    prof["width"] = 1

    with rasterio.Env(**rasterio_env_options):
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**prof) as src:
                _add_metadata(src,profile)
                if mask:
                    src.write_mask(False)
                # todo include tags,scale,offset,desctription,units etc
            memfile.seek(0)
            data = memfile.read()

    with io.BytesIO(data) as memfileio:
        with tifffile.TiffFile(memfileio) as tif:
            main_page_ids = []
            mask_page_ids = []
            trash_offsets = []
            for p in tif.pages:
                if p.tags["TileOffsets"].valueoffset > p.tags["TileOffsets"].offset + 12:
                    trash_offsets.append(p.tags["TileOffsets"].valueoffset)
                # adjust for gdal cog ghost leader
                trash_offsets.extend(v - 4 for v in p.tags["TileOffsets"].value if v != 0)
                if p.tags["TileByteCounts"].valueoffset > p.tags["TileByteCounts"].offset + 12:
                    trash_offsets.append(p.tags["TileByteCounts"].valueoffset)

                tif.pages[p.index].tags["ImageWidth"].overwrite(1)
                tif.pages[p.index].tags["ImageLength"].overwrite(1)
                tif.pages[p.index].tags["TileByteCounts"].overwrite([0])
                tif.pages[p.index].tags["TileOffsets"].overwrite([0])

                if "MASK" in str(p.tags.get("NewSubfileType", "")):
                    mask_page_ids.append(p.index)
                else:
                    main_page_ids.append(p.index)
        # truncate file to get rid of old offsets and counts data
        if trash_offsets:
            memfileio.truncate(min(trash_offsets))
        h = profile["height"]
        w = profile["width"]
        for main_and_mask in zip_longest(main_page_ids, mask_page_ids):
            num_tiles = math.ceil(h / profile["blocksize"]) * math.ceil(w / profile["blocksize"])
            for p in main_and_mask:
                # interleaving the main and mask offsets/bytecounts
                if p is None:
                    # no mask
                    continue
                # tifffile gave warnings if editing offsets and bytes at once. so opening and closing the file to avoid this
                memfileio.seek(0)
                with tifffile.TiffFile(memfileio) as tif:
                    tif.pages[p].tags["ImageWidth"].overwrite(w, dtype="H" if w < 2**16 else "I")
                    tif.pages[p].tags["ImageLength"].overwrite(h, dtype="H" if h < 2**16 else "I")
                    tif.pages[p].tags["TileOffsets"].overwrite([0] * num_tiles)
                memfileio.seek(0)
                with tifffile.TiffFile(memfileio) as tif:
                    # i see no reason for the single tile long8 dtype but its what gdal does and matching it makes testing easier
                    tif.pages[p].tags["TileByteCounts"].overwrite([0] * num_tiles, dtype="Q" if num_tiles == 1 else "I")
            h = max(1, h // 2)
            w = max(1, w // 2)
        memfileio.seek(0)
        if not profile["cog_ghost_data"]:
            _delete_COG_ghost_header(memfileio)
        memfileio.seek(0)
        data_fixed = memfileio.read()
    return data_fixed



def _add_metadata(rasterio_src,profile):
    """Adds various metadata to the header"""
    #profile first so others have higher priority (a number of them are tags anyway)
    if "update_tags" in profile:  
        for k,v in profile['update_tags'].items():
            #i could not get namespaces to work -rasterio bug?
            rasterio_src.update_tags(k,**v)
            
    if "descriptions" in profile:
        rasterio_src.descriptions = profile['descriptions']
    if "offsets" in profile:
        rasterio_src.offsets = profile['offsets']
    if "scales" in profile:
        rasterio_src.scales = profile['scales']
    if "units" in profile:
        rasterio_src.units = profile['units']
    if "colorinterp" in profile:
        rasterio_src.colorinterp = profile['colorinterp']
    if "write_colormap" in profile:
        #note: rasterio suggests multiple colormaps can be a thing. I cant make it work -rasterio bug?
        #for k,v in profile['write_colormap'].items():
        #    rasterio_src.write_colormap(k,v)
        rasterio_src.write_colormap(1,profile['write_colormap'])


def _delete_COG_ghost_header(memfile: io.BytesIO):
    """Delete the COG ghost header from a memory file.

    If messing with the gdal COG block order optimisations its likely a good idea to let GDAL know about it in the ghost header
    its a little annoying to have to fully switch off the optimisations as its still 'mostly' got all the optimisations in place
    Ive not tested what GDAL does if if I dont switch off this optimisation.
    GDAL complains if I only modify the header so im removing it fully. Leaving the space as its only a few hundred bytes.

    Args:
    memfile (io.BytesIO): The memory file containing the COG.

    Returns:
    None
    """
    assert not memfile.closed, "memory file was closed before writing"
    memfile.seek(16 + 30)
    GDAL_STRUCTURAL_METADATA_SIZE = int(memfile.read(6))
    memfile.seek(16)
    memfile.write((43 + GDAL_STRUCTURAL_METADATA_SIZE) * b"\0")
    memfile.seek(0)
    return


def _test_jpegtables(JPEGTables: bytes, profile: Dict, rasterio_env_options: Optional[Dict] = None):
    """Test if JPEGTables match those of an empty TIFF.

    This function compares the JPEGTables of a COG with the JPEGTables of an empty TIFF.
    If there is a difference, it raises a ValueError.

    Args:
        JPEGTables (bytes): The JPEGTables to test.
        profile (Dict): The profile of the COG.
        rasterio_env_options (Optional[Dict], optional): Rasterio environment options. Defaults to None.

    Raises:
        ValueError: If the JPEGTables are different.

    Returns:
        None
    """
    profile = profile.copy()
    profile["height"] = 512
    profile["width"] = 512
    with rasterio.Env(**rasterio_env_options):
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as src:
                pass
            memfile.seek(0)
            with tifffile.TiffFile(memfile) as tif:
                empty_JPEGTables = tif.pages[0].tags["JPEGTables"].value
    if empty_JPEGTables != JPEGTables:
        raise ValueError("different JPEGTables")


def _overview_maker(
    arr: np.ndarray,
    mask: Union[np.ndarray, None],
    overlap_map,
    profile: Dict,
    rasterio_env_options: Optional[Dict],
    return_file=True,
    return_overviews=True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    making reduced resolution overview.

    using GDAL
    there is probably already some nice dask tools for this but matching GDAL exactly is
    worthwhile here in the first instance.
    """

    profile = profile.copy()  # careful not to update in place
    profile["height"] = arr.shape[1]
    profile["width"] = arr.shape[2]
    # not using the transform in profile as its going to be wrong - make it obviously wrong avoids rasterio warnings
    profile["transform"] = Affine.scale(2)

    if not return_file:
        profile["compress"] = "NONE"
    # the overview is only used for resampling its not stored
    profile["overview_compress"] = "NONE"
    profile["overview_count"] = 1
    overview_arr = None
    mask_overview_arr = None
    file_bytes = None

    if profile["height"] == 1 and profile["width"] == 1:
        # stops gdal throwing an error
        # in the case of a 1x1 pixel input return it as the overview
        profile["overview_count"] = 0
        overview_arr = arr
        mask_overview_arr = mask

    # window for removing overlaps in overview
    overlap_map = [None if x is None else x // 2 for x in overlap_map]
    window = rasterio.windows.Window.from_slices(
        rows=overlap_map[:2],
        cols=overlap_map[-2:],
        height=max(1, profile["height"] // 2),
        width=max(1, profile["width"] // 2),
    )

    with rasterio.Env(**rasterio_env_options), rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as src:
            src.write(arr)
            del arr  # reducing mem footprint
            if mask is not None:
                src.write_mask(mask)

        if profile["overview_count"] and return_overviews:
            with memfile.open(overview_level=0) as src:
                # get the required overview back as an array
                overview_arr = src.read(window=window)
                if mask is not None:
                    # mask is the same for all bands
                    mask_overview_arr = src.read_masks(1, window=window)
            to_return = [overview_arr, mask_overview_arr]

        if return_file:
            _ = memfile.seek(0)
            file_bytes = memfile.read()

    return overview_arr, mask_overview_arr, file_bytes


def _partial_COG_maker(
    arr: np.ndarray, mask: Union[np.ndarray, None], overlap_map, profile: Dict, rasterio_env_options: Optional[Dict]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bytes, Tuple[np.ndarray, np.ndarray, int]]:
    """
    Writes 'arr' to an in-memory COG file and then pulls the COG file apart,
    returning it in parts that are useful for reconstituting it differently.

    This function leverages rasterio/GDAL for writing the file to avoid doing
    that work from scratch.

    Args:
        arr (np.ndarray): The array to be written to the COG file.
        mask (Optional[np.ndarray]): The mask array. Default is None.
        profile (Dict): The profile of the COG.
        rasterio_env_options (Optional[Dict]): Rasterio environment options. Default is None.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], bytes, Tuple[np.ndarray, np.ndarray, int]]:
        - overview_arr (Optional[np.ndarray]): The overview array.
        - mask_overview_arr (Optional[np.ndarray]): The overview mask array.
        - part_bytes (bytes): The bytes containing the COG data.
        - part_info (Tuple[np.ndarray, np.ndarray, int]): Information about COG parts.
            - databyteoffsets (np.ndarray): Offset of each data block.
            - databytecounts (np.ndarray): Count of bytes in each data block.
            - len(part_bytes) (int): Length of part_bytes.
    """

    overview_arr, mask_overview_arr, file_bytes = _overview_maker(arr, mask, overlap_map, profile, rasterio_env_options)
    if not (overlap_map == None).all():
        # if resampling requires overlaps then the creation of the overview has to happed seperate to the creation of the main image data.
        overview_arr, mask_overview_arr, _ = _overview_maker(
            arr, mask, overlap_map, profile, rasterio_env_options, return_file=False
        )
        # strip the overlap from arr and mask
        arr = arr[:, slice(*overlap_map[:2]), slice(*overlap_map[-2:])]
        if mask is not None:
            mask = mask[slice(*overlap_map[:2]), slice(*overlap_map[-2:])]
        _, _, file_bytes = _overview_maker(
            arr, mask, overlap_map, profile, rasterio_env_options, return_overviews=False
        )

    with io.BytesIO(file_bytes) as memfile, tifffile.TiffFile(memfile) as tif:
        page = tif.pages[0]
        if profile["compress"] == "jpeg":
            _test_jpegtables(page.tags["JPEGTables"].value, profile, rasterio_env_options)

        # always make mask data ignore it later if not needed. dim order to interleave data and mask.
        tile_dims_count = (
            math.ceil(page.imagelength / page.tilelength),
            math.ceil(page.imagewidth / page.tilewidth),
            2,
        )
        databytecounts = np.zeros(tile_dims_count, dtype=np.int32)
        databytecounts[:, :, 0] = np.reshape(page.databytecounts, tile_dims_count[:2])
        databyteoffsets = np.zeros(tile_dims_count, dtype=np.int64)
        databyteoffsets[:, :, 0] = np.reshape(page.dataoffsets, tile_dims_count[:2])
        if mask is not None:
            mask_page = tif.pages[1]
            databytecounts[:, :, 1] = np.reshape(mask_page.databytecounts, tile_dims_count[:2])
            databyteoffsets[:, :, 1] = np.reshape(mask_page.dataoffsets, tile_dims_count[:2])

        new_databyteoffsets = []
        current_offset = 0
        part_bytes = bytearray()
        # Collect up bytes and optional gdal leading and trailing ghosts
        for offset, bytecount in zip(databyteoffsets.ravel(), databytecounts.ravel()):
            if bytecount:
                if profile["cog_ghost_data"]:
                    part_bytes.extend(bytecount.tobytes())
                    current_offset += 4

                new_databyteoffsets.append(current_offset)
                _ = memfile.seek(offset)
                part_bytes.extend(memfile.read(bytecount))
                current_offset += bytecount

                if profile["cog_ghost_data"]:
                    part_bytes.extend(part_bytes[-4:])
                    current_offset += 4
            else:
                new_databyteoffsets.append(0)  # sparse values

    # moveaxis to un interleave this
    databyteoffsets = np.moveaxis(np.reshape(new_databyteoffsets, tile_dims_count), -1, 0)
    databytecounts = np.moveaxis(databytecounts, -1, 0)
    part_info = (databyteoffsets, databytecounts, len(part_bytes))
    return overview_arr, mask_overview_arr, bytes(part_bytes), part_info


def _adjust_compression(profile: Dict) -> None:
    """
    Adjusts the compression settings in a profile for GDAL.

    This function modifies the input profile dictionary to ensure that compression settings
    for overview levels are correctly adjusted. When writing blocks that are actually overviews,
    GDAL will still think the first level is not an overview, so this function makes sure that
    the profile accounts for this.

    Args:
        profile (dict): A dictionary containing the profile settings.

    Returns:
        None: This function does not return a value, it operates in-place, modifying the 'profile' argument.
    """
    if "overview_compress" in profile:
        if profile.get("overview_compress", None) != profile.get("compress", None):
            for k in ["quality", "predictor"]:
                if k in profile:
                    del profile[k]
    for k1, k2 in [
        ("overview_compress", "compress"),
        ("overview_quality", "quality"),
        ("overview_predictor", "predictor"),
    ]:
        if k1 in profile:
            profile[k2] = profile[k1]


def _chunk_adjuster(arr, min_chunk_dim=12):
    """
    analyse chunks and combine edge chunks in some limited cases

    this needs to be done so that overlaps can done for resampling

    the default of 12 is based on analysis of the resampling in gdal and that is the largest radius of edge effects
    """
    chunk_heights, chunk_widths = arr.chunks[-2:]
    # merge the second to last chunks if too small
    if len(chunk_heights) > 1 and chunk_heights[-1] <= min_chunk_dim:
        chunk_heights = (*chunk_heights[:-2], sum(chunk_heights[-2:]))
    if len(chunk_widths) > 1 and chunk_widths[-1] <= min_chunk_dim:
        chunk_widths = (*chunk_widths[:-2], sum(chunk_widths[-2:]))
    arr = arr.rechunk(chunks=(*arr.chunks[:-2], chunk_heights, chunk_widths))
    return arr


def _unoverlap_slices(shape, overlap):
    """
    map to overlaps on dask chunks when using dask.array.overlap.overlap
    """
    arr = np.full((*shape, 4), None)
    if overlap:
        arr[1:, :, 0] = overlap
        arr[0:-1, :, 1] = -overlap
        arr[:, 1:, 2] = overlap
        arr[:, 0:-1, 3] = -overlap
    return arr.reshape((-1, 4))


def _COG_graph_builder(
    arr: da.Array, mask: Optional[da.Array], profile: Dict[str, any], rasterio_env_options: Dict[str, any]
) -> dask.delayed:
    """
    Makes a dask delayed graph that, when computed, writes a COG file to S3.

    This function constructs a Dask delayed computation graph for creating a COG file from a Dask array.
    It splits the array into parts, processes them, and returns a delayed computation graph to build the COG.

    Args:
        arr (dask.array.Array): The Dask array to create a COG from.
        mask (dask.array.Array or None): An optional mask array for the COG.
        profile (dict): A dictionary containing the COG profile settings.
        rasterio_env_options (dict): A dictionary containing Rasterio environment options.

    Returns:
        dask.delayed: A Dask delayed object representing the computation graph to build the COG.
    """
    tif_part_maker_func = dask.delayed(_partial_COG_maker, nout=4)

    # ensure the overview count is set
    profile["overview_count"] = _get_maximum_overview_level(
        profile["width"],
        profile["height"],
        minsize=profile["blocksize"],
        overview_count=profile.get("overview_count", None),
    )
    profile = profile.copy()
    overview_profile = profile.copy()
    _adjust_compression(overview_profile)

    del_profile = dask.delayed(profile, traverse=False)
    del_overview_profile = dask.delayed(overview_profile, traverse=False)
    del_rasterio_env_options = dask.delayed(rasterio_env_options, traverse=False)
    current_level_profile = del_profile
    parts_info = {}
    parts_bytes = {}

    for level in range(profile["overview_count"] + 1):
        arr = _chunk_adjuster(arr)
        parts_info[level] = []
        parts_bytes[level] = []
        chunk_depth, chunk_heights, chunk_widths = arr.chunks
        # matching gdal behaviour
        if len(chunk_heights) > 1 and chunk_heights[-1] == 1:
            chunk_heights = chunk_heights[:-1]
        if len(chunk_widths) > 1 and chunk_widths[-1] == 1:
            chunk_widths = chunk_widths[:-1]

        overlap_count = resample_overlaps[profile["overview_resampling"]]
        if overlap_count:
            arr = dask.array.overlap.overlap(arr, (0, overlap_count, overlap_count), "none", allow_rechunk=False)

        # this is the slowest line by far. if not optimize_graph its faster but slower overall.
        # if optimize_graph=True then the graph ends up reading in the source data many times
        # if optimize_graph=False then the data is read more optimally but the building of the graph gets slower.
        data_del = arr.to_delayed(optimize_graph=False)
        res_arr = np.ndarray((len(chunk_heights), len(chunk_widths)), dtype=object)

        mask_del = []
        if mask is not None:
            mask = _chunk_adjuster(mask)
            if overlap_count:
                mask = dask.array.overlap.overlap(mask, (overlap_count, overlap_count), "none", allow_rechunk=False)
            mask_del = mask.to_delayed(optimize_graph=False).ravel()
            mask_res_arr = np.ndarray((len(chunk_heights), len(chunk_widths)), dtype=object)
        unoverlap = _unoverlap_slices(data_del.shape[-2:], overlap_count)
        for blk in zip_longest(data_del.ravel(), mask_del, unoverlap):
            ind = blk[0].key[1:][-2:]
            overview_arr, mask_arr, part_bytes, part_info = tif_part_maker_func(
                *blk, profile=current_level_profile, rasterio_env_options=del_rasterio_env_options
            )
            parts_info[level].append((ind, part_info))
            parts_bytes[level].append(part_bytes)

            # checks if the index falls within the array
            # throwing away overviews that are an artifact of the way gdal produces overviews when there is a dimension of 1
            if len([1 for s, i in zip(res_arr.shape[-2:], ind) if i < s]) == 2:
                blk_final_shape = (
                    chunk_depth[0],
                    max(1, chunk_heights[ind[0]] // 2),
                    max(1, chunk_widths[ind[1]] // 2),
                )
                res_arr[ind] = dask.array.from_delayed(overview_arr, shape=blk_final_shape, dtype=arr.dtype)
                if mask is not None:
                    mask_res_arr[ind] = dask.array.from_delayed(mask_arr, shape=blk_final_shape[-2:], dtype=mask.dtype)

        arr = dask.array.block(res_arr.tolist())
        # copy the chunking from the initial data - assuming the first chunk is indicative of appropriate chunking to use for overviews
        arr = arr.rechunk(chunks=(chunk_depth[0], chunk_heights[0], chunk_widths[0]))
        if mask is not None:
            mask = dask.array.block(mask_res_arr.tolist())
            mask = mask.rechunk(chunks=(chunk_heights[0], chunk_widths[0]))

        current_level_profile = del_overview_profile

    header_bytes_final = dask.delayed(prep_tiff_header)(
        parts_info, del_profile, del_rasterio_env_options, mask=mask is not None
    )
    # sum as a shorthand for flattening a list in the right order
    delayed_parts = sum([v for k, v in sorted(parts_bytes.items(), reverse=True)], [header_bytes_final])
    return delayed_parts


def _ifd_updater(
    memfile: io.BytesIO,
    block_data: Tuple[List[np.ndarray], List[np.ndarray]],
    mask_data: Tuple[List[np.ndarray], List[np.ndarray]],
) -> None:
    """
    Updates IFD (Image File Directory) entries in a TIFF file stored in a BytesIO object.

    This function updates the 'TileOffsets' and 'TileByteCounts' entries in the TIFF file's IFD
    based on the provided block data for main and mask pages.

    Args:
        memfile (BytesIO): A BytesIO object containing the TIFF file data.
        block_data (List[List[List[int]]]): A list of lists containing block offsets and byte counts for main pages.
        mask_data (List[List[List[int]]]): A list of lists containing block offsets and byte counts for mask pages.
    """
    assert not memfile.closed, "memory file was closed before writing"

    def _update_page(page_id, block_offsets, block_counts):
        tif.pages[page_id].tags["TileOffsets"].overwrite(block_offsets)
        tif.pages[page_id].tags["TileByteCounts"].overwrite(block_counts)

    # first gather the indexes for main and mask pages
    memfile.seek(0)
    with tifffile.TiffFile(memfile) as tif:
        main = []
        mask = []
        for p in tif.pages:
            if "MASK" in str(p.tags.get("NewSubfileType", "")):
                mask.append(p.index)
            else:
                main.append(p.index)

    # write the data to the ifd ensuring to interleave the main and mask offsets/bytecounts
    # ignore mask data if there are no mask ifds
    for (
        main_id,
        mask_id,
        main_page_data_off,
        main_page_data_cnts,
        mask_page_data_off,
        mask_page_data_cnts,
    ) in zip_longest(main, mask, *block_data, *mask_data):
        _update_page(main_id, main_page_data_off, main_page_data_cnts)
        if mask_id is not None:
            _update_page(mask_id, mask_page_data_off, mask_page_data_cnts)


def _ifd_offset_adjustments(
    header_length: int, parts_info: dict
) -> Tuple[Tuple[List[np.ndarray], List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Merges IFD (Image File Directory) data into the right order needed for the concatenated COG TIFF header.
    Also applies part offsets to the image data offsets.

    Args:
        header_length (int): The length of the TIFF header in bytes.
        parts_info (dict): A dictionary containing information about image parts, including offsets and byte counts.

    Returns:
        Tuple[Tuple[List[np.ndarray], List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray]]]:
            A tuple containing two sub-tuples, each consisting of a list of arrays.
            - The first sub-tuple contains block offsets and block byte counts for main image data.
            - The second sub-tuple contains block offsets and block byte counts for mask image data (if present).

    Notes:
        - This function assumes that the provided `parts_info` dictionary contains information about image parts
          for different overview levels. The data is processed and merged in the appropriate order.
        - The resulting offsets and byte counts are returned in reverse order, as header pages are usually stored
          in reverse order in a concatenated COG TIFF.
    """
    block_offsets = []
    block_counts = []
    mask_offsets = []
    mask_counts = []
    # apply cumulative offsets to tile offset data
    offset = header_length
    # the order produced is right except the levels need reversing - should match the order data is written to file
    for level in sorted(parts_info, reverse=True):
        blk_ind_last_blk = [xy + 1 for xy in parts_info[level][-1][0]]
        TileOffsets_arr = np.ndarray(blk_ind_last_blk, dtype=object)
        TileByteCounts_arr = np.ndarray(blk_ind_last_blk, dtype=object)
        for ind, (TileOffsets, TileByteCounts, bytes_total_len) in parts_info[level]:
            # note this modifies in place!
            TileOffsets[TileByteCounts > 0] += offset
            TileOffsets[TileByteCounts == 0] = 0  # sparse tif data

            # set the offset for the next level
            offset += bytes_total_len

            TileOffsets_arr[ind] = TileOffsets
            TileByteCounts_arr[ind] = TileByteCounts

        # make block arrays into arrays and then flatten them
        TileOffsets_merged = np.block(TileOffsets_arr.tolist())
        TileByteCounts_merged = np.block(TileByteCounts_arr.tolist())

        block_offsets.append(TileOffsets_merged[0].ravel())
        block_counts.append(TileByteCounts_merged[0].ravel())

        mask_offsets.append(TileOffsets_merged[1].ravel())
        mask_counts.append(TileByteCounts_merged[1].ravel())

    # reverse as the header pages are in reverse order
    return (block_offsets[::-1], block_counts[::-1]), (mask_offsets[::-1], mask_counts[::-1])


def prep_tiff_header(parts_info: dict, profile: dict, rasterio_env_options: dict, mask: bool) -> bytes:
    """
    Update the TIFF header.

    Args:
        parts_info (dict): A dictionary containing information about image parts, including offsets and byte counts.
        profile (dict): A dictionary representing the profile of the COG image.
        rasterio_env_options (dict): Options for configuring Rasterio's environment.
        mask (bool): A boolean indicating whether the COG image has a mask.

    Returns:
        bytes: The updated COG TIFF header data.
    """
    header_bytes = _empty_COG(profile, rasterio_env_options=rasterio_env_options, mask=mask)
    # adjust the block data
    # print (parts_info)
    block_data, mask_data = _ifd_offset_adjustments(len(header_bytes), parts_info)

    # write the block data into the header
    with io.BytesIO(header_bytes) as memfile:
        _ifd_updater(memfile, block_data, mask_data)
        memfile.seek(0)
        header_data = memfile.getvalue()
    return header_data

def write_ccog(
    arr: Union[da.Array, xarray.DataArray, np.ndarray],
    mask: Optional[Union[da.Array, xarray.DataArray, np.ndarray]] = None,
    *,
    store: Optional[str] = None,
    COG_creation_options: Optional[Dict[str, any]] = None,
    rasterio_env_options: Dict[str, any] = None,
    storage_options: Optional[Dict[str, any]] = None,
) -> Delayed:
    """Creates a dask graph that when computed produces a concatenated COG either as bytes or written to S3.

    If arr chunks are full width OR there height equals profile['blocksize'] then the GDAL COG ghost optimisations are retained



    Args:
        arr (Union[da.Array, xarray.DataArray, np.ndarray]): The data array to write as a COG. It can be a dask array, xarray DataArray or numpy ndarray.
        mask (Optional[Union[da.Array, xarray.DataArray, np.ndarray]]): A mask array for the data. It can be a dask array, xarray DataArray or numpy ndarray. Default is None.
        store (Optional[str]): An S3 file path or fsspec mapping. If not included, the function will produce a dask graph that when computed produces a list of bytes that make up the COG file. Default is None.
        COG_creation_options (Dict[str, any]): Options for configuring the COG creation.
            This is required. OVERVIEW_RESAMPLING is needed as a minimum.
            This is very similar to what is entered into rasterio.open.
            There is no need to specify a path, mode,driver,width,height,count, dtype or sharing and if you do these will be ignored.
            CRS, transform and nodata will be taken from the rioxarray rio accessor for a xarray DataArray, or can be input by the user. The users input takes precedence.
            GCPS and RPCS are handled by rasterio if included.
            
            Additional creation options as specified by GDAL can also be included.
            https://gdal.org/drivers/raster/cog.html
            A few differences are noted here:
            OVERVIEW_RESAMPLING is required by CCOG.
                For resampling values of NEAREST/AVERAGE/MODE/RMS will result in a simpler dask graph and work faster and therfore should be prefered.
                Values of BILINEAR/CUBIC/CUBICSPLINE/LANCZOS/GAUSS use a more complex graph to avoid edge effects.
            BIGTIFF is always yes
            GEOTIFF_VERSION defaults to 1.1
            SPARSE_OK defaults to True
            
            The following are ignored or raise a ValueError.
            OVERVIEWS,WARP_RESAMPLING,TILING_SCHEME,ZOOM_LEVEL,ZOOM_LEVEL_STRATEGY,TARGET_SRS,RES,EXTENT,ALIGNED_LEVELS and ADD_ALPHA

            CCOG also extends this with additional creation options.
            These represent other information that can be added to a COG file that are
            normally set through the rasterio API at the time of writing the file.
            Refer to the rasterio API for details on valid values for these options
            
            update_tags - a dict with integer keys where key 0 is dataset tags and 
                tag 1 is tags for band 1 etc.
                values are a dict of tags eg dict(tagname1x=0, tagname2='tag content') see
                https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetWriter.update_tags
            descriptions - a list of strings: with one value per band in band order
            offsets - a list of numbers: with one value per band in band order
            scales - a list of numbers: with one value per band in band order
            units - a list of strings: with one value per band in band order
            colorinterp - a list with one value per band in band order see 
                https://rasterio.readthedocs.io/en/latest/topics/color.html#color-interpretation
            write_colormap - a colour dict as described here. Note only a single colormap applied to the first band is supported
                https://rasterio.readthedocs.io/en/latest/topics/color.html#writing-colormaps

        rasterio_env_options (Optional[Dict[str, any]]): Options for configuring Rasterio's environment. Default is None.
        storage_options (Optional[Dict[str, any]]): Any additional parameters for the storage backend. Default is None.

    Returns:
        Delayed: A dask.delayed object representing the process of creating the COG.
    """
    # todo:
    # In addition to the gdal option an optional colormap is accepted. This only works with a single int8 band.
    # eg colormap = {0:(4,0,0,4),255:(0,0,4,4)}
    # for more info see https://rasterio.readthedocs.io/en/stable/topics/color.html

    # TODO: shortcut route if a small array can just be handled as a single part
    # TODO: work with GDAL config settings (environment)

    COG_creation_options = {} if COG_creation_options is None else COG_creation_options
    rasterio_env_options = {} if rasterio_env_options is None else rasterio_env_options
    storage_options = {} if storage_options is None else storage_options

    # normalise keys and string values to lower case for ease of use
    user_creation_options = {
        key.lower(): val.lower() if isinstance(val, str) else val for key, val in COG_creation_options.items()
    }
    rasterio_env_options = {
        key.lower(): val.lower() if isinstance(val, str) else val for key, val in rasterio_env_options.items()
    }

    rasterio_env_options = rasterio_env_options.copy()
    rasterio_env_options.update(required_rasterio_env_options)

    # throw error as these options have not been tested and likely wont work as expected
    incompatable_options = [
        "warp_resampling",  # no reprojection allowed
        "target_srs",  # no reprojection allowed
        "tiling_scheme",  # no reprojection allowed
        "interleave",  # not currently available in the COG driver but if set to BAND the code will need some further work
        # not currently available in the COG driver but if becomes available will need testing
        "jpegtablesmode",
        "blockxsize",  # use blocksize
        "blockysize",  # use blocksize
    ]
    incompatable_options.extend(required_creation_options.keys())
    exclude_opts = [opt for opt in incompatable_options if opt in user_creation_options]
    if exclude_opts:
        raise ValueError(f"ccog cant work with COG_creation_options of {exclude_opts}")

    if "overview_resampling" not in user_creation_options and "resampling" in user_creation_options:
        user_creation_options["overview_resampling"] = user_creation_options["resampling"]
        del user_creation_options["resampling"]

    if (
        "overview_resampling" not in user_creation_options
        or user_creation_options["overview_resampling"] not in resample_overlaps
    ):
        raise ValueError(
            f"COG_creation_options needs to include a valid GDAL COG driver 'overview_resampling' selection"
        )

    if not isinstance(arr, (xarray.core.dataarray.DataArray, dask.array.core.Array, np.ndarray)):
        raise TypeError(" arr must be an instance of xarray DataArray or dask Array")
        
    # todo: raise error if required_creation_options are going to change user_creation_options
    # build the profile and env_options by layering sources sequentially making sure most important end up with precendence.
    profile = default_creation_options.copy()

    # get useful stuff from xarray via rioxarray
    if isinstance(arr, xarray.core.dataarray.DataArray):
        profile["transform"] = arr.rio.transform()
        profile["crs"] = arr.rio.crs
        profile["nodata"] = arr.rio.nodata
        arr = arr.data
        
    if isinstance(arr, np.ndarray):
        arr = dask.array.from_array(arr)

    profile.update(user_creation_options)
    profile.update(required_creation_options)

    if int(profile["blocksize"]) % 16:
        # the tiff rule is 16 but that is a very small block, resulting in a large header. consider a 256 minimum?
        raise ValueError(f"blocksize must be multiples of 16")
    profile["blockxsize"] = profile["blockysize"] = profile["blocksize"]

    # handle single band data the same as multiband data
    if len(arr.shape) == 2:
        arr = arr.reshape([1] + [*arr.shape])
    # check chunking.
    if any([dim % profile["blocksize"] for dim in arr.chunks[-2][:-1]]) or any(
        [dim % profile["blocksize"] for dim in arr.chunks[-1][:-1]]
    ):
        raise ValueError("chunking needs to be multiples of the blocksize (except the last in any spatial dimension)")
    if len(arr.chunks) == 3 and len(arr.chunks[0]) > 1:
        raise ValueError("non spatial dimension chunking needs to be a single chunk")

    # keep the ghost data if the chunking allows it
    if all([dim == profile["blocksize"] for dim in arr.chunks[-2][:-1]]) or len(arr.chunks[-1]) == 1:
        profile["cog_ghost_data"] = True

    profile["count"] = arr.shape[0]
    profile["height"] = arr.shape[1]
    profile["width"] = arr.shape[2]
    profile["dtype"] = arr.dtype.name

    # mask - check
    if mask is not None:
        if not isinstance(mask, (xarray.core.dataarray.DataArray, dask.array.core.Array, np.ndarray)):
            raise TypeError(" mask must be an instance of xarray DataArray or dask Array")
        if isinstance(mask, xarray.core.dataarray.DataArray):
            mask = mask.data
        if isinstance(mask, np.ndarray):
            mask = dask.array.from_array(mask)
        if arr.chunks[-2:] != mask.chunks:
            raise ValueError("mask spatial chunks needs to match those of arr")
        # todo: also check CRS and transform match

    # building the delayed graph
    delayed_graph = _COG_graph_builder(arr, mask, profile=profile, rasterio_env_options=rasterio_env_options)
    if store:
        delayed_graph = aws_tools.mpu_upload_dask_partitioned(delayed_graph, store, storage_options=storage_options)
    return delayed_graph
