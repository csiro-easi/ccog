# CCOG - Create Cloud Optimised Geotiffs

CCOG is a Python package for the creation of COG (Cloud Optimised Geotiff) files. 

## How CCOG works

CCOG uses dask to process large arrays (dask or xarray arrays) in chunks and this processing can be distributed over many workers in parallel.

CCOG uses rasterio, gdal and tifffile internally to create and manipulate parts of the file. Users familiar with rasterio should find it easy use.

The output file is either written directly from the dask workers to an S3 bucket or bytes can be returned for the user to store locally.

## Rationale

CCOG was created because while COG files are cloud optimised for reading, the experience of writing large COG files isn’t always so easy.

CCOGs aim is to fit easily into dask based workflows and enable fast and easy creation of COG files limited only by the S3 file size limit of 5 TiB.

## Example

```python
import ccog
import xarray
import dask

#start a dask cluster with CCOG installed. This is not shown here for brevity and users are likely already familiar with how to do this on their system.

#uses fsspec but is limited to paths on S3
store = 's3://bucket/path/file.tif'
#my bucket needs some additional settings
storage_options = dict(anon=False, s3_additional_kwargs={"ACL": "bucket-owner-full-control"})

#make an xarray array some how
x_arr = xarray.open_zarr(.....)['......']
#do some processing on it.... whatever you want
#make sure you think about chunking either at read time or later on
#ccog is opinionated about the chunks it recieves (see the docs of write_ccog) 
x_arr = x_arr.rechunk(chunks= 512*8)

#choose settings for making the COG
profile = dict(
    compress= 'deflate',
    overview_resampling = 'average',
)

#make a writer - see the docs for more options
delayed_writer = ccog.write_ccog(x_arr,store,COG_creation_options=profile,storage_options=storage_options)
#compute it now or later on
result = dask.compute(delayed_writer)

```

## Install

CCOG can be installed from this repository
```
pip install https://github.com/csiro-easi/ccog/archive/refs/heads/master.zip

```
To install on distributed dask workers the recommendation is that CCOG is installed in the worker image however the following link is useful for installation on workers as needed.

https://distributed.dask.org/en/stable/plugins.html#distributed.diagnostics.plugin.PipInstall


## Status

This package is a work in progress. It is tested to install and run on linux. Please report any issues and PRs are welcome.

The aim is for CCOG to create COG files identical to what rasterio and GDAL produce. This is currently possible depending on the layout of the input arrays chunks. Other chunk layouts will still produce a valid COG but won’t exactly match the internal file layout that GDAL uses or include some of GDALs COG optimisations.

Some specific GDAL COG creation options will not be supported as they involve reprojection. This will be best handled as an earlier processing step on the input array.
