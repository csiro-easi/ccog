COG files are cloud optimised for reading but writing large COG files isnt so easy.

# CCOG

Makes a COG file on s3 storage from an xarray. Does compression, pyramid/overview generation and writing of the file all in a distributed way using dask.

Im not aware of another tool that can write a COG file without all the writes going through a single process or using locks. Im not aware of another tool that can spread this work to a cluster.

Many tools struggle to write a large COG. CCOG should (subject to further testing) be able to quickly and efficently write a cog up to the s3 file size limit of 5 TiB while using many dask workers.

CCOG works by producing many smaller COG files and carefully combining the headers and compressed pixel data.

The output file should be indistinguishable from a COG fully written by the gdal cog driver except for:
- The storage order of the compressed image data is different
- In some cases there may be a small increase in file size as some padding may be used due to an s3 limitation.
- the gdal COG ghost header is modified to indicate the above changes.

## Status

This package is a work in progress. It is tested to install and run on linux with single band approx 200GiB datasets but further work is required to improve its functionality.

A few people are activly using CCOG, however further testing is required.

At this stage the output isnt as "cloud optimised" as other COGs but there is a list of tasks to improve this. There is also potential to go beyond the current COG spec and make further "cloud optimisations" particularly suited to reading chunks that are useful sizes for analysis work.

## Installation

CCOG can be installed directly from this repository.

## Usage

```python
import ccog
import xarray

#start a dask cluster with CCOG installed. This is not shown here for brevity and users are likely already familiar with how to do this on their system.

#my bucket needs some additional settings
storage_options = dict(anon=False, s3_additional_kwargs={"ACL": "bucket-owner-full-control"})

#make an xarray array some how
x_arr = xarray.open_zarr(.....)['......']
#do some processing on it.... what ever you want
#make sure you think about your chunking either at read time or later on
#ccog is opinionated about the chunks it recieves (see the docs of write_ccog) 
x_arr = x_arr.rechunk(chunks= 2**14)

#make a writer - see the docs for how to specify gdal cog options
delayed_writer = ccog.write_ccog(x_arr,store,storage_options=storage_options)
#compute it now or later on
delayed_writer.compute()
```
    
