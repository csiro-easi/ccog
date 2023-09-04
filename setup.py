from setuptools import setup#, find_packages

setup(
    name='ccog',
    version='0.1.0',
    description='makes concatenated COG files',
    author='',
    author_email='',
    url='',
    python_requires=">=3.10.6",
    packages=['ccog'], #find_packages(),
    install_requires=[
        'fsspec',
        'numpy',
        'xarray',
        'rasterio',
        'dask',
        'tifffile',
        'affine>=2.4', #recent bug fix
        'more_itertools',
        ],
    entry_points={
        'console_scripts': []
    },
    package_data={}
)