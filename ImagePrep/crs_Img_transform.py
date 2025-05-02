# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:50:40 2025

@author: mccullru
"""

from pyproj import CRS
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling



# Define ITRF2014 UTM 33N manually
itrf14_utm33n = CRS.from_proj4("+proj=utm +zone=33 +datum=ITRF2014 +units=m +no_defs")



# Input raster
input_raster_path = r"E:\Thesis Stuff\AcoliteWithPython\Corrected_Imagery\Punta_test_SD_pyth\PSScene-20230321_093910_07_2481"
# Output raster
output_raster_path = r"E:\Thesis Stuff\AcoliteWithPython\Corrected_Imagery\reproj_test\reprojected_to_ITRF2014_UTM33N.tif"

# Open input raster
with rasterio.open(input_raster_path) as src:
    # Calculate the transform and metadata for new CRS
    transform, width, height = calculate_default_transform(
        src.crs, itrf14_utm33n, src.width, src.height, *src.bounds
    )
    
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': itrf14_utm33n,
        'transform': transform,
        'width': width,
        'height': height
    })

    # Write reprojected raster
    with rasterio.open(output_raster_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=itrf14_utm33n,
                resampling=Resampling.nearest
            )