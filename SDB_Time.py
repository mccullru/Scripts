# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:33:25 2024

@author: mccullru
"""

import os
import rasterio
import numpy as np
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

from glob import glob
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from rasterio.enums import Resampling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from rasterio.mask import mask
from difflib import get_close_matches


##############################################################################
##############################################################################

# Composite R, G, B tiffs into one RGB tiff

def extract_rrs_number(file_name):
    """Extract the number after 'Rrs_' in the file name."""
    match = re.search(r'Rrs_(\d+)', file_name)
    if match:
        return int(match.group(1))
    return None  # Return None if 'Rrs_' is not found


def is_close_to(value, target, tolerance):
    return abs(value - target) <= tolerance


def combine_bands_to_rgb(input_folder, output_folder):
    """
    Combines separate red, green, and blue TIFF files into a single GeoTIFF.
    Only processes the .tif files.

    Args:
        input_folder (str): Path to the folder containing the band TIFFs.
        output_folder (str): Path to save the combined RGB GeoTIFFs.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Recursively find all .tif files
    files = list(Path(input_folder).rglob("*.tif"))
    
    # Filter bands based on proximity to target wavelengths
    red_bands = sorted([f for f in files if (num := extract_rrs_number(f.name)) is not None and is_close_to(num, 666, 10)])
    green_bands = sorted([f for f in files if (num := extract_rrs_number(f.name)) is not None and is_close_to(num, 560, 10)])
    blue_bands = sorted([f for f in files if (num := extract_rrs_number(f.name)) is not None and is_close_to(num, 492, 10)])

    print(f"Red bands found: {len(red_bands)}")
    print(f"Green bands found: {len(green_bands)}")
    print(f"Blue bands found: {len(blue_bands)}")

    # Check if all band files match
    if not (len(red_bands) == len(green_bands) == len(blue_bands)):
        print("Mismatch in the number of red, green, and blue band files.")
        return

    # Process each set of bands
    for r, g, b in zip(red_bands, green_bands, blue_bands):
        try:
            # Read band data
            with rasterio.open(r) as red:
                profile = red.profile
                red_data = red.read(1)
                red_nodata = red.nodata

            with rasterio.open(g) as green:
                green_data = green.read(1)
                green_nodata = green.nodata

            with rasterio.open(b) as blue:
                blue_data = blue.read(1)
                blue_nodata = blue.nodata

            # Ensure NoData is handled: keep NoData as NaN in the output
            if red_nodata is not None:
                red_data = np.ma.masked_equal(red_data, red_nodata)

            if green_nodata is not None:
                green_data = np.ma.masked_equal(green_data, green_nodata)

            if blue_nodata is not None:
                blue_data = np.ma.masked_equal(blue_data, blue_nodata)

            # Update profile for RGB output (using float32 to allow NaN values)
            profile.update(count=3, dtype=rasterio.float32, nodata=np.nan)  # 3 bands (R, G, B)

            # Output file name (remove "Rrs_<number>" from filename)
            input_folder_name = Path(input_folder).name
            output_file_name = re.sub(r"Rrs_\d+", "", r.name).replace("__", "_")  # Clean up any double underscores
            output_file = os.path.join(output_folder, f"{input_folder_name}_{Path(output_file_name).stem}_RGB.tif")

            # Write combined RGB TIFF
            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(red_data, 1)   # Red band
                dst.write(green_data, 2) # Green band
                dst.write(blue_data, 3)  # Blue band

            print(f"Combined RGB saved to: {output_file}")

        except Exception as e:
            print(f"Error processing {r}, {g}, {b}: {e}")


################# CHECK DIRECTORIES/INPUTS #####################

input_folder = r"E:\Thesis Stuff\AcoliteWithPython\Corrected_Imagery\Marathon"
output_folder = r"E:\Thesis Stuff\RGBCompositOutput"
combine_bands_to_rgb(input_folder, output_folder)


###############################################################################
###############################################################################

# Reproject images from WGS84/UTM to ITRF14/UTM





###############################################################################
###############################################################################

# Downsample SD (3m) to something else
# NOTE: Does not downsample to exact target resolution, just a little bit over as in mm

# import os
# import rasterio
# import numpy as np
# import re
# import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import xarray as xr

# from glob import glob
# from rasterio.warp import calculate_default_transform, reproject, Resampling
# from pathlib import Path
# from rasterio.enums import Resampling
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from rasterio.mask import mask
# from difflib import get_close_matches

# def downsample_image(input_path, target_resolution):
#     """
#     Downsamples a GeoTIFF to a specified resolution using bilinear interpolation.
    
#     Args:
#         input_path (str): Path to the input GeoTIFF.
#         downscale_factor (float): Factor by which to downscale the image (e.g., 0.5 for halving the resolution).
    
#     Returns:
#         data (numpy.ndarray): The downsampled image data.
#         transform (Affine): The transformed affine matrix for the downsampled image.
#         profile (dict): The metadata of the downsampled image.
#     """
   
    
   
#     with rasterio.open(input_path) as dataset:
        
#         # Get the original resolution (cell size) from the dataset transform
#         original_resolution_x = dataset.transform[0]  # pixel width
#         original_resolution_y = -dataset.transform[4]  # pixel height (negative because of the top-left origin)

#         # Calculate the number of pixels needed for the target resolution
#         new_width = int(dataset.width * (original_resolution_x / target_resolution))
#         new_height = int(dataset.height * (original_resolution_y / target_resolution))
        
#         # Downsample data to the target shape
#         data = dataset.read(
#             out_shape=(
#                 dataset.count,
#                 new_height,
#                 new_width
#             ),
#             resampling=Resampling.bilinear
#         )

#         # Scale image transform for downsampling
#         transform = dataset.transform * dataset.transform.scale(
#             (dataset.width / data.shape[-1]),  # scale factor in x direction
#             (dataset.height / data.shape[-2])  # scale factor in y direction
#         )

#         # Update metadata for the downsampled image
#         profile = dataset.profile
#         profile.update(
#             height=data.shape[1],
#             width=data.shape[2],
#             transform=transform
#         )

#     return data, transform, profile

# def process_images_in_folder(input_folder, target_resolution):
#     """
#     Processes all images in the input folder, downsamples them, and replaces the original images.
    
#     Args:
#         input_folder (str): Path to the folder containing the input GeoTIFFs.
#         downscale_factor (float): Factor by which to downscale the image.
#     """
#     # Loop through all files in the input folder
#     for filename in os.listdir(input_folder):
#         input_path = os.path.join(input_folder, filename)
        
#         # Only process files containing "PlanetScope" and .tif
#         if not filename.endswith(".tif") or "PlanetScope" not in filename:
#             continue

#         print(f"Processing {filename}...")

#         # Downsample the image
#         data, transform, profile = downsample_image(input_path, target_resolution)

#         # Modify the file name to add "_ds" (downsampled) before the extension
#         output_filename = os.path.splitext(filename)[0] + "_ds7.tif"
#         output_path = os.path.join(input_folder, output_filename)


#         # Overwrite the original file with the downsampled image
#         with rasterio.open(output_path, 'w', **profile) as dst:
#             dst.write(data)
#         os.remove(input_path)

#         print(f"Downsampled image saved (replaced): {filename}")


#       #################  CHECK DIRECTORIES/INPUTS #####################

# input_folder = r"E:\Thesis Stuff\RGBCompositOutput"  
# target_resolution = 7  # Downsample the image (e.g., from 3m to 10m resolution)

# process_images_in_folder(input_folder, target_resolution)



##############################################################################
##############################################################################

# Import icesat reference data (Only needs to be done once)
# NOTE: If coordinate transforms are needed this is the place to do it


# def process_lidar_files(input_folder, output_folder):
#     """
#     Processes all LiDAR CSV files in the input folder. Extracts easting, northing, and elevation data, 
#     and keeps only the rows where the 'Class' column has a value of 40.
    
#     Args:
#         input_folder (str): Path to the folder containing the LiDAR CSV files.
#         output_folder (str): Path to the folder where the filtered CSV files will be saved.
    
#     Returns:
#         List of DataFrames: List containing the filtered DataFrames for each LiDAR file.
#     """
#     lidar_data_list = []  # List to store the DataFrames of each file

#     # Check if the output folder exists; if not, create it
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Loop through all files in the input folder
#     for filename in os.listdir(input_folder):
#         input_path = os.path.join(input_folder, filename)
        
#         # Skip non-CSV files
#         if not filename.endswith(".csv"):
#             continue

#         print(f"Processing {filename}...")

#         # Read the CSV file into a pandas DataFrame
#         lidar_data = pd.read_csv(input_path)

#         # Assuming the CSV columns are labeled as 'Latitude', 'Longitude', 'Geoid_Corrected_Ortho_Height', 'Manual_Label'
#         if 'Latitude' in lidar_data.columns and 'Longitude' in lidar_data.columns and 'Geoid_Corrected_Ortho_Height' in lidar_data.columns and 'Manual_Label' in lidar_data.columns:
#             # Filter the data to keep only rows where 'Class' == 40
#             lidar_data_filtered = lidar_data[lidar_data['Manual_Label'] == 40]

#             # Keep only the desired columns: 'Latitude', 'Longitude', 'Geoid_Corrected_Ortho_Height'
#             lidar_data_filtered = lidar_data_filtered[['Latitude', 'Longitude', 'Geoid_Corrected_Ortho_Height']]
            
            
#             # Optionally, print first few rows to inspect
#             #print(f"First few rows of filtered {filename}:")
#             #print(lidar_data_filtered.head())  # Print first few rows of filtered data
            
#             # Append the filtered DataFrame to the list
#             lidar_data_list.append(lidar_data_filtered)

#             # Save the filtered data to a new CSV file in the output folder
#             output_filename = f"filtered_{filename}"
#             output_path = os.path.join(output_folder, output_filename)
#             lidar_data_filtered.to_csv(output_path, index=False)
#             print(f"Filtered data saved to: {output_path}")
#         else:
#             print(f"Skipping {filename}, required columns are missing.")

#     return lidar_data_list

# def process_lidar_data(lidar_data_list):
#     """
#     Example function to process the lidar data (e.g., apply transformations, analysis).
    
#     Args:
#         lidar_data_list (list of DataFrame): List of DataFrames from each LiDAR CSV file.
#     """
#     for lidar_data in lidar_data_list:
#         # Example processing: Print statistics for each filtered LiDAR dataset
#         print("\nProcessing Filtered Data:")
#         print(f"Longitude range: {lidar_data['Longitude'].min()} to {lidar_data['Longitude'].max()}")
#         print(f"Latitude range: {lidar_data['Latitude'].min()} to {lidar_data['Latitude'].max()}")
#         print(f"Elevation range: {lidar_data['Geoid_Corrected_Ortho_Height'].min()} to {lidar_data['Geoid_Corrected_Ortho_Height'].max()}")




      #################  CHECK DIRECTORIES/INPUTS #####################

# input_folder = r"B:\Thesis Project\Reference Data\Classified_ICESat_Tracks\UpdatedCSVs"
# output_folder = r"B:\Thesis Project\Reference Data\Classified_ICESat_Tracks\BathyOnly_UpdatedCSVs"  

# lidar_data_list = process_lidar_files(input_folder, output_folder)

# # Optionally, process the filtered data further
# process_lidar_data(lidar_data_list)



##############################################################################
##############################################################################

# Create pSDB red and green

import os
import rasterio
import numpy as np
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

from glob import glob
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from rasterio.enums import Resampling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from rasterio.mask import mask
from difflib import get_close_matches


def process_rgb_geotiffs(input_folder, output_folder):
    """
    Processes a folder of RGB GeoTIFF files to compute the pSDBgreen index
    and saves the results as new GeoTIFF files.

    Args:
        input_folder (str): Path to the folder containing RGB GeoTIFF files.
        output_folder (str): Path to save the processed GeoTIFF files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all GeoTIFF files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.tif'):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing file: {file_name}")

            try:
                # Open the GeoTIFF file
                with rasterio.open(file_path) as src:
                    # Read in desired bands 
                    
                    red_band = src.read(1)   # Red band
                    green_band = src.read(2)  # Green band
                    blue_band = src.read(3)   # Blue band
                    profile = src.profile

                # Scale the bands
                scaled_red = red_band * 10000
                scaled_green = green_band * 10000
                scaled_blue = blue_band * 10000

                # Avoid log errors: Set negative or zero values to NaN
                scaled_red[scaled_red <= 0] = np.nan
                scaled_green[scaled_green <= 0] = np.nan
                scaled_blue[scaled_blue <= 0] = np.nan

                # Compute the log-transformed values
                ln_red = np.log(scaled_red)
                ln_green = np.log(scaled_green)
                ln_blue = np.log(scaled_blue)

                # Compute the pSDBgreen index
                pSDBred = ln_blue / ln_red
                pSDBgreen = ln_blue / ln_green

                # Update profile for the output GeoTIFF
                profile.update(
                    dtype=rasterio.float32,
                    count=1,
                    nodata=np.nan
                )

                # Save the computed index to the output folder
                
                ######## Change between pSDBred and pSDBgreen as needed #########
                
                # pSDBred
                output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_pSDBred.tif")    
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(pSDBred.astype(rasterio.float32), 1)
                print(f"Saved pSDBred to: {output_file}")

                # pSDBgreen
                output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_pSDBgreen.tif")    
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(pSDBgreen.astype(rasterio.float32), 1)
                print(f"Saved pSDBgreen to: {output_file}")


            except Exception as e:
                print(f"Error processing {file_name}: {e}")


      #################  CHECK DIRECTORIES/INPUTS #####################

input_folder = r"E:\Thesis Stuff\RGBCompositOutput"
output_folder = r"E:\Thesis Stuff\pSDB"


process_rgb_geotiffs(input_folder, output_folder)


##############################################################################
##############################################################################
# Extract pSDB values at reference point locations

import os
import rasterio
import numpy as np
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

from glob import glob
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from rasterio.enums import Resampling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from rasterio.mask import mask
from difflib import get_close_matches

def extract_raster_values(csv_file, raster_folder, output_folder):
    """
    Extracts raster values at the locations provided in the CSV file and saves the results.

    Args:
        csv_file (str): Path to the CSV file containing latitude, longitude, and elevation.
        raster_folder (str): Path to the folder containing GeoTIFF raster files.
        output_folder (str): Path to the folder where the results will be saved.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    df.columns = ['Easting(m)', 'Northing(m)', 'Orthometric Height(m)']
    
    # Convert the CSV to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_xy(df['Easting(m)'], df['Northing(m)']))
    
    # Loop through all raster files in the folder
    for raster_file in os.listdir(raster_folder):
        if raster_file.endswith(".tif"):  # Assuming GeoTIFF raster files
            raster_path = os.path.join(raster_folder, raster_file)
            print(f"Processing raster: {raster_file}")

            # Open the raster file using rasterio
            with rasterio.open(raster_path) as src:
                
                # Filter reference points that are within the raster bounds
                raster_bounds = src.bounds
                gdf_in_bounds = gdf[gdf.geometry.apply(lambda point: is_point_within_bounds(point, raster_bounds))]

                # Ensure there are points within the bounds
                if gdf_in_bounds.empty:
                    print(f"No points overlap with raster: {raster_file}")
                    continue
                
                # Extract the raster values at each valid location
                gdf_in_bounds.loc[:, 'Raster_Value'] = gdf_in_bounds.geometry.apply(lambda point: get_raster_value_at_point(src, point))
            
            # Check if 'Raster_Value' exists and drop rows with NaN
            if 'Raster_Value' in gdf_in_bounds.columns:
                gdf_filtered = gdf_in_bounds.dropna(subset=['Raster_Value'])
            else:
                print(f"No valid raster values found for raster: {raster_file}")
                continue
            
            # Save the results to a new CSV file
            output_filename = os.path.splitext(raster_file)[0] + "_extracted.csv"
            output_path = os.path.join(output_folder, output_filename)
            gdf_filtered.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")

def get_raster_value_at_point(raster_src, point):
    """
    Gets the raster value at a specific point (latitude, longitude).
    
    Args:
        raster_src (rasterio.io.DatasetReader): The raster source object.
        point (shapely.geometry.point.Point): The point geometry (Longitude, Latitude).
    
    Returns:
        float: The raster value at the specified point.
    """
    try:
        # Convert point to raster coordinates
        row, col = raster_src.index(point.x, point.y)
        value = raster_src.read(1)[row, col]
    except (IndexError, ValueError):
        # Return a NoData value or NaN if the point is out of bounds
        value = float('nan')
    
    return value

def is_point_within_bounds(point, bounds):
    """
    Checks if a point is within raster bounds.
    
    Args:
        point (shapely.geometry.point.Point): The point geometry.
        bounds (tuple): The raster bounds (left, bottom, right, top).
    
    Returns:
        bool: True if the point is within bounds, False otherwise.
    """
    left, bottom, right, top = bounds
    return left <= point.x <= right and bottom <= point.y <= top



      #################  CHECK DIRECTORIES/INPUTS #####################

csv_file = r"E:\Thesis Stuff\ReferenceData\Topobathy\Marathon_2019_combined_and_downsampled_points.csv"
raster_folder = r"E:\Thesis Stuff\pSDB"
output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts"

extract_raster_values(csv_file, raster_folder, output_folder)



##############################################################################
##############################################################################

# Perform linear regressions between pSDB red and green and reference calibration data

# import os
# import rasterio
# import numpy as np
# import re
# import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import xarray as xr
# import matplotlib
# matplotlib.use('TkAgg')

# from glob import glob
# from rasterio.warp import calculate_default_transform, reproject, Resampling
# from pathlib import Path
# from rasterio.enums import Resampling
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from rasterio.mask import mask
# from difflib import get_close_matches
# from matplotlib.widgets import Slider


# def process_csv_files(input_folder, output_folder):
#     """
#     Processes CSV files in the input folder, performs linear regression, 
#     and saves the results in the output folder.
#     """
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # Define the output file path
#     #output_file = os.path.join(output_folder, "linear_regression_results.csv")

#     # Loop through all CSV files in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".csv"):
#             input_path = os.path.join(input_folder, filename)
#             print(f"Processing {filename}...")

#             # Read the CSV file
#             data = pd.read_csv(input_path, delimiter=',', decimal='.')

#             # Drop rows where 'Raster_Value' is blank
#             data = data.dropna(subset=['Raster_Value', 'OrthometricHeight(m)'])
            
#             # Initialize a results list
#             results = []

#             # plot the data
#             x = data[['Raster_Value']].values
#             y = data[['OrthometricHeight(m)']].values
            
    
#             # Create an interactive plot to choose the max X value
#             fig, ay = plt.subplots(figsize=(8, 6))
#             ay.scatter(y,x, color='blue', label='Data Points', alpha=0.7)
#             ay.set_title(f"Select max X Value for {filename}")
#             ay.set_ylabel("Raster Value")
#             ay.set_xlabel("Elevation")
#             ay.grid(True)
            
            
            
#             # Create a slider to adjust max X value
#             ay_slider = plt.ayes([0.1, 0.01, 0.8, 0.03], facecolor='lightgoldenrodyellow')
#             slider = Slider(ay_slider, 'max X', np.min(x), np.max(x), valinit=np.max(x), valstep=0.1)
            
            
            
#             def update(val):
           
#                 # Get selected max X value from the slider
#                 max_x_value = slider.val
#                 ay.clear()
#                 ay.scatter(x, y, color='blue', label='Data Points', alpha=0.7)
#                 ay.set_title(f"max X = {max_x_value} for {filename}")
#                 ay.set_xlabel("Raster Value")
#                 ay.set_ylabel("Elevation")
#                 ay.grid(True)
#                 ay.set_xlim([np.min(x), max_x_value])
#                 plt.draw()


            
#             slider.on_changed(update)

#             # Display the plot with interactive slider
#             plt.subplots_adjust(bottom=0.15)  # Adjust to make room for the slider
#             plt.show  # Show the plot and keep the interactive mode active
           
#             # Now filter the data based on the selected max x-value from the slider
#             max_x_value = slider.val
#             data_filtered = data[data['Raster_Value'] <= max_x_value]
            
#             # Perform linear regression on filtered data
#             x_filtered = data_filtered[['Raster_Value']].values
#             y_filtered = data_filtered['OrthometricHeight(m)'].values
#             model = LinearRegression()
#             model.fit(x_filtered, y_filtered)
            
#             print('check1')
#             # Calculate regression metrics
#             y_pred = model.predict(x_filtered)
#             r2 = r2_score(y_filtered, y_pred)
#             rmse = np.sqrt(mean_squared_error(y_filtered, y_pred))
#             coef = model.coef_[0]
#             intercept = model.intercept_
#             print('check2')
#             # Create the line of best fit equation
#             equation = f"y = {coef:.4f}x + {intercept:.4f}"

            
#             print('check3')

            
#             max_x = np.max(x)
#             max_y = np.max(y)
            
#             # Plot the data and the regression line
#             plt.figure(figsize=(8, 6))
#             plt.scatter(x, y, color='blue', label='Data Points', alpha=0.7)
#             plt.plot(x, y_pred, color='red', label='Best Fit Line', linewidth=2)
#             plt.title(f"Linear Regression for {filename}")
#             plt.xlabel("Raster Value")
#             plt.ylabel("Elevation")
#             #plt.xlim(None, 1.15)
#             plt.ylim(top=0)
#             plt.legend()
#             plt.grid(True)    
#             # Add R^2 and RMSE as text on the plot
#             plt.text(max_x, max_y, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}", fontsize=10, 
#                      bbox=dict(facecolor='white', alpha=0.5), ha='left')

#             print('check4')

#             # Invert both ayes so 0 is bottom left, and up and right are negative
#             plt.gca().invert_yayis()

#             # Generate and save the plot
#             plot_filename = f"{os.path.splitext(filename)[0]}_LR_plot.png"
#             plot_path = os.path.join(output_folder, plot_filename)
#             plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#             plt.close()  # Close the plot to free memory

#             print(f"Plot saved to {plot_path}")
            
#             # Append the results to the list
#             results.append({
#                 "Image Name": filename,
#                 "R^2": r2,
#                 "RMSE": rmse,        
#                 "Line of Best Fit": equation,
#                 "m1": coef,
#                 "m0": intercept})


#             # Convert the results into a DataFrame
#             results_df = pd.DataFrame(results)

#             # Generate the output file name
#             output_filename = f"{os.path.splitext(filename)[0]}_LR_stats.csv"
#             output_path = os.path.join(output_folder, output_filename)


#             # Save the results to the output CSV file
#             results_df.to_csv(output_path, index=False)
#             print(f"Results saved to {output_path}")


#       #################  CHECK DIRECTORIES/INPUTS #####################

# input_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts"  
# output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_Results" 
# process_csv_files(input_folder, output_folder)


# Perform linear regressions between pSDB red and green and reference calibration data

import os
import rasterio
import numpy as np
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

from glob import glob
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from rasterio.enums import Resampling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from rasterio.mask import mask
from difflib import get_close_matches


def process_csv_files(input_folder, output_folder):
    """
    Processes CSV files in the input folder, performs linear regression, 
    and saves the results in the output folder.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the output file path
    #output_file = os.path.join(output_folder, "linear_regression_results.csv")

    # Loop through all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")

            # Read the CSV file
            data = pd.read_csv(input_path)

            # Drop rows where 'Raster_Value' is blank
            data = data.dropna(subset=['Raster_Value'])

            # Initialize a results list
            results = []

            # Perform linear regression
            x = data[['Raster_Value']].values
            y = data['Orthometric Height(m)'].values
            model = LinearRegression()
            model.fit(x, y)

            # Calculate regression metrics
            y_pred = model.predict(x)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            coef = model.coef_[0]
            intercept = model.intercept_

            # Create the line of best fit equation
            equation = f"y = {coef:.4f}x + {intercept:.4f}"

            # Append the results to the list
            results.append({
                "Image Name": filename,
                "R^2": r2,
                "RMSE": rmse,        
                "Line of Best Fit": equation,
                "m1": coef,
                "m0": intercept
            })

            
            max_x = np.max(x)
            max_y = np.max(y)
            
            # Plot the data and the regression line
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, color='blue', label='Data Points', alpha=0.7)
            plt.plot(x, y_pred, color='red', label='Best Fit Line', linewidth=2)
            plt.title(f"Linear Regression for {filename}")
            plt.xlabel("Raster Value")
            plt.ylabel("Elevation")
            #plt.xlim(None, 1.15)
            plt.ylim(top=0)
            plt.legend()
            plt.grid(True)    
            # Add R^2 and RMSE as text on the plot
            plt.text(max_x, max_y, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}", fontsize=10, 
                      bbox=dict(facecolor='white', alpha=0.5), ha='left')
            plt.show
            # Invert both ayes so 0 is bottom left, and up and right are negative
            plt.gca().invert_yaxis()

            # Generate and save the plot
            plot_filename = f"{os.path.splitext(filename)[0]}_LR_plot.png"
            plot_path = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            #plt.close()  # Close the plot to free memory

            print(f"Plot saved to {plot_path}")


            # Convert the results into a DataFrame
            results_df = pd.DataFrame(results)

            # Generate the output file name
            output_filename = f"{os.path.splitext(filename)[0]}_LR_stats.csv"
            output_path = os.path.join(output_folder, output_filename)


            # Save the results to the output CSV file
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")


      #################  CHECK DIRECTORIES/INPUTS #####################

input_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts"  
output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_Results" 
process_csv_files(input_folder, output_folder)


##############################################################################
##############################################################################

# Create SDB red and green with constants from linear regression


import os
import rasterio
import numpy as np
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

from glob import glob
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from rasterio.enums import Resampling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from rasterio.mask import mask
from difflib import get_close_matches


def create_sdb_rasters(raster_folder, csv_folder, output_folder, nodata_value=np.nan):
    """
    Creates SDB rasters using constants from linear regression stored in CSV files.
    
    Args:
        raster_folder (str): Path to the folder containing input rasters.
        csv_folder (str): Path to the folder containing CSV files with coefficients.
        output_folder (str): Path to the folder where output rasters will be saved.
        nodata_value (float): NoData value to set in output rasters (default: NaN).
    """
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all CSV files from the specified folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    # Loop through each raster in the folder
    for raster_name in os.listdir(raster_folder):
        if raster_name.endswith('.tif'):  # Ensure only .tif files are processed
            raster_path = os.path.join(raster_folder, raster_name)
            base_raster_name = raster_name.replace('.tif', '')

            # Find the closest matching CSV file to the raster name
            closest_csv = get_close_matches(base_raster_name, csv_files, n=1, cutoff=0.6)
            if closest_csv:
                csv_file = closest_csv[0]
                csv_path = os.path.join(csv_folder, csv_file)

                # Read the CSV file containing coefficients
                coefficients_df = pd.read_csv(csv_path)

                # Process the raster using the matched CSV file
                with rasterio.open(raster_path) as src:
                    pSDB = src.read(1)  # Assuming the raster is single-band

                    # Find the row in the CSV where the raster name matches
                    coeff_row = coefficients_df[coefficients_df['Image Name'].str.contains(base_raster_name, 
                                                                                           case=False, na=False)]
                    if not coeff_row.empty:
                        # Extract coefficients
                        m1 = coeff_row['m1'].values[0]
                        m0 = coeff_row['m0'].values[0]

                        # Perform the SDB raster calculation
                        result = m1 * pSDB + m0

                        # Mask NaN values and set NoData value
                        result = np.ma.masked_where(np.isnan(result), result)
                        result_filled = result.filled(nodata_value)

                        
                        # Generate output raster path based on the input raster name
                        if raster_name.endswith('_pSDBgreen.tif'):
                            output_raster_name = raster_name.replace('_pSDBgreen.tif', '_SDBgreen.tif')
                        elif raster_name.endswith('_pSDBred.tif'):
                            output_raster_name = raster_name.replace('_pSDBred.tif', '_SDBred.tif')
                        else:
                            print(f"Skipping {raster_name}: does not end with either pSDBgreen or pSDBred")
                            continue

                        output_raster_path = os.path.join(output_folder, output_raster_name)

                        # Write to output raster
                        with rasterio.open(output_raster_path, 'w',
                                           driver='GTiff',
                                           count=1,
                                           dtype=result.dtype,
                                           crs=src.crs,
                                           transform=src.transform,
                                           width=src.width,
                                           height=src.height,
                                           nodata=nodata_value) as dst:
                            dst.write(result_filled, 1)

                        print(f"Saved SDB raster: {output_raster_path}")
                    else:
                        print(f"No matching row found in CSV for {raster_name}")
            else:
                print(f"No matching CSV file found for {raster_name}")


# ############## CHECK DIRECTORIES/INPUTS ###########################

raster_folder = r"E:\Thesis Stuff\pSDB"
csv_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_Results"
output_folder = r"E:\Thesis Stuff\SDB"

create_sdb_rasters(raster_folder, csv_folder, output_folder)


##############################################################################
##############################################################################

# Merge SDB red and green together

import os
import rasterio
import numpy as np
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

from glob import glob
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from rasterio.enums import Resampling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from rasterio.mask import mask
from difflib import get_close_matches


def create_sdb_raster(sdb_red, sdb_green):
    """
    Create a merged SDB raster with the following rules:
    - If 0 >= SDBred >= -2, use SDBred.
    - If -2 > SDBred >= -3.5, use a weighted average of SDBred and SDBgreen.
    - If SDBred < -3.5, use SDBgreen.
    """
    # Initialize output array
    sdb_merged = np.full_like(sdb_red, np.nan)

    lower_limit = -2 
    upper_limit = -3.5

    # Debug: Check input min/max values
    print(f"SDB Red Min: {np.nanmin(sdb_red)}, max: {np.nanmax(sdb_red)}")
    print(f"SDB Green Min: {np.nanmin(sdb_green)}, max: {np.nanmax(sdb_green)}")

    # Condition 1: Use SDBred if 0 >= SDBred >= -2
    red_condition = sdb_red > lower_limit
    print(f"Red Condition Count: {np.sum(red_condition)}")
    sdb_merged[red_condition] = sdb_red[red_condition]

    # Condition 2: Weighted average if -2 > SDBred >= -3.5
    weighted_condition = (sdb_red <= lower_limit) & (sdb_green >= upper_limit) & (sdb_red >sdb_green)    
    print(f"Weighted Condition Count: {np.sum(weighted_condition)}")

    # Calculate weights (linear relationship)
    alpha = (sdb_red - lower_limit) / (upper_limit - lower_limit)  # Correct weight calculation
    beta = 1 - alpha
    print(f"Alpha Min: {np.nanmin(alpha)}, max: {np.nanmax(alpha)}")
    print(f"Beta Min: {np.nanmin(beta)}, max: {np.nanmax(beta)}")

    sdb_weighted = alpha * sdb_red + beta * sdb_green
    sdb_merged[weighted_condition] = sdb_weighted[weighted_condition]

    # Condition 3: Use SDBgreen if SDBred < -3.5
    green_condition = (sdb_red <= lower_limit) & (sdb_green < upper_limit) | (sdb_red <= lower_limit) & (sdb_green >= upper_limit) & (sdb_red <=sdb_green)
    
    
    print(f"Green Condition Count: {np.sum(green_condition)}")
    sdb_merged[green_condition] = sdb_green[green_condition]

    # Debug: Check output min/max
    print(f"SDB Merged Min: {np.nanmin(sdb_merged)}, max: {np.nanmax(sdb_merged)}")

    return sdb_merged


# def process_sdb_folder(input_folder, csv_folder, r2_threshold=0.7):
#     """Processes all matching SDBred and SDBgreen rasters in a folder, merging only if R² ≥ threshold."""

#     # Get lists of all SDBred and SDBgreen files
#     sdb_red_files = glob(os.path.join(input_folder, '*SDBred.tif'))
#     sdb_green_files = glob(os.path.join(input_folder, '*SDBgreen.tif'))

#     # Create dictionaries mapping filenames without 'SDBred' or 'SDBgreen' to their full paths
#     red_dict = {os.path.basename(f).replace('SDBred.tif', ''): f for f in sdb_red_files}
#     green_dict = {os.path.basename(f).replace('SDBgreen.tif', ''): f for f in sdb_green_files}

#     # Find matching pairs
#     common_keys = set(red_dict.keys()) & set(green_dict.keys())

#     for key in common_keys:
#         red_raster_path = red_dict[key]
#         green_raster_path = green_dict[key]
#         output_path = os.path.join(input_folder, f"{key}SDB_merged.tif")

#         # Find corresponding CSV file for R² lookup
#         csv_file = os.path.join(csv_folder, f"{key}_LR_stats.csv")

#         if not os.path.exists(csv_file):
#             print(f"Warning: No regression CSV found for {key}, skipping merge.")
#             continue

#         # Read the R² value from the CSV
#         try:
#             df = pd.read_csv(csv_file)
#             r2_value = df.loc[0, 'R^2']  # Assuming the R^2 value is in the first row

#             if r2_value < r2_threshold:
#                 print(f"Skipping merge for {key}: R² = {r2_value:.4f} (below {r2_threshold})")
#                 continue  # Skip merging for this raster pair

#         except Exception as e:
#             print(f"Error reading {csv_file}: {e}")
#             continue  # Skip this file and move on

#         # Open raster files and read data
#         with rasterio.open(red_raster_path) as red_dataset:
#             sdb_red = red_dataset.read(1)

#         with rasterio.open(green_raster_path) as green_dataset:
#             sdb_green = green_dataset.read(1)

#         # Create the merged SDB raster
#         sdb_merged = create_sdb_raster(sdb_red, sdb_green)

#         # Save the new merged raster
#         with rasterio.open(output_path, 'w', driver='GTiff', count=1, dtype=sdb_merged.dtype,
#                            height=sdb_merged.shape[0], width=sdb_merged.shape[1],
#                            crs=red_dataset.crs, transform=red_dataset.transform) as dst:
#             dst.write(sdb_merged, 1)

#         print(f"Saved merged SDB raster: {output_path}")


# # Example usage
# input_folder = r"E:\Thesis Stuff\SDB"  # Folder with input rasters
# csv_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_Results"  # Folder containing regression results

# process_sdb_folder(input_folder, csv_folder, r2_threshold=0.7)



# This way still works
def process_sdb_folder(input_folder):
    """Processes all matching SDBred and SDBgreen rasters in a folder."""

    # Get lists of all SDBred and SDBgreen files
    sdb_red_files = glob(os.path.join(input_folder, '*SDBred.tif'))
    sdb_green_files = glob(os.path.join(input_folder, '*SDBgreen.tif'))

    # Create a dictionary mapping filenames without 'SDBred' or 'SDBgreen' to their full paths
    red_dict = {os.path.basename(f).replace('SDBred.tif', ''): f for f in sdb_red_files}
    green_dict = {os.path.basename(f).replace('SDBgreen.tif', ''): f for f in sdb_green_files}

    # Find matching pairs
    common_keys = set(red_dict.keys()) & set(green_dict.keys())

    for key in common_keys:
        red_raster_path = red_dict[key]
        green_raster_path = green_dict[key]
        output_path = os.path.join(input_folder, f"{key}SDB_merged.tif")


        # Open raster files and read data
        with rasterio.open(red_raster_path) as red_dataset:
            sdb_red = red_dataset.read(1)  # Read first band

        with rasterio.open(green_raster_path) as green_dataset:
            sdb_green = green_dataset.read(1)  # Read first band

        # Create the merged SDB raster
        sdb_merged = create_sdb_raster(sdb_red, sdb_green)

        # Save the new merged raster
        with rasterio.open(output_path, 'w', driver='GTiff', count=1, dtype=sdb_merged.dtype,
                            height=sdb_merged.shape[0], width=sdb_merged.shape[1],
                            crs=red_dataset.crs, transform=red_dataset.transform) as dst:
            dst.write(sdb_merged, 1)

        print(f"Saved merged SDB raster: {output_path}")

# Example usage
input_folder = r"E:\Thesis Stuff\SDB"  # Folder with input rasters

process_sdb_folder(input_folder)




##############################################################################################################
##############################################################################################################

# Comparisons, SDB truthiness, other stats

# Ok so I will have lots of SDB from both S2 and SD with the exact same boundary/area.
# I should somehow use Matt's bathy truthiness tool to see what percent of area 
# bathymetry can be identified between SD and S2. Would assume S2 to have a 
# higher percent because it has better camera and sensors. Could then average 
# percent that SD reaches vs S2. Higher percent = more optically shallow waters,
# low percent = more optically deep waters.



##############################################################################################################
##############################################################################################################
# Extract SDB values at reference point locations

import os
import rasterio
import numpy as np
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

from glob import glob
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from rasterio.enums import Resampling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from rasterio.mask import mask
from difflib import get_close_matches

def extract_raster_values(csv_file, raster_folder, output_folder):
    """
    Extracts raster values at the locations provided in the CSV file and saves the results.

    Args:
        csv_file (str): Path to the CSV file containing latitude, longitude, and elevation.
        raster_folder (str): Path to the folder containing GeoTIFF raster files.
        output_folder (str): Path to the folder where the results will be saved.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    df.columns = ['Easting(m)', 'Northing(m)', 'Orthometric Height(m)']
    
    # Convert the CSV to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_xy(df['Easting(m)'], df['Northing(m)']))
    
    # Loop through all raster files in the folder
    for raster_file in os.listdir(raster_folder):
        if raster_file.endswith(".tif"):  # Assuming GeoTIFF raster files
            raster_path = os.path.join(raster_folder, raster_file)
            print(f"Processing raster: {raster_file}")

            # Open the raster file using rasterio
            with rasterio.open(raster_path) as src:
                
                # Filter reference points that are within the raster bounds
                raster_bounds = src.bounds
                gdf_in_bounds = gdf[gdf.geometry.apply(lambda point: is_point_within_bounds(point, raster_bounds))]

                # Ensure there are points within the bounds
                if gdf_in_bounds.empty:
                    print(f"No points overlap with raster: {raster_file}")
                    continue
                
                # Extract the raster values at each valid location
                gdf_in_bounds.loc[:, 'Raster_Value'] = gdf_in_bounds.geometry.apply(lambda point: get_raster_value_at_point(src, point))
            
            # Check if 'Raster_Value' exists and drop rows with NaN
            if 'Raster_Value' in gdf_in_bounds.columns:
                gdf_filtered = gdf_in_bounds.dropna(subset=['Raster_Value'])
            else:
                print(f"No valid raster values found for raster: {raster_file}")
                continue
            
            # Save the results to a new CSV file
            output_filename = os.path.splitext(raster_file)[0] + "_extracted.csv"
            output_path = os.path.join(output_folder, output_filename)
            gdf_filtered.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")


      #################  CHECK DIRECTORIES/INPUTS #####################

csv_file = r"E:\Thesis Stuff\ReferenceData\Topobathy\Marathon_2017_combined_and_downsampled_points.csv"
raster_folder = r"E:\Thesis Stuff\SDB"
output_folder = r"E:\Thesis Stuff\SDB_ExtractedPts"

extract_raster_values(csv_file, raster_folder, output_folder)



##############################################################################################################
##############################################################################################################
# Perform linear regressions between SDB and other reference data for accuracy


import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
import re
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from rasterio.mask import mask
from difflib import get_close_matches
import xarray as xr
from glob import glob
from rasterio.warp import calculate_default_transform, reproject, Resampling

def process_csv_files(input_folder, output_folder):
    """
    Processes CSV files in the input folder, performs linear regression, 
    and saves the results in the output folder.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")

            # Read the CSV file
            data = pd.read_csv(input_path)

            # Drop rows where 'Raster_Value' is blank
            data = data.dropna(subset=['Raster_Value'])

            # Initialize a results list
            results = []

            # Perform linear regression
            x = data[['Raster_Value']].values
            y = data['Orthometric Height(m)'].values
            model = LinearRegression()
            model.fit(x, y)

            # Calculate regression metrics
            y_pred = model.predict(x)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            coef = model.coef_[0]
            intercept = model.intercept_

            # Create the line of best fit equation
            equation = f"y = {coef:.4f}x + {intercept:.4f}"

            # Calculate regression metrics
            y_pred = model.predict(x)
            r2 = r2_score(y, y_pred)  # Scikit-learn R² calculation
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            # Calculate perpendicular distances
            distances = np.abs(coef * x.flatten() - y + intercept) / np.sqrt(coef**2 + 1)
            
            # Compute statistics for distances
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            data["Perpendicular distances to line"] = distances


            # Append the results to the list
            results.append({
                "Image Name": filename,
                "R^2": r2,
                "RMSE": rmse,        
                "Line of Best Fit": equation,
                "m1": coef,
                "m0": intercept,
                "min perp dist": min_dist,
                "max perp dist": max_dist,
                "mean perp dist": mean_dist,
                "std dev perp dist": std_dist 
            })
            
            max_x = np.max(x)
            max_y = np.max(y)

            # Compute the x-intercept (where y = 0)
            x_intercept = -intercept / coef if coef != 0 else np.min(x)

            # Generate extended x values from x_intercept to max x
            x_ext = np.linspace(x_intercept, np.min(x), 100)

            # Compute corresponding y values
            y_ext = coef * x_ext + intercept

            # Regression Plot
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, color='blue', label='Data Points', alpha=0.7)
            plt.plot(x_ext, y_ext, color='red', label='Best Fit Line', linewidth=2)
            plt.title(f"Linear Regression for {filename}")
            plt.xlabel("Raster Value")
            plt.ylabel("Elevation")
            plt.legend()
            plt.grid(True)    
            plt.xlim(None, 0)
            plt.ylim(None, 0)
            
            # Add R^2 and RMSE as text on the plot
            plt.text(max_x, max_y, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nIntercept = {intercept:0.2f}", fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.5), ha='left')
            
            # Invert both ayes so 0 is bottom left, and up and right are negative
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()

            # Save the regression plot in the output folder
            plot_filename = f"{os.path.splitext(filename)[0]}_LR_plot.png"
            plot_path = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_path)
            #print(f"Regression plot saved as {plot_path}")
            plt.close()


            max_x_perp = np.max(data["Raster_Value"])
            min_y_perp = np.min(distances)

           ## Scatter plot of perpendicular distance vs. raster value
           #  plt.figure(figsize=(8, 6))
           #  plt.scatter(data["Raster_Value"], distances, color='purple', alpha=0.5)
           #  plt.ayhline(mean_dist, color='red', linestyle='dashed', label="Mean Distance")
           #  plt.xlabel("Raster Value")
           #  plt.ylabel("Perpendicular Distance to Best Fit Line")
           #  plt.title(f"Perpendicular Distance vs. Raster Value ({filename})")
           #  plt.legend()
           #  plt.grid(True)
           #  # Invert both ayes so 0 is bottom left, and up and right are negative
           #  plt.gca().invert_xaxis()
           # # plt.gca().invert_yaxis()
           #  # Perform linear regression for the perpendicular distance vs raster value
           #  x_perp = data["Raster_Value"].values.reshape(-1, 1)
           #  y_perp = distances
           #  model_perp = LinearRegression()
           #  model_perp.fit(x_perp, y_perp)


           #  # Calculate regression metrics for the perpendicular distance plot
           #  y_pred_perp = model_perp.predict(x_perp)
           #  r2_perp = r2_score(y_perp, y_pred_perp)
           #  rmse_perp = np.sqrt(mean_squared_error(y_perp, y_pred_perp))
           #  coef_perp = model_perp.coef_[0]
           #  intercept_perp = model_perp.intercept_

           #  # Create the line of best fit equation for the perpendicular distance
           #  equation_perp = f"y = {coef_perp:.4f}x + {intercept_perp:.4f}"

           #  # Add the line of best fit to the plot
           #  plt.plot(data["Raster_Value"], y_pred_perp, color='green', label='Best Fit Line', linewidth=2)
            
           #  # Add R^2 and RMSE as text on the plot
           #  plt.text(max_x_perp, min_y_perp, f"$R^2$ = {r2_perp:.4f}\nRMSE = {rmse_perp:.4f}", fontsize=10, 
           #           bbox=dict(facecolor='white', alpha=0.5), ha='left')

           #  # Save the perpendicular distance plot in the output folder
           #  perp_plot_filename = f"{os.path.splitext(filename)[0]}_PerpDist_plot.png"
           #  perp_plot_path = os.path.join(output_folder, perp_plot_filename)
           #  plt.savefig(perp_plot_path)
           #  print(f"Perpendicular distance plot saved as {perp_plot_path}")
           #  plt.close()
            
            
            # Convert the results into a DataFrame
            results_df = pd.DataFrame(results)

            # Generate the output file name
            output_filename = f"{os.path.splitext(filename)[0]}_LR_stats.csv"
            output_path = os.path.join(output_folder, output_filename)


            # Save the results to the output CSV file
            results_df.to_csv(output_path, index=False)
            #print(f"Results saved to {output_path}")


      #################  CHECK DIRECTORIES/INPUTS #####################

input_folder = r"E:\Thesis Stuff\SDB_ExtractedPts"  
output_folder = r"E:\Thesis Stuff\SDB_ExtractedPts_Results" 
process_csv_files(input_folder, output_folder)



##############################################################################################################
##############################################################################################################

# Calculate the deviations for each point from the line of best fit  






































