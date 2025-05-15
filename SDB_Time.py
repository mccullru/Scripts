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
            output_file = os.path.join(output_folder, f"{Path(output_file_name).stem}_RGB.tif")

            # Write combined RGB TIFF
            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(red_data, 1)   # Red band
                dst.write(green_data, 2) # Green band
                dst.write(blue_data, 3)  # Blue band

            print(f"Combined RGB saved to: {output_file}")

        except Exception as e:
            print(f"Error processing {r}, {g}, {b}: {e}")


################# CHECK DIRECTORIES/INPUTS #####################

input_folder = r"E:\Thesis Stuff\AcoliteWithPython\Corrected_Imagery\All_Sentinel2\S2_Anegada_output"
output_folder = r"E:\Thesis Stuff\RGBCompositOutput"
combine_bands_to_rgb(input_folder, output_folder)



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


" !!! IF you want to keep the RGB outputs, change delete_input_files to FALSE !!!"

def process_rgb_geotiffs(input_folder, output_folder, delete_input_files=True):
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
                scaled_red = red_band * 100000
                scaled_green = green_band * 100000
                scaled_blue = blue_band * 100000

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


     # --- MINIMAL CHANGE 2: Added optional deletion block ---
    if delete_input_files:
        print("\n--- Deleting Input Files ---")
        print(f"WARNING: Attempting to delete FILES from input folder: {input_folder}")
        print("         (Subdirectories will be skipped)")
        deleted_count = 0
        deletion_errors = 0
        try: # Wrap the whole deletion attempt
             for item_name in os.listdir(input_folder):
                 item_path = os.path.join(input_folder, item_name)
                 try:
                     if os.path.isfile(item_path): # Delete only files
                         os.remove(item_path)
                         deleted_count += 1
                 except Exception as e_del:
                     print(f"  Error deleting file {item_name}: {e_del}")
                     deletion_errors += 1
             print(f"Deletion attempt complete. Deleted {deleted_count} files. Errors: {deletion_errors}")
        except Exception as e_list:
             print(f"ERROR: Could not list or access input folder for deletion: {input_folder} - {e_list}")
    # --- END MINIMAL CHANGE ---

    print("\nFunction process_rgb_geotiffs finished.") # Added



      #################  CHECK DIRECTORIES/INPUTS #####################

input_folder = r"E:\Thesis Stuff\RGBCompositOutput"

# Save Results Path
#output_folder = r"B:\Thesis Project\SDB_Time\Results_main\BumBum\Sentinel-2\pSDB"

# Workspace Path
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



""" Input a folder of reference data and match the names """
# def extract_raster_values(csv_folder, raster_folder, output_folder):
#     """
#     Matches rasters and CSVs where the second word in the raster filename 
#     matches the first word in the CSV filename.
#     """
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Build dictionary: {scene_id (from csv): full csv path}
#     csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
#     csv_prefix_map = {f.split('_')[0].lower(): os.path.join(csv_folder, f) for f in csv_files}

#     for raster_file in os.listdir(raster_folder):
#         if not raster_file.endswith(".tif"):
#             continue

#         parts = raster_file.split('_')
#         if len(parts) < 2:
#             print(f"Skipping invalid raster name: {raster_file}")
#             continue

#         scene_id = parts[1].lower()
#         matching_csv_path = csv_prefix_map.get(scene_id)

#         if not matching_csv_path:
#             print(f"No matching CSV for raster {raster_file} (SceneID: {scene_id})")
#             continue

#         print(f"Processing raster: {raster_file} with CSV: {os.path.basename(matching_csv_path)}")

#         # Load and prepare CSV
#         df = pd.read_csv(matching_csv_path)
#         easting = df.iloc[:, 0]
#         northing = df.iloc[:, 1]
#         height = df.iloc[:, 2]
#         gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_xy(easting, northing))

#         raster_path = os.path.join(raster_folder, raster_file)
#         with rasterio.open(raster_path) as src:
#             bounds = src.bounds
#             gdf_in_bounds = gdf[gdf.geometry.apply(lambda pt: is_point_within_bounds(pt, bounds))]

#             if gdf_in_bounds.empty:
#                 print(f"No points in bounds for: {raster_file}")
#                 continue

#             gdf_in_bounds.loc[:, 'Raster_Value'] = gdf_in_bounds.geometry.apply(lambda pt: get_raster_value_at_point(src, pt))

#         gdf_filtered = gdf_in_bounds.dropna(subset=['Raster_Value'])

#         output_filename = os.path.splitext(raster_file)[0] + "_extracted.csv"
#         output_path = os.path.join(output_folder, output_filename)
#         gdf_filtered.to_csv(output_path, index=False)
#         print(f"Saved: {output_path}")

# def get_raster_value_at_point(raster_src, point):
#     """
#     Gets the raster value at a specific point (latitude, longitude).
    
#     Args:
#         raster_src (rasterio.io.DatasetReader): The raster source object.
#         point (shapely.geometry.point.Point): The point geometry (Longitude, Latitude).
    
#     Returns:
#         float: The raster value at the specified point.
#     """
#     try:
#         # Convert point to raster coordinates
#         row, col = raster_src.index(point.x, point.y)
#         value = raster_src.read(1)[row, col]
#     except (IndexError, ValueError):
#         # Return a NoData value or NaN if the point is out of bounds
#         value = float('nan')
    
#     return value

# def is_point_within_bounds(point, bounds):
#     """
#     Checks if a point is within raster bounds.
    
#     Args:
#         point (shapely.geometry.point.Point): The point geometry.
#         bounds (tuple): The raster bounds (left, bottom, right, top).
    
#     Returns:
#         bool: True if the point is within bounds, False otherwise.
#     """
#     left, bottom, right, top = bounds
#     return left <= point.x <= right and bottom <= point.y <= top

# csv_folder = r"B:\Thesis Project\Reference Data\ICESat_refractionCorrected"
# raster_folder = r"E:\Thesis Stuff\pSDB"
# output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts"

# extract_raster_values(csv_folder, raster_folder, output_folder)



""" Specifically choose reference data file"""
def extract_raster_values(cal_csv_file, raster_folder, output_folder):
    """
    Extracts raster values at the locations provided in the CSV file and saves the results.

    Args:
        cal_csv_file (str): Path to the CSV file containing latitude, longitude, and elevation.
        raster_folder (str): Path to the folder containing GeoTIFF raster files.
        output_folder (str): Path to the folder where the results will be saved.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the CSV file
    df = pd.read_csv(cal_csv_file)

   
    # Multiply the height column by -1 DON'T DO IF ALREADY POSITIVE
    print(f"Multiplying height column '{df.columns[2]}' (index 2) by -1.")
    # Make sure the column is positive before multiplying
    df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce') * -1
   
    # Remove rows where final depth < 0 
    original_rows = len(df)
    
    # Removes rows that became NaN during `to_numeric` conversion
    df = df[df.iloc[:, 2] >= 1]
    rows_removed = original_rows - len(df)
    if rows_removed > 0: # Added print for feedback
         print(f"Removed {rows_removed} rows with negative or NaN depth values.")

    
    # Easting has to be first column, northing second, and ortho heights third
    easting = df.iloc[:, 0]
    northing = df.iloc[:, 1]
    # height = df.iloc[:, 2]
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_xy(easting, northing))
    
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
            
            # Check if 'Raster_Value' exists and drop rows with NaN (NoData or out-of-bounds)
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

def get_raster_value_at_point(src, point):
    """
    Gets the raster value at a specific point (latitude, longitude).
    Returns None if the value is NoData.
    """
    try:
        # Convert point to raster coordinates (row, column)
        row, col = src.index(point.x, point.y)
        value = src.read(1)[row, col]
        
        # Check if value is NoData and return None if so
        if value == src.nodata:
            return None  # Return None for NoData
        else:
            return value
    except (IndexError, ValueError):
        # If an error occurs (e.g., point is out of bounds), return None
        return None

def is_point_within_bounds(point, bounds):
    """
    Checks if a point is within raster bounds.
    """
    return bounds.left <= point.x <= bounds.right and bounds.bottom <= point.y <= bounds.top



      #################  CHECK DIRECTORIES/INPUTS #####################

cal_csv_file = r"B:\Thesis Project\Reference Data\Processed_ICESat\Anegada_corrected.csv"     # Calibration reference data

### Save Results Path ###
#raster_folder = r"B:\Thesis Project\SDB_Time\Results_main\BumBum\Sentinel-2\pSDB"

### Workspace Path ###
raster_folder = r"E:\Thesis Stuff\pSDB"


### Save Results Path ###
#output_folder = r"B:\Thesis Project\SDB_Time\Results_main\BumBum\Sentinel-2\Extracted Pts\pSDB"

### Workspace Path ###
output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts"

extract_raster_values(cal_csv_file, raster_folder, output_folder)



##############################################################################
##############################################################################

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
            y = data['Geoid_Corrected_Ortho_Height'].values
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

            
            min_x = np.min(x)
            mean_y = np.mean(y)
            
            # Plot the data and the regression line
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, color='blue', label='Data Points', alpha=0.7)
            plt.plot(x, y_pred, color='red', label='Best Fit Line', linewidth=2)
            plt.title(f"Linear Regression for {filename}")
            plt.xlabel("pSDB Values (unitless)")
            plt.ylabel("Reference Depths (m)")
            #plt.xlim(None, 1.5)
            plt.ylim(0, None)
            plt.legend()
            plt.grid(True)    
            # Add R^2 and RMSE as text on the plot
            plt.text(min_x, mean_y, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}", fontsize=10, 
                      bbox=dict(facecolor='white', alpha=0.5), ha='left')
            plt.show
            # Invert both ayes so 0 is bottom left, and up and right are negative
            #plt.gca().invert_yaxis()

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

### Save Results Path ###
#input_folder = r"B:\Thesis Project\SDB_Time\Results_main\BumBum\Sentinel-2\Extracted Pts\pSDB"

### Workspace Path ###
input_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts"


### Save Results Path ###
#output_folder = r"B:\Thesis Project\SDB_Time\Results_main\BumBum\Sentinel-2\Figures\pSDB"

### Workspace Path ###
output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_Results"


process_csv_files(input_folder, output_folder)


##############################################################################################################
##############################################################################################################

# Perform another linear regression but this time find the best line of fit with the highest R^2 value


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import glob


data_folder_path = r"E:\Thesis Stuff\pSDB_ExtractedPts" 

output_save_folder_path = r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results"


# Create the output folder if it doesn't exist
if not os.path.exists(output_save_folder_path):
    os.makedirs(output_save_folder_path)
    print(f"Created output folder: {output_save_folder_path}")

csv_files = glob.glob(os.path.join(data_folder_path, "*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in the specified folder: {data_folder_path}")

print(f"Found {len(csv_files)} CSV files to process in: {data_folder_path}")

for data_name_full_path in csv_files:
    print(f"\n--- Processing file: {data_name_full_path} ---")
    current_file_name_for_output = os.path.basename(data_name_full_path)
    just_the_filename_for_output_csv = os.path.splitext(current_file_name_for_output)[0]
    
    fig = None 
    current_file_iterations_data = [] # Initialize list for THIS FILE'S iteration data

    try:
        # --- Load Data ---
        if not os.path.isfile(data_name_full_path):
            print(f'Warning: Data CSV file not found: {data_name_full_path}')
            continue
        try:
            data_df = pd.read_csv(data_name_full_path)
            if data_df.shape[1] < 5:
                raise ValueError('CSV file does not have enough columns. Expecting at least 5.')

            y_original = data_df.iloc[:, 2].values.astype(float)
            x_original = data_df.iloc[:, 4].values.astype(float)

            nan_mask = ~ (np.isnan(x_original) | np.isnan(y_original))
            x_original = x_original[nan_mask]
            y_original = y_original[nan_mask]

            if x_original.size == 0 or y_original.size == 0:
                raise ValueError('No valid data points after removing NaNs from x or y.')
            # print(f'Loaded {len(x_original)} valid data points after initial NaN removal.') # Less verbose

            # print('Filtering pSDB (x) values to keep only those >= 0 and <= 5...') # Less verbose
            original_point_count = len(x_original)
            valid_x_idx = (x_original >= 0) & (x_original <= 5)
            x_original = x_original[valid_x_idx]
            y_original = y_original[valid_x_idx]
            # print(f'Retained {len(x_original)} points after filtering x values (removed {original_point_count - len(x_original)} points).') # Less verbose
            if x_original.size == 0 or y_original.size == 0:
                raise ValueError('No data points remain after filtering x values between 0 and 5.')

        except Exception as e:
            print(f'Failed to read or process input CSV file {data_name_full_path}. Error: {e}')
            continue

        data_name_for_conditions = str(just_the_filename_for_output_csv).lower()


        # --- Initial Plot Setup ---
        fig, ax = plt.subplots(figsize=(10, 7))

        if "green" in data_name_for_conditions: depth_min_limit_for_plot = 2.0
        elif "red" in data_name_for_conditions: depth_min_limit_for_plot = 0.0
        else: depth_min_limit_for_plot = 0.0

        plot_filter_idx = (y_original >= depth_min_limit_for_plot)
        x_for_scatter = x_original[plot_filter_idx]
        y_for_scatter = y_original[plot_filter_idx]

        scatter_handle = None
        if len(x_for_scatter) == 0:
            print('Warning: No data points meet the initial depth_min_limit for plotting. Scatter plot will be empty.')
            scatter_handle = ax.scatter([], [], s=36, c='k', alpha=0.3, label='Filtered Data Points (None)')
        else:
            scatter_handle = ax.scatter(x_for_scatter, y_for_scatter, s=36, c='k', alpha=0.3, label='Filtered Data Points')

        ax.set_xlabel('pSDB Value')
        ax.set_ylabel('Reference Depth (m)')
        ax.set_title(f'pSDB Linear Regression Analysis: {just_the_filename_for_output_csv.replace("_", " ")}')
        ax.grid(True)



        # --- Iterative Regression Section ---
        if "green" in data_name_for_conditions:
            depth_min_limit_regr = 2.0; overall_max_depth = 20.0; step = 0.25; initial_depth_max = 2.5
        elif "red" in data_name_for_conditions:
            depth_min_limit_regr = 1.0; overall_max_depth = 20.0; step = 0.25; initial_depth_max = 0.5
        else:
            depth_min_limit_regr = 1.0; overall_max_depth = 12.0; step = 0.25; initial_depth_max = 0.5

        if initial_depth_max > overall_max_depth:
            depth_max_limits_to_test = np.array([overall_max_depth])
        else:
            depth_max_limits_to_test = np.arange(initial_depth_max, overall_max_depth + step, step)
            if depth_max_limits_to_test.size == 0 or (depth_max_limits_to_test[-1] < overall_max_depth and abs(depth_max_limits_to_test[-1] - overall_max_depth) > 1e-9) :
                depth_max_limits_to_test = np.append(depth_max_limits_to_test, overall_max_depth)

        if depth_min_limit_regr < 1.0 and initial_depth_max > 1.0:
            if 1.0 not in depth_max_limits_to_test: depth_max_limits_to_test = np.sort(np.append(depth_max_limits_to_test, 1.0))
        elif initial_depth_max <= 1.0:
            if 1.0 not in depth_max_limits_to_test and 1.0 <= overall_max_depth and 1.0 >= initial_depth_max:
                depth_max_limits_to_test = np.sort(np.append(depth_max_limits_to_test, 1.0))
        
        if len(depth_max_limits_to_test) == 0 and initial_depth_max <= overall_max_depth:
             depth_max_limits_to_test = np.array([overall_max_depth])

        if len(depth_max_limits_to_test) == 0:
            print('Warning: No depth ranges defined to test for regression. Skipping regression for this file.')
            plot_filename = f"{just_the_filename_for_output_csv}_plot.png"
            plot_save_path = os.path.join(output_save_folder_path, plot_filename)
            plt.savefig(plot_save_path); print(f"Plot saved to: {plot_save_path}")
            plt.show(); plt.close(fig)
            continue

        iteration_results_for_plot_obj = [] # For finding best line for the plot object
        num_iterations = len(depth_max_limits_to_test)
        print(f'Calculating regression for {num_iterations} depth ranges for {current_file_name_for_output}...')
        for k in range(num_iterations):
            current_depth_max_limit = depth_max_limits_to_test[k]
            
            plot_result_entry = {'depth_limit': current_depth_max_limit, 'R2': np.nan, 'params': None, 
                                 'point_count': 0, 'x_min_fit': np.nan, 'x_max_fit': np.nan}

            m_for_iteration, b_for_iteration = np.nan, np.nan
            equation_for_iteration = "N/A"
            R2_for_iteration = np.nan
            rmse_for_iteration = np.nan
            
            range_idx = (y_original >= depth_min_limit_regr) & (y_original <= current_depth_max_limit)
            x_iter_range = x_original[range_idx]
            y_iter_range = y_original[range_idx]
            num_points = len(x_iter_range)
            plot_result_entry['point_count'] = num_points
            
            if num_points > 1:
                params = np.polyfit(x_iter_range, y_iter_range, 1)
                y_fit_iter_range = np.polyval(params, x_iter_range)
                
                if len(np.unique(y_iter_range)) > 1:
                    R2_for_iteration = r2_score(y_iter_range, y_fit_iter_range)
                    if R2_for_iteration < 0: R2_for_iteration = 0.0
                elif len(x_iter_range) > 0: # All y are same
                    if np.allclose(y_iter_range, y_fit_iter_range): R2_for_iteration = 1.0
                    else: R2_for_iteration = 0.0
                
                plot_result_entry['R2'] = R2_for_iteration
                plot_result_entry['params'] = params
                if len(x_iter_range) > 0:
                    plot_result_entry['x_min_fit'] = np.min(x_iter_range)
                    plot_result_entry['x_max_fit'] = np.max(x_iter_range)

                m_for_iteration = params[0]
                b_for_iteration = params[1]
                equation_for_iteration = f"y = {m_for_iteration:.4f}x + {b_for_iteration:.4f}"
                rmse_for_iteration = np.sqrt(mean_squared_error(y_iter_range, y_fit_iter_range))
            
            iteration_results_for_plot_obj.append(plot_result_entry)
            
            current_file_iterations_data.append({
                'Image Name': current_file_name_for_output, # Redundant if filename is in CSV name, but good for completeness
                'Min Depth Range': depth_min_limit_regr,
                'Max Depth Range': current_depth_max_limit,
                'R2 Value': R2_for_iteration,
                'RMSE': rmse_for_iteration,
                'Line of Best Fit': equation_for_iteration,
                'm1': m_for_iteration,
                'm0': b_for_iteration,
                'Pt Count': num_points 
            })
        
        results_df_for_plot = pd.DataFrame(iteration_results_for_plot_obj)


        # --- Find the iteration with the best R² (for plotting purposes) ---
        best_R2_for_plot = -np.inf
        best_fit_params_for_plot = None
        best_depth_limit_for_plot_annotation = np.nan
        best_k_overall_index_for_plot = -1
        rmse_for_best_plot_fit = np.nan

        if not results_df_for_plot.empty and not results_df_for_plot['R2'].isna().all():
            best_idx_plot = results_df_for_plot['R2'].idxmax()
            best_result_row_plot = results_df_for_plot.loc[best_idx_plot]
            
            best_R2_for_plot = best_result_row_plot['R2']
            best_fit_params_for_plot = best_result_row_plot['params']
            best_depth_limit_for_plot_annotation = best_result_row_plot['depth_limit']
            best_k_overall_index_for_plot = best_idx_plot
            
            if best_fit_params_for_plot is not None and len(best_fit_params_for_plot) == 2:
                best_fit_range_idx_plot = (y_original >= depth_min_limit_regr) & (y_original <= best_depth_limit_for_plot_annotation)
                x_for_rmse_plot = x_original[best_fit_range_idx_plot]
                y_true_for_rmse_plot = y_original[best_fit_range_idx_plot]
                if len(x_for_rmse_plot) > 1:
                    y_pred_for_rmse_plot = np.polyval(best_fit_params_for_plot, x_for_rmse_plot)
                    rmse_for_best_plot_fit = np.sqrt(mean_squared_error(y_true_for_rmse_plot, y_pred_for_rmse_plot))


        # --- Plotting All Regression Lines (for current file) ---
        if not results_df_for_plot.empty:
            for index, row_data in results_df_for_plot.iterrows():
                if not np.isnan(row_data['R2']) and row_data['params'] is not None and \
                    not np.isnan(row_data['x_min_fit']) and not np.isnan(row_data['x_max_fit']):
                    p_current = row_data['params']
                    current_x_min, current_x_max = row_data['x_min_fit'], row_data['x_max_fit']
                    if current_x_min == current_x_max: x_line_segment = np.array([current_x_min]*2)
                    else: x_line_segment = np.linspace(current_x_min, current_x_max, 20)
                    y_plot_fit_segment = np.polyval(p_current, x_line_segment)
                    ax.plot(x_line_segment, y_plot_fit_segment, color=[0.7, 0.7, 0.7], linewidth=0.5)
        
        best_line_handle = None
        if best_k_overall_index_for_plot != -1 and best_fit_params_for_plot is not None and not results_df_for_plot.empty:
            best_x_min_line = results_df_for_plot.loc[best_k_overall_index_for_plot, 'x_min_fit']
            best_x_max_line = results_df_for_plot.loc[best_k_overall_index_for_plot, 'x_max_fit']
            if not (np.isnan(best_x_min_line) or np.isnan(best_x_max_line)):
                if best_x_min_line == best_x_max_line: x_best_line_segment = np.array([best_x_min_line]*2)
                else: x_best_line_segment = np.linspace(best_x_min_line, best_x_max_line, 20)
                y_best_plot_fit_segment = np.polyval(best_fit_params_for_plot, x_best_line_segment)
                line_label_for_plot = f'Best R² Fit (R²={best_R2_for_plot:.2f}, RMSE={rmse_for_best_plot_fit:.2f})'
                line_plots = ax.plot(x_best_line_segment, y_best_plot_fit_segment, color='r', linewidth=2.5, label=line_label_for_plot)
                if line_plots: best_line_handle = line_plots[0]
        

        # --- Text Annotation for Best Fit (on Plot) ---
        if best_k_overall_index_for_plot != -1 and best_fit_params_for_plot is not None:
            if len(x_for_scatter) > 0: 
                plot_x_min, plot_x_max = np.min(x_for_scatter), np.max(x_for_scatter)
                plot_y_min, plot_y_max = np.min(y_for_scatter), np.max(y_for_scatter)
                text_x_pos = plot_x_min + (plot_x_max - plot_x_min) * 0.05
                text_y_pos = plot_y_max - (plot_y_max - plot_y_min) * 0.05
                if ax.get_yaxis().get_inverted(): text_y_pos = plot_y_min + (plot_y_max - plot_y_min) * 0.05
                
                m_plot_annot, b_plot_annot = best_fit_params_for_plot[0], best_fit_params_for_plot[1]
                eq_plot_annot = f"y = {m_plot_annot:.2f}x + {b_plot_annot:.2f}"
                annotation_text = (f'Overall Best Fit (Plot)\nRange: {depth_min_limit_regr:.2f}-{best_depth_limit_for_plot_annotation:.2f} m\n'
                                   f'{eq_plot_annot}\nR² = {best_R2_for_plot:.2f}, RMSE = {rmse_for_best_plot_fit:.2f}')
                ax.text(text_x_pos, text_y_pos, annotation_text, color='r', fontsize=9, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
                        verticalalignment='top' if not ax.get_yaxis().get_inverted() else 'bottom')
        # --- End Text Annotation ---

        # --- Legend ---
        handles_for_legend = []
        if scatter_handle and hasattr(scatter_handle, 'get_offsets') and len(scatter_handle.get_offsets()) > 0 :
            handles_for_legend.append(scatter_handle)
        if best_line_handle and hasattr(best_line_handle, 'get_xdata') and any(np.isfinite(best_line_handle.get_xdata())): 
            handles_for_legend.append(best_line_handle)
        if handles_for_legend: ax.legend(handles=handles_for_legend, loc='best')
       

        plt.tight_layout()

        plot_filename = f"{just_the_filename_for_output_csv}_plot.png"
        plot_save_path = os.path.join(output_save_folder_path, plot_filename)
        plt.savefig(plot_save_path)
        print(f"Plot for {current_file_name_for_output} saved to: {plot_save_path}")

        plt.show() 
        plt.close(fig) 

        # --- Save Iteration Data for the Current File to its own CSV ---
        if current_file_iterations_data:
            file_summary_df = pd.DataFrame(current_file_iterations_data)
            output_csv_filename = f"{just_the_filename_for_output_csv}_iterations_summary.csv"
            output_csv_save_path = os.path.join(output_save_folder_path, output_csv_filename)
            file_summary_df.to_csv(output_csv_save_path, index=False, float_format='%.4f')
            print(f"Iterations summary for {current_file_name_for_output} saved to: {output_csv_save_path}")
        else:
            print(f"No iteration data to save for {current_file_name_for_output}.")

    except Exception as e_outer:
        print(f"An unexpected error occurred while processing file {data_name_full_path}: {e_outer}")
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        # If an error occurs for a file, its specific CSV won't be saved if the error is before that point.
        continue

print("\n--- All CSV files processed. ---")





###################################################################################################################################################
##################################################################################################################################################

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


def create_sdb_rasters(raster_folder, csv_folder, output_folder, nodata_value=-9999):
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
                    src_nodata = src.nodata
                    
                    # Find the row in the CSV where the raster name matches
                    coeff_row = coefficients_df[coefficients_df['Image Name'].str.contains(base_raster_name, 
                                                                                           case=False, na=False)]
                    if not coeff_row.empty:
                        # Extract coefficients
                        m1 = coeff_row['m1'].values[0]
                        m0 = coeff_row['m0'].values[0]

                        pSDB_f = pSDB.astype(np.float32)
                        
                        # Convert source nodata to NaN if it exists
                        if src_nodata is not None:
                            pSDB_f[pSDB_f == src_nodata] = np.nan

                        # Perform the SDB raster calculation
                        result = m1 * pSDB + m0

                        # Replace NaNs and original nodata values with output nodata_value
                        result_filled = np.where(
                            np.isnan(pSDB) | np.isnan(result) | (result < 0),
                            nodata_value,
                            result
                        )


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
csv_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results"
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
    - If 0 <= SDBred <= 2, use SDBred.
    - If 2 < SDBred <= 3.5, use a weighted average of SDBred and SDBgreen.
    - If SDBred > 3.5, use SDBgreen.
    """
    # Initialize output array
    sdb_merged = np.full_like(sdb_red, np.nan)

    lower_limit = 2 
    upper_limit = 3.5

    # Handle NaNs in input data
    sdb_red = np.nan_to_num(sdb_red, nan=np.nan)  # Changed: Ensure NaNs are handled explicitly
    sdb_green = np.nan_to_num(sdb_green, nan=np.nan)  # Changed: Ensure NaNs are handled explicitly


    # Debug: Check input min/max values
    print(f"SDB Red Min: {np.nanmin(sdb_red)}, max: {np.nanmax(sdb_red)}")
    print(f"SDB Green Min: {np.nanmin(sdb_green)}, max: {np.nanmax(sdb_green)}")

    # Condition 1: Use SDBred if 0 <= SDBred <= 2
    red_condition = sdb_red <= lower_limit
    print(f"Red Condition Count: {np.sum(red_condition)}")
    sdb_merged[red_condition] = sdb_red[red_condition]

    # Condition 2: Weighted average if 2 < SDBred <= 3.5
    weighted_condition = (sdb_red > lower_limit) & (sdb_red <= upper_limit) & (sdb_red < sdb_green)    
    print(f"Weighted Condition Count: {np.sum(weighted_condition)}")




    # Calculate weights (linear relationship)
    alpha = (sdb_red - lower_limit) / (upper_limit - lower_limit)  # Correct weight calculation
    beta = 1 - alpha
    print(f"Alpha Min: {np.nanmin(alpha)}, max: {np.nanmax(alpha)}")
    print(f"Beta Min: {np.nanmin(beta)}, max: {np.nanmax(beta)}")





    sdb_weighted = alpha * sdb_red + beta * sdb_green
    sdb_merged[weighted_condition] = sdb_weighted[weighted_condition]

    # Condition 3: Use SDBgreen if SDBred > 3.5
    green_condition = (sdb_red >= lower_limit) & (sdb_green > upper_limit) | (sdb_red >= lower_limit) & (sdb_green <= upper_limit) & (sdb_red >=sdb_green)
    
    
    print(f"Green Condition Count: {np.sum(green_condition)}")
    sdb_merged[green_condition] = sdb_green[green_condition]

    # Debug: Check output min/max
    print(f"SDB Merged Min: {np.nanmin(sdb_merged)}, max: {np.nanmax(sdb_merged)}")

    return sdb_merged



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
        # Replace np.nan with NoData value (e.g., -9999)
        nodata_value = -9999
        sdb_merged_filled = np.where(np.isnan(sdb_merged), nodata_value, sdb_merged).astype(np.float32)
        
        # Save the new merged raster
        with rasterio.open(output_path, 'w', driver='GTiff', count=1, dtype=sdb_merged.dtype,
                            height=sdb_merged.shape[0], width=sdb_merged.shape[1],
                            crs=red_dataset.crs, transform=red_dataset.transform, nodata=nodata_value) as dst:
            dst.write(sdb_merged, 1)

        print(f"Saved merged SDB raster: {output_path}")

# Example usage
input_folder = r"E:\Thesis Stuff\SDB"  # Folder with input rasters

process_sdb_folder(input_folder)




"""Processes all matching SDBred and SDBgreen rasters in a folder, merging only if R² ≥ threshold."""
# def process_sdb_folder(input_folder, csv_folder, r2_threshold=0.7):


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

def extract_raster_values(val_csv_file, raster_folder, output_folder):
    """
    Extracts raster values at the locations provided in the CSV file and saves the results.

    Args:
        val_csv_file (str): Path to the CSV file containing latitude, longitude, and elevation.
        raster_folder (str): Path to the folder containing GeoTIFF raster files.
        output_folder (str): Path to the folder where the results will be saved.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the CSV file
    df = pd.read_csv(val_csv_file)
    
    df.columns = ['Easting(m)', 'Northing(m)', 'Geoid_Corrected_Ortho_Height']
    
   
    # Multiply the 'Geoid_Corrected_Ortho_Height' column by -1 DON"T DO IT IF ALREADY POSITIVE
    print("Multiplying 'Geoid_Corrected_Ortho_Height' column by -1.")
    # Ensure the column is numeric before multiplying
    height_col_name = 'Geoid_Corrected_Ortho_Height'
    df[height_col_name] = pd.to_numeric(df[height_col_name], errors='coerce') * -1
   
    
    
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

val_csv_file = r"B:\Thesis Project\Reference Data\Processed_Topobathy\Marathon_validation_points.csv"
raster_folder = r"E:\Thesis Stuff\SDB"
output_folder = r"E:\Thesis Stuff\SDB_ExtractedPts"

extract_raster_values(val_csv_file, raster_folder, output_folder)



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
            y = data['Geoid_Corrected_Ortho_Height'].values
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
            
            min_x = np.min(x)
            mean_y = np.mean(y)

            # Compute the x-intercept (where y = 0)
            x_intercept = -intercept / coef if coef != 0 else np.min(x)

            # Generate extended x values from x_intercept to max x
            x_ext = np.linspace(x_intercept, np.max(x), 100)

            # Compute corresponding y values
            y_ext = coef * x_ext + intercept

            # Regression Plot
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, color='blue', label='Data Points', alpha=0.7)
            plt.plot(x_ext, y_ext, color='red', label='Best Fit Line', linewidth=2)
            plt.title(f"Linear Regression for {filename}")
            plt.xlabel("SDB Value (m)")
            plt.ylabel("Reference Depths (m)")
            plt.legend()
            plt.grid(True)    
            plt.xlim(0, None)
            plt.ylim(0, None)
            
            # Add R^2 and RMSE as text on the plot
            plt.text(min_x, mean_y, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nIntercept = {intercept:0.2f}", fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.5), ha='left')
            
            # Invert both ayes so 0 is bottom left, and up and right are negative
            #plt.gca().invert_xaxis()
            #plt.gca().invert_yaxis()

            # Save the regression plot in the output folder
            plot_filename = f"{os.path.splitext(filename)[0]}_LR_plot.png"
            plot_path = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_path)
            #print(f"Regression plot saved as {plot_path}")
            plt.close()


            max_x_perp = np.max(data["Raster_Value"])
            min_y_perp = np.min(distances)

            
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









































