## -*- coding: utf-8 -*-
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

from glob import glob
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from difflib import get_close_matches
from rasterio.sample import sample_gen
from shapely.geometry import box

##############################################################################
##############################################################################

"""Composite R, G, B tiffs into one RGB tiff"""

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
            #input_folder_name = Path(input_folder).name
            output_file_name = re.sub(r"Rrs_\d+", "", r.name).replace("__", "_")  # Clean up any double underscores
            output_file = os.path.join(output_folder, f"{Path(output_file_name).stem}RGB.tif")

            # Write combined RGB TIFF
            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(red_data, 1)   # Red band
                dst.write(green_data, 2) # Green band
                dst.write(blue_data, 3)  # Blue band

            print(f"Combined RGB saved to: {output_file}")

        except Exception as e:
            print(f"Error processing {r}, {g}, {b}: {e}")


################# CHECK DIRECTORIES/INPUTS #####################
#!!!
input_folder = r"E:\Thesis Stuff\AcoliteWithPython\Corrected_Imagery\All_SuperDove\SD_Anegada_output"
output_folder = r"E:\Thesis Stuff\RGBCompositOutput"


combine_bands_to_rgb(input_folder, output_folder)


##############################################################################
##############################################################################

"""First Optically Deep Finder (ODF) where blue/green values < 0.003 sr^-1 are omitted"""


def mask_optically_deep_water(input_rgb_folder, 
                              output_masked_folder, # Folder for the processed RGBs
                              output_binary_mask_folder=None, 
                              threshold=0.003):
    """
    Processes RGB GeoTIFF files to mask optically deep water pixels.
    Pixels where either the green OR blue band value is <= threshold are set to NaN.
    Saves the masked RGB to a NEW FILE in output_masked_folder.
    Optionally saves a separate binary mask of the changed pixels.

    Args:
        input_rgb_folder (str): Path to the folder containing RGB GeoTIFF files.
        output_masked_folder (str): Path to save the processed GeoTIFFs with ODW masked.
        output_binary_mask_folder (str, optional): Path to save binary masks of ODW areas.
                                                    If None, binary masks are not saved.
        threshold (float): Reflectance threshold to identify ODW pixels.
    """
    if not os.path.exists(output_masked_folder):
        os.makedirs(output_masked_folder)
        print(f"Created masked output folder: {output_masked_folder}")

    if output_binary_mask_folder and not os.path.exists(output_binary_mask_folder):
        os.makedirs(output_binary_mask_folder)
        print(f"Created binary mask output folder: {output_binary_mask_folder}")

    for file_name in os.listdir(input_rgb_folder):
        if file_name.endswith('.tif') and "_RGB" in file_name: # Process only the RGB composites
            input_file_path = os.path.join(input_rgb_folder, file_name) # Changed variable name for clarity
            print(f"Masking ODW for file: {input_file_path}")

            try:
                with rasterio.open(input_file_path) as src: # Use input_file_path
                    profile = src.profile
                    red_band = src.read(1).astype(rasterio.float32)
                    green_band = src.read(2).astype(rasterio.float32)
                    blue_band = src.read(3).astype(rasterio.float32)

                    nodata_value = src.nodata
                    if nodata_value is None and profile.get('nodata') is np.nan:
                        nodata_value = np.nan
                    
                    red_masked = red_band.copy()
                    green_masked = green_band.copy()
                    blue_masked = blue_band.copy()

                    odw_condition = np.logical_or(green_band <= threshold, blue_band <= threshold)
                    
                    if nodata_value is not None and not np.isnan(nodata_value):
                        existing_nodata_mask_g = (green_band == nodata_value)
                        existing_nodata_mask_b = (blue_band == nodata_value)
                        odw_condition = np.logical_and(odw_condition, ~existing_nodata_mask_g)
                        odw_condition = np.logical_and(odw_condition, ~existing_nodata_mask_b)
                    elif np.isnan(nodata_value): 
                        existing_nodata_mask_g = np.isnan(green_band)
                        existing_nodata_mask_b = np.isnan(blue_band)
                        odw_condition = np.logical_and(odw_condition, ~existing_nodata_mask_g)
                        odw_condition = np.logical_and(odw_condition, ~existing_nodata_mask_b)

                    red_masked[odw_condition] = np.nan
                    green_masked[odw_condition] = np.nan
                    blue_masked[odw_condition] = np.nan
                    
                    profile.update(nodata=np.nan, dtype=rasterio.float32, count=3)

                    # adds file extention
                    original_filename_stem = os.path.splitext(file_name)[0]
                    output_masked_filename = f"{original_filename_stem}_m1.tif"
                    output_masked_file_path = os.path.join(output_masked_folder, output_masked_filename)

                    with rasterio.open(output_masked_file_path, 'w', **profile) as dst: # Use output_masked_file_path
                        dst.write(red_masked, 1)
                        dst.write(green_masked, 2)
                        dst.write(blue_masked, 3)
                    print(f"Saved ODW masked RGB to: {output_masked_file_path}")

                    if output_binary_mask_folder:
                        # ... (binary mask saving logic as before) ...
                        binary_odw_mask = np.zeros_like(red_band, dtype=rasterio.uint8)
                        binary_odw_mask[odw_condition] = 1
                        
                        mask_nodata_val = 255 # Default for uint8 mask nodata
                        if nodata_value is not None and not np.isnan(nodata_value):
                            original_nodata_combined = np.logical_or(red_band == nodata_value, 
                                                                      green_band == nodata_value, 
                                                                      blue_band == nodata_value)
                            binary_odw_mask[original_nodata_combined] = mask_nodata_val
                        elif np.isnan(nodata_value):
                            original_nodata_combined = np.logical_or(np.isnan(red_band), 
                                                                      np.isnan(green_band), 
                                                                      np.isnan(blue_band))
                            binary_odw_mask[original_nodata_combined] = mask_nodata_val
                            
                        mask_profile = profile.copy()
                        mask_profile.update(count=1, dtype=rasterio.uint8, nodata=mask_nodata_val)

                        output_binary_mask_file = os.path.join(output_binary_mask_folder, f"{original_filename_stem}_Mask.tif")
                        with rasterio.open(output_binary_mask_file, 'w', **mask_profile) as dst_mask:
                            dst_mask.write(binary_odw_mask, 1)
                        print(f"Saved ODW binary mask to: {output_binary_mask_file}")

            except Exception as e:
                print(f"Error processing ODW for {file_name}: {e}")

    print("\nFunction mask_optically_deep_water finished.")



################# CHECK DIRECTORIES/INPUTS #####################
#!!!
raw_bands_input_folder = r"E:\Thesis Stuff\AcoliteWithPython\Corrected_Imagery\All_SuperDove\SD_Anegada_output"
rgb_and_masked_folder = r"E:\Thesis Stuff\RGBCompositOutput"

# Path for optional binary masks
output_for_binary_masks = r"E:\Thesis Stuff\RGBCompositOutput_ODWbinarymasks" # Or set to None

combine_bands_to_rgb(raw_bands_input_folder, rgb_and_masked_folder) 

print("Masking Optically Deep Water from RGB composites")
mask_optically_deep_water(
    input_rgb_folder=rgb_and_masked_folder,        
    output_masked_folder=rgb_and_masked_folder,     
    output_binary_mask_folder=output_for_binary_masks,
    
    # Current sr^-1 threshold
    threshold=0.003) 


##############################################################################
##############################################################################

"""Create pSDB red and green"""

" !!! IF you want to keep the RGB outputs, change delete_input_files to FALSE !!!"


def process_rgb_geotiffs(input_folder, output_folder, delete_input_files=False):
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
        if file_name.lower().endswith('m1.tif'):
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
                scaled_red = red_band * 1000 * 3.14159     # Original 100000
                scaled_green = green_band * 1000 * 3.14159  # Original 100000
                scaled_blue = blue_band * 1000 * 3.14159    # Original 100000

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

                # Remove any negative pSDB values
                pSDBred[pSDBred < 0] = np.nan
                pSDBgreen[pSDBgreen < 0] = np.nan

                # Update profile for the output GeoTIFF
                profile.update(
                    dtype=rasterio.float32,
                    count=1,
                    nodata=np.nan
                )

                # Save the computed index to the output folder                
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


     # Optional deletion block
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

    print("\nFunction process_rgb_geotiffs finished.")



      #################  CHECK DIRECTORIES/INPUTS #####################

input_folder = r"E:\Thesis Stuff\RGBCompositOutput"

# Save Results Path
#output_folder = r"B:\Thesis Project\SDB_Time\Results_main\Homer\SuperDove\pSDB"

# Workspace Path
output_folder = r"E:\Thesis Stuff\pSDB"

process_rgb_geotiffs(input_folder, output_folder)



##############################################################################
##############################################################################
"""Extract pSDB values at reference point locations"""



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



def extract_raster_values_optimized(cal_csv_file, raster_folder, output_folder):
    """
    Extracts raster values at specified locations using optimized vectorized operations.

    Args:
        cal_csv_file (str): Path to the CSV file with point coordinates.
        raster_folder (str): Path to the folder containing GeoTIFF raster files.
        output_folder (str): Path to the folder for saving results.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load and Prepare Point Data (Done Once)
    print("Loading and preparing reference points...")
    df = pd.read_csv(cal_csv_file)
    
    # Assuming first col is Easting, second is Northing, third is Elevation
    easting_col, northing_col, elev_col = df.columns[0], df.columns[1], df.columns[2]

    # Convert elevation to depth and filter out invalid points
    df[elev_col] = pd.to_numeric(df[elev_col], errors='coerce') * -1
    original_rows = len(df)
    df.dropna(subset=[elev_col], inplace=True)
    df = df[df[elev_col] >= 1] # Keep only valid depths
    rows_removed = original_rows - len(df)
    if rows_removed > 0:
        print(f"Removed {rows_removed} rows with invalid or negative depth values.")

    # Create a GeoDataFrame from the points
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[easting_col], df[northing_col])
    )

    # Process Each Raster File 
    for raster_file in os.listdir(raster_folder):
        if raster_file.lower().endswith(".tif"):
            raster_path = os.path.join(raster_folder, raster_file)
            print(f"\nProcessing raster: {raster_file}")

            with rasterio.open(raster_path) as src:
                # Set the CRS of the points to match the raster's CRS
                # This is crucial for accurate alignment!
                if gdf.crs != src.crs:
                    print(f"Assigning raster CRS ({src.crs}) to points.")
                    gdf.set_crs(src.crs, inplace=True)

                # Fast Filtering with a Spatial Index 
                # Create a bounding box from the raster's extent
                raster_bbox = box(*src.bounds)
                
                # Use a spatial clip, which is much faster than .apply()
                gdf_in_bounds = gdf.clip(raster_bbox)

                if gdf_in_bounds.empty:
                    print(f"No points overlap with raster: {raster_file}")
                    continue
                
                print(f"Found {len(gdf_in_bounds)} points within the raster bounds.")

                # Vectorized Value Extraction ---
                # Get the coordinates of the valid points for sampling
                coords = [(p.x, p.y) for p in gdf_in_bounds.geometry]

                # Use rasterio.sample.sample_gen for efficient, bulk extraction
                # It returns a generator, so we convert it to a list
                raster_values = [val[0] for val in src.sample(coords)]

                # Add the extracted values to the GeoDataFrame
                gdf_in_bounds = gdf_in_bounds.copy()
                gdf_in_bounds.loc[:, 'Raster_Value'] = raster_values
                
                # --- 3. Clean and Save Results ---
                # Remove points where the raster has NoData values
                gdf_in_bounds.dropna(subset=['Raster_Value'], inplace=True)

                if gdf_in_bounds.empty:
                    print(f"No valid raster values found for raster: {raster_file}")
                    continue

                # Save the results to a new CSV
                output_filename = os.path.splitext(raster_file)[0] + "_extracted.csv"
                output_path = os.path.join(output_folder, output_filename)
                gdf_in_bounds.to_csv(output_path, index=False)
                print(f"Results saved to: {output_path}")

# #################  CHECK DIRECTORIES/INPUTS #####################
#!!!
cal_csv_file = r"B:\Thesis Project\Reference Data\Processed_ICESat\Anegada_cal.csv"

raster_folder = r"E:\Thesis Stuff\pSDB"

output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts"

# Run the optimized function
extract_raster_values_optimized(cal_csv_file, raster_folder, output_folder)


##############################################################################
##############################################################################

""" Perform linear regressions between pSDB red and green and reference calibration data """


# def process_csv_files(input_folder, output_folder):
#     """
#     Processes CSV files in the input folder, performs linear regression, 
#     and saves the results in the output folder.
#     """
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)


#     # Loop through all CSV files in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".csv"):
#             input_path = os.path.join(input_folder, filename)
#             print(f"Processing {filename}...")

#             # Read the CSV file
#             data = pd.read_csv(input_path)

#             # Drop rows where 'Raster_Value' is blank
#             data = data.dropna(subset=['Raster_Value'])

#             # Initialize a results list
#             results = []

#             # Perform linear regression
#             x = data[['Raster_Value']].values
#             y = data['Geoid_Corrected_Ortho_Height'].values
#             model = LinearRegression()
#             model.fit(x, y)

#             # Calculate regression metrics
#             y_pred = model.predict(x)
#             r2 = r2_score(y, y_pred)
#             rmse = np.sqrt(mean_squared_error(y, y_pred))
#             coef = model.coef_[0]
#             intercept = model.intercept_

#             # Create the line of best fit equation
#             equation = f"y = {coef:.4f}x + {intercept:.4f}"

#             # Append the results to the list
#             results.append({
#                 "Image Name": filename,
#                 "R2 Value": r2,
#                 "RMSE": rmse,        
#                 "Line of Best Fit": equation,
#                 "m1": coef,
#                 "m0": intercept
#             })

#             min_x = np.min(x)
#             mean_y = np.mean(y)
            
#             # Plot the data and the regression line
#             plt.figure(figsize=(8, 6))
#             plt.scatter(x, y, color='blue', label='Data Points', alpha=0.7)
#             plt.plot(x, y_pred, color='red', label='Best Fit Line', linewidth=2)
#             plt.title(f"Linear Regression for {filename}")
#             plt.xlabel("pSDB Values (unitless)")
#             plt.ylabel("Reference Depths (m)")
#             #plt.xlim(None, 1.5)
#             plt.ylim(0, None)
#             plt.legend()
#             plt.grid(True)    
            
#             # Add R^2 and RMSE as text on the plot
#             plt.text(min_x, mean_y, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}", fontsize=10, 
#                       bbox=dict(facecolor='white', alpha=0.5), ha='left')
#             #plt.show

#             # Generate and save the plot
#             plot_filename = f"{os.path.splitext(filename)[0]}_LR_plot.png"
#             plot_path = os.path.join(output_folder, plot_filename)
#             plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#             #plt.close()  # Close the plot to free memory

#             print(f"Plot saved to {plot_path}")


#             # Convert the results into a DataFrame
#             results_df = pd.DataFrame(results)

#             # Generate the output file name
#             output_filename = f"{os.path.splitext(filename)[0]}_LR_stats.csv"
#             output_path = os.path.join(output_folder, output_filename)


#             # Save the results to the output CSV file
#             results_df.to_csv(output_path, index=False)
#             print(f"Results saved to {output_path}")


#       #################  CHECK DIRECTORIES/INPUTS #####################

# ### Save Results Path ###
# #input_folder = r"B:\Thesis Project\SDB_Time\Results_main\Homer\SuperDove\Extracted Pts\pSDB"

# ### Workspace Path ###
# input_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts"


# ### Save Results Path ###
# #output_folder = r"B:\Thesis Project\SDB_Time\Results_main\Homer\SuperDove\Figures\pSDB"

# ### Workspace Path ###
# output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_Results"


# process_csv_files(input_folder, output_folder)


##############################################################################################################
##############################################################################################################


"""Better Optically Deep finder: Perform another linear regression but this time find the best line of fit 
   with the highest R^2 value """


### Save Results Path ###
#data_folder_path = r"B:\Thesis Project\SDB_Time\Results_main\Homer\SuperDove\Extracted Pts\pSDB"

### Workspace Path ###
data_folder_path = r"E:\Thesis Stuff\pSDB_ExtractedPts" 


### Save Results Path ###
#output_save_folder_path = r"B:\Thesis Project\SDB_Time\Results_main\Homer\SuperDove\Figures\pSDB_maxR2"

### Workspace Path ###
output_save_folder_path = r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results"



# Create the output folder if it doesn't exist
if not os.path.exists(output_save_folder_path):
    os.makedirs(output_save_folder_path)
    print(f"Created output folder: {output_save_folder_path}")

csv_files = glob(os.path.join(data_folder_path, "*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in the specified folder: {data_folder_path}")

print(f"Found {len(csv_files)} CSV files to process in: {data_folder_path}")

for data_name_full_path in csv_files:
    print(f"\n--- Processing file: {data_name_full_path} ---")
    current_file_name_for_output = os.path.basename(data_name_full_path)
    just_the_filename_for_output_csv = os.path.splitext(current_file_name_for_output)[0]

    fig = None
    current_file_iterations_data = []

    try:
        # Load Data
        if not os.path.isfile(data_name_full_path):
            print(f'Warning: Data CSV file not found: {data_name_full_path}')
            continue
        try:
            data_df = pd.read_csv(data_name_full_path)
            if data_df.shape[1] < 5:
                raise ValueError('CSV file does not have enough columns. Expecting at least 5.')

            y_original = data_df["Geoid_Corrected_Ortho_Height"].values.astype(float)
            x_original = data_df["Raster_Value"].values.astype(float)

            nan_mask = ~ (np.isnan(x_original) | np.isnan(y_original))
            x_original = x_original[nan_mask]
            y_original = y_original[nan_mask]

            if x_original.size == 0 or y_original.size == 0:
                raise ValueError('No valid data points after removing NaNs from x or y.')

            valid_x_idx = (x_original >= 0) & (x_original <= 5)
            x_original = x_original[valid_x_idx]
            y_original = y_original[valid_x_idx]

            if x_original.size == 0 or y_original.size == 0:
                raise ValueError('No data points remain after filtering x values between 0 and 5.')

        except Exception as e:
            print(f'Failed to read or process input CSV file {data_name_full_path}. Error: {e}')
            if fig and plt.fignum_exists(fig.number): plt.close(fig)
            continue

        data_name_for_conditions = str(just_the_filename_for_output_csv).lower()

        # Initial Plot Setup
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
        ax.set_title(f'pSDB Regression: {just_the_filename_for_output_csv.replace("_", " ")}') # Slightly shorter title
        ax.grid(True)

        iteration_results_for_plot_obj = []

        # Define the two sets of depth_min_limit_regr ###
        depth_min_regr_sets = [1.0, 1.25, 1.5, 1.75, 2.0]
        step = 0.25 # Common step

        if y_original.size > 0:
            overall_max_depth_data = np.max(y_original) # Max depth from this specific file
            print(f"DEBUG: Dynamically set overall_max_depth_data to: {overall_max_depth_data:.2f} m")
        else:
            print("Warning: y_original is empty for regression. Skipping iterations.")
            # Save empty plot if needed and continue
            if fig: plt.savefig(os.path.join(output_save_folder_path, f"{just_the_filename_for_output_csv}_plot_no_data_for_regr.png")); plt.close(fig)
            continue

        # Loop through each depth_min_limit_regr set ###
        for depth_min_limit_regr in depth_min_regr_sets:
            print(f"\n--- Iterating with Min Depth Range starting at: {depth_min_limit_regr:.2f} m ---")

            # Determine initial_depth_max based on the current depth_min_limit_regr
            initial_depth_max = depth_min_limit_regr + step # Start one step above the min
            if initial_depth_max < depth_min_limit_regr: # handles if step is 0 or negative, though unlikely
                initial_depth_max = depth_min_limit_regr

            # Cap initial_depth_max at overall_max_depth_data
            initial_depth_max = min(initial_depth_max, overall_max_depth_data)

            if initial_depth_max >= overall_max_depth_data:
                if overall_max_depth_data >= depth_min_limit_regr:
                    depth_max_limits_to_test = np.array([overall_max_depth_data])
                else:
                    depth_max_limits_to_test = np.array([])
            else:
                start_arange = max(initial_depth_max, depth_min_limit_regr) # Ensure arange starts correctly
                if start_arange >= overall_max_depth_data:
                    if overall_max_depth_data >= depth_min_limit_regr:
                        depth_max_limits_to_test = np.array([overall_max_depth_data])
                    else:
                        depth_max_limits_to_test = np.array([])
                else:
                    depth_max_limits_to_test = np.arange(start_arange, overall_max_depth_data + step, step)
                    if not np.isclose(depth_max_limits_to_test[-1], overall_max_depth_data) and \
                       depth_max_limits_to_test[-1] < overall_max_depth_data and \
                       overall_max_depth_data >= depth_min_limit_regr:
                        depth_max_limits_to_test = np.append(depth_max_limits_to_test, overall_max_depth_data)
            
            # Ensure all test limits are valid for the current depth_min_limit_regr
            if len(depth_max_limits_to_test) > 0:
                depth_max_limits_to_test = depth_max_limits_to_test[depth_max_limits_to_test >= depth_min_limit_regr]
                depth_max_limits_to_test = np.unique(depth_max_limits_to_test) # Remove duplicates and sort

            if len(depth_max_limits_to_test) == 0:
                print(f'Warning: No valid depth ranges for Min Depth {depth_min_limit_regr:.2f}m. Skipping this set.')
                continue

            print(f'Calculating regression for {len(depth_max_limits_to_test)} depth ranges (Min Depth: {depth_min_limit_regr:.2f}m) for {current_file_name_for_output}...')
            for k_loop_idx, current_depth_max_limit in enumerate(depth_max_limits_to_test):
                plot_result_entry = {'depth_limit': current_depth_max_limit, 'R2': np.nan, 'params': None,
                                     'point_count': 0, 'x_min_fit': np.nan, 'x_max_fit': np.nan, 'slope_m': np.nan,
                                     'original_k_index': k_loop_idx, # You can use this or a combined index later
                                     'min_depth_regr_setting': depth_min_limit_regr} # Store which min_depth this iteration belongs to

                m_for_iteration, b_for_iteration, R2_for_iteration, rmse_for_iteration = np.nan, np.nan, np.nan, np.nan
                equation_for_iteration = "N/A"
                # IMPORTANT: range_idx now uses the current depth_min_limit_regr from the outer loop
                range_idx = (y_original >= depth_min_limit_regr) & (y_original <= current_depth_max_limit)
                x_iter_range, y_iter_range = x_original[range_idx], y_original[range_idx]
                num_points = len(x_iter_range)
                plot_result_entry['point_count'] = num_points

                if num_points > 1:
                    params = np.polyfit(x_iter_range, y_iter_range, 1)
                    y_fit_iter_range = np.polyval(params, x_iter_range)
                    m_for_iteration, b_for_iteration = params[0], params[1]
                    plot_result_entry['slope_m'] = m_for_iteration
                    if len(np.unique(y_iter_range)) > 1:
                        R2_for_iteration = r2_score(y_iter_range, y_fit_iter_range)
                        if R2_for_iteration < 0: R2_for_iteration = 0.0
                    elif len(x_iter_range) > 0:
                        R2_for_iteration = 1.0 if np.allclose(y_iter_range, y_fit_iter_range) else 0.0
                    
                    plot_result_entry['R2'] = R2_for_iteration
                    plot_result_entry['params'] = params
                    if len(x_iter_range) > 0:
                        plot_result_entry['x_min_fit'], plot_result_entry['x_max_fit'] = np.min(x_iter_range), np.max(x_iter_range)
                    equation_for_iteration = f"y = {m_for_iteration:.4f}x + {b_for_iteration:.4f}"
                    rmse_for_iteration = np.sqrt(mean_squared_error(y_iter_range, y_fit_iter_range))
                
                iteration_results_for_plot_obj.append(plot_result_entry) # Appends to the list for the plot object
                
                # Append to the CSV data list, including the Min Depth Range setting
                current_file_iterations_data.append({
                    'Image Name': current_file_name_for_output,
                    'Min Depth Range': depth_min_limit_regr, # This now reflects 1.0 or 2.0
                    'Max Depth Range': current_depth_max_limit,
                    'R2 Value': R2_for_iteration, 'RMSE': rmse_for_iteration,
                    'Line of Best Fit': equation_for_iteration, 'm1': m_for_iteration, 'm0': b_for_iteration,
                    'Pt Count': num_points,
                    'Indicator': 0 # Initialize Indicator to 0, will update later
                })
        
        results_df_for_plot = pd.DataFrame(iteration_results_for_plot_obj)
        # Ensure sorted by depth_limit for the subsequent logic, though arange should provide this.
        results_df_for_plot = results_df_for_plot.sort_values(by='depth_limit').reset_index(drop=True)


        peak_R2_iteration_details = None
        peak_R2_value = -np.inf
        rmse_for_peak_R2_fit = np.nan

        deepest_tolerable_iteration_details = None
        rmse_for_deepest_tolerable_fit = np.nan
        R2_threshold = -np.inf

        if not results_df_for_plot.empty:
            positive_slope_results_df = results_df_for_plot[results_df_for_plot['slope_m'] > 1e-9].copy()

            if not positive_slope_results_df.empty and not positive_slope_results_df['R2'].isna().all():
                # 1. Find Peak R² Line
                peak_idx_in_positive_df = positive_slope_results_df['R2'].idxmax()
                peak_R2_iteration_details = positive_slope_results_df.loc[peak_idx_in_positive_df]
                peak_R2_value = peak_R2_iteration_details['R2']

                if peak_R2_iteration_details['params'] is not None and len(peak_R2_iteration_details['params']) == 2:
                    fit_range_idx = (y_original >= depth_min_limit_regr) & (y_original <= peak_R2_iteration_details['depth_limit'])
                    if np.sum(fit_range_idx) > 1:
                        y_pred = np.polyval(peak_R2_iteration_details['params'], x_original[fit_range_idx])
                        rmse_for_peak_R2_fit = np.sqrt(mean_squared_error(y_original[fit_range_idx], y_pred))

                # Find Deepest Tolerable R² Line
                R2_threshold = peak_R2_value * 0.90
                
                # Iterate through positive slope results to find the last one meeting the threshold
                for index, row in positive_slope_results_df.iterrows():
                    if pd.notna(row['R2']) and row['R2'] >= R2_threshold:
                        deepest_tolerable_iteration_details = row # Keep updating, last one will be the deepest
                
                if deepest_tolerable_iteration_details is not None and \
                   deepest_tolerable_iteration_details['params'] is not None and \
                   len(deepest_tolerable_iteration_details['params']) == 2:
                    fit_range_idx_deep = (y_original >= depth_min_limit_regr) & (y_original <= deepest_tolerable_iteration_details['depth_limit'])
                    if np.sum(fit_range_idx_deep) > 1:
                        y_pred_deep = np.polyval(deepest_tolerable_iteration_details['params'], x_original[fit_range_idx_deep])
                        rmse_for_deepest_tolerable_fit = np.sqrt(mean_squared_error(y_original[fit_range_idx_deep], y_pred_deep))
            else:
                print(f"No iterations with a positive slope and valid R2 found for file: {current_file_name_for_output}")
        else:
            print(f"No iteration results to process for R2 for file: {current_file_name_for_output}")


        # # --- Plotting All Regression Lines (for current file) ---
        # if not results_df_for_plot.empty:
        #     for index, row_data in results_df_for_plot.iterrows():
        #         if pd.notna(row_data['R2']) and row_data['params'] is not None and \
        #            pd.notna(row_data['x_min_fit']) and pd.notna(row_data['x_max_fit']):
        #             p_current = row_data['params']
        #             current_x_min, current_x_max = row_data['x_min_fit'], row_data['x_max_fit']
        #             if current_x_min == current_x_max:
        #                 if len(x_for_scatter) > 0:
        #                     overall_x_min_scatter, overall_x_max_scatter = np.min(x_for_scatter), np.max(x_for_scatter)
        #                     x_line_segment = np.array([overall_x_min_scatter, overall_x_max_scatter]) if overall_x_min_scatter != overall_x_max_scatter else np.array([current_x_min -0.1, current_x_min + 0.1])
        #                 else: x_line_segment = np.array([current_x_min - 0.1, current_x_min + 0.1])
        #             else: x_line_segment = np.linspace(current_x_min, current_x_max, 20)
        #             y_plot_fit_segment = np.polyval(p_current, x_line_segment)
        #             ax.plot(x_line_segment, y_plot_fit_segment, color=[0.7, 0.7, 0.7], linewidth=0.5)

        peak_R2_line_handle = None
        deepest_tolerable_line_handle = None
        color_deepest_tolerable = 'blue' # Choose a distinct color

        # Plot Peak R² Line (Red)
        if peak_R2_iteration_details is not None and peak_R2_iteration_details['params'] is not None:
            params = peak_R2_iteration_details['params']
            x_min = peak_R2_iteration_details['x_min_fit']
            x_max = peak_R2_iteration_details['x_max_fit']
            if pd.notna(x_min) and pd.notna(x_max):
                if x_min == x_max:
                    if len(x_for_scatter) > 0:
                        overall_x_min_scatter, overall_x_max_scatter = np.min(x_for_scatter), np.max(x_for_scatter)
                        x_segment = np.array([overall_x_min_scatter, overall_x_max_scatter]) if overall_x_min_scatter != overall_x_max_scatter else np.array([x_min -0.1, x_min + 0.1])
                    else: x_segment = np.array([x_min - 0.1, x_min + 0.1])
                else: x_segment = np.linspace(x_min, x_max, 20)
                y_segment = np.polyval(params, x_segment)
                label = f"Peak R² Fit (R²={peak_R2_iteration_details['R2']:.2f}, RMSE={rmse_for_peak_R2_fit:.2f})"
                line_plots = ax.plot(x_segment, y_segment, color='r', linewidth=2.5, label=label)
                if line_plots: peak_R2_line_handle = line_plots[0]

        # Plot Deepest Tolerable R² Line (New Color)
        if deepest_tolerable_iteration_details is not None and deepest_tolerable_iteration_details['params'] is not None:
            # Avoid plotting twice if it's the same as the peak R2 line
            # Check based on original_k_index or a combination of R2 and depth_limit
            is_same_as_peak = False
            if peak_R2_iteration_details is not None:
                 if deepest_tolerable_iteration_details['original_k_index'] == peak_R2_iteration_details['original_k_index'] and \
                    np.isclose(deepest_tolerable_iteration_details['min_depth_regr_setting'], peak_R2_iteration_details['min_depth_regr_setting']):
                    is_same_as_peak = True
            
            if not is_same_as_peak:
                params = deepest_tolerable_iteration_details['params']
                x_min = deepest_tolerable_iteration_details['x_min_fit']
                x_max = deepest_tolerable_iteration_details['x_max_fit']
                if pd.notna(x_min) and pd.notna(x_max):
                    if x_min == x_max:
                        if len(x_for_scatter) > 0:
                            overall_x_min_scatter, overall_x_max_scatter = np.min(x_for_scatter), np.max(x_for_scatter)
                            x_segment = np.array([overall_x_min_scatter, overall_x_max_scatter]) if overall_x_min_scatter != overall_x_max_scatter else np.array([x_min -0.1, x_min + 0.1])
                        else: x_segment = np.array([x_min - 0.1, x_min + 0.1])
                    else: x_segment = np.linspace(x_min, x_max, 20)
                    y_segment = np.polyval(params, x_segment)
                    label = (f"Deepest Tolerable R² "
                             f"(R²={deepest_tolerable_iteration_details['R2']:.2f}, RMSE={rmse_for_deepest_tolerable_fit:.2f})")
                    line_plots = ax.plot(x_segment, y_segment, color=color_deepest_tolerable, linewidth=2.0, linestyle='--', label=label)
                    if line_plots: deepest_tolerable_line_handle = line_plots[0]
            elif peak_R2_line_handle: # If same, update label of peak line to include this info or just let it be
                    current_label = peak_R2_line_handle.get_label()
                    peak_R2_line_handle.set_label(f"{current_label}\n(Also Deepest Tolerable)")
                    print(f"INFO: Peak R2 line is also the Deepest Tolerable R2 line for {current_file_name_for_output}")


        # --- Text Annotation for Highlighted Fits (on Plot) ---
        annotation_y_offset = 0
        if peak_R2_iteration_details is not None and peak_R2_iteration_details['params'] is not None:
            if len(x_for_scatter) > 0:
                plot_x_min_s, plot_x_max_s = np.min(x_for_scatter), np.max(x_for_scatter)
                plot_y_min_s, plot_y_max_s = np.min(y_for_scatter), np.max(y_for_scatter)
                text_x_pos = plot_x_min_s + (plot_x_max_s - plot_x_min_s) * 0.05
                
                # Determine y position based on axis inversion
                if ax.get_yaxis().get_inverted():
                    text_y_base = plot_y_min_s + (plot_y_max_s - plot_y_min_s) * 0.05
                    vertical_align = 'bottom'
                    y_step = (plot_y_max_s - plot_y_min_s) * 0.15 # Step for multiple annotations
                else:
                    text_y_base = plot_y_max_s - (plot_y_max_s - plot_y_min_s) * 0.05
                    vertical_align = 'top'
                    y_step = -(plot_y_max_s - plot_y_min_s) * 0.15 # Step for multiple annotations

                m_p, b_p = peak_R2_iteration_details['params'][0], peak_R2_iteration_details['params'][1]
                eq_p = f"y = {m_p:.2f}x + {b_p:.2f}"
                ann_text_peak = (f"Peak R² Fit\n"
                                 f"Range: {peak_R2_iteration_details['min_depth_regr_setting']:.2f}-{peak_R2_iteration_details['depth_limit']:.2f} m\n"
                                 f"{eq_p}\nR² = {peak_R2_iteration_details['R2']:.2f}, RMSE = {rmse_for_peak_R2_fit:.2f}")
                ax.text(text_x_pos, text_y_base + annotation_y_offset, ann_text_peak, color='r', fontsize=8, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3', alpha=0.8),
                        verticalalignment=vertical_align)
                annotation_y_offset += y_step

        if deepest_tolerable_iteration_details is not None and \
           deepest_tolerable_iteration_details['params'] is not None and \
           (peak_R2_iteration_details is None or \
            deepest_tolerable_iteration_details['original_k_index'] != peak_R2_iteration_details['original_k_index'] or \
            not np.isclose(deepest_tolerable_iteration_details['min_depth_regr_setting'], peak_R2_iteration_details['min_depth_regr_setting'])): # Only if different
            if len(x_for_scatter) > 0: # x_for_scatter should exist if peak_R2 was found
                m_d, b_d = deepest_tolerable_iteration_details['params'][0], deepest_tolerable_iteration_details['params'][1]
                eq_d = f"y = {m_d:.2f}x + {b_d:.2f}"
                ann_text_deep = (f"Deepest Tolerable R² Fit (within 5% of Peak)\n"
                                 f"Range: {deepest_tolerable_iteration_details['min_depth_regr_setting']:.2f}-{deepest_tolerable_iteration_details['depth_limit']:.2f} m\n"
                                 f"{eq_d}\nR² = {deepest_tolerable_iteration_details['R2']:.2f}, RMSE = {rmse_for_deepest_tolerable_fit:.2f}")
                ax.text(text_x_pos, text_y_base + annotation_y_offset, ann_text_deep, color=color_deepest_tolerable, fontsize=8, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor=color_deepest_tolerable, boxstyle='round,pad=0.3', alpha=0.8),
                        verticalalignment=vertical_align)


        # --- Legend ---
        handles_for_legend = []
        if scatter_handle and hasattr(scatter_handle, 'get_offsets') and len(scatter_handle.get_offsets()) > 0 :
            handles_for_legend.append(scatter_handle)
        if peak_R2_line_handle: # Add only if it was plotted
            handles_for_legend.append(peak_R2_line_handle)
        if deepest_tolerable_line_handle: # Add only if it was plotted and is distinct
             # Check if it's truly a different handle than peak_R2_line_handle
            if peak_R2_line_handle is None or (deepest_tolerable_line_handle != peak_R2_line_handle):
                handles_for_legend.append(deepest_tolerable_line_handle)
            elif peak_R2_line_handle and (deepest_tolerable_iteration_details is not None and peak_R2_iteration_details is not None and \
                 deepest_tolerable_iteration_details['original_k_index'] == peak_R2_iteration_details['original_k_index'] and \
                 np.isclose(deepest_tolerable_iteration_details['min_depth_regr_setting'], peak_R2_iteration_details['min_depth_regr_setting'])):
                 # If they are the same and label was updated, we don't need a duplicate handle
                 pass


        if handles_for_legend:
            # Remove duplicate handles if any (e.g. if peak and deepest were the same and plotted with same object but updated label)
            unique_handles = []
            seen_labels = set()
            for h in handles_for_legend:
                label = h.get_label()
                if label not in seen_labels:
                    unique_handles.append(h)
                    seen_labels.add(label)
            ax.legend(handles=unique_handles, loc='best', fontsize=8)


        plt.tight_layout()
        plot_filename = f"{just_the_filename_for_output_csv}_plot_dual_highlight.png" # New filename
        plot_save_path = os.path.join(output_save_folder_path, plot_filename)
        if fig:
            plt.savefig(plot_save_path)
            print(f"Plot for {current_file_name_for_output} saved to: {plot_save_path}")
            #plt.show() 
            plt.close(fig)

        # --- Save Iteration Data for the Current File to its own CSV ---
        if current_file_iterations_data:
            file_summary_df = pd.DataFrame(current_file_iterations_data)

            # --- Add Indicator Column Logic ---
            # Set all to 0 initially as per initialization in current_file_iterations_data.append
            # Then update based on conditions.
            if peak_R2_iteration_details is not None:
                # Find the row corresponding to the peak R^2 line and set Indicator to 1
                peak_R2_mask = (file_summary_df['Min Depth Range'] == peak_R2_iteration_details['min_depth_regr_setting']) & \
                               (file_summary_df['Max Depth Range'] == peak_R2_iteration_details['depth_limit']) & \
                               (file_summary_df['R2 Value'] == peak_R2_iteration_details['R2'])
                
                # Check if peak_R2_iteration_details refers to a row in current_file_iterations_data before assigning
                # This handles cases where positive_slope_results_df might be empty or peak_R2_iteration_details might be from a different source
                if not peak_R2_mask.empty and peak_R2_mask.any():
                    file_summary_df.loc[peak_R2_mask, 'Indicator'] = 1

            if deepest_tolerable_iteration_details is not None:
                # Find the row corresponding to the deepest tolerable R^2 line
                deepest_tolerable_mask = (file_summary_df['Min Depth Range'] == deepest_tolerable_iteration_details['min_depth_regr_setting']) & \
                                         (file_summary_df['Max Depth Range'] == deepest_tolerable_iteration_details['depth_limit']) & \
                                         (file_summary_df['R2 Value'] == deepest_tolerable_iteration_details['R2'])
                
                # Check if deepest_tolerable_iteration_details refers to a row in current_file_iterations_data
                if not deepest_tolerable_mask.empty and deepest_tolerable_mask.any():
                    # Check if it's the same line as the peak R^2 line
                    is_same_line = False
                    if peak_R2_iteration_details is not None:
                        is_same_line = (deepest_tolerable_iteration_details['original_k_index'] == peak_R2_iteration_details['original_k_index'] and \
                                        np.isclose(deepest_tolerable_iteration_details['min_depth_regr_setting'], peak_R2_iteration_details['min_depth_regr_setting']))
                    
                    if is_same_line:
                        # If only one regression line, mark as 2
                        file_summary_df.loc[deepest_tolerable_mask, 'Indicator'] = 2
                    else:
                        # If different, mark as 2 (deepest tolerable)
                        file_summary_df.loc[deepest_tolerable_mask, 'Indicator'] = 2

            output_csv_filename = f"{just_the_filename_for_output_csv}_LR_Stats_iterations.csv" # Slightly different name
            output_csv_save_path = os.path.join(output_save_folder_path, output_csv_filename)
            file_summary_df.to_csv(output_csv_save_path, index=False, float_format='%.4f')
            print(f"Iterations summary for {current_file_name_for_output} saved to: {output_csv_save_path}")
        else:
            print(f"No iteration data to save for {current_file_name_for_output}.")

    except Exception as e_outer:
        print(f"An unexpected error occurred while processing file {data_name_full_path}: {e_outer}")
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        continue

print("\n--- All CSV files processed. ---")



#################################################################################################################
#################################################################################################################

""" Flags pSDB that has R^2 values below a set threshold, saves them to csv file.
    Since I'm doing this more when creating the figures I'll leave it commented out for now """

#input_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results"

#output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results\Flagged Results"


# def process_r2_with_indicator_fallback(input_folder_path, output_csv_path,
#                                         r2_threshold=0.6,
#                                         indicator_col="Indicator",
#                                         r2_col="R2 Value",
#                                         image_name_col="Image Name",
#                                         flag_col_name="R2_Threshold_Outcome_Flag"):

#     results_to_save = []
#     output_column_names = [image_name_col, r2_col, flag_col_name]

#     if not os.path.isdir(input_folder_path):
#         print(f"Error: Input folder not found at '{input_folder_path}'")
#         return False

#     csv_files = glob(os.path.join(input_folder_path, "*.csv"))

#     if not csv_files:
#         print(f"No CSV files found in '{input_folder_path}'.")
#         if output_csv_path:
#             output_dir = os.path.dirname(output_csv_path)
#             if output_dir and not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             pd.DataFrame(columns=output_column_names).to_csv(output_csv_path, index=False)
#             print(f"No input CSVs, empty output file with headers created at '{output_csv_path}'")
#         return True

#     print(f"Found {len(csv_files)} CSV files to process in '{input_folder_path}'.")

#     for file_path in csv_files:
#         print(f"\nProcessing '{os.path.basename(file_path)}'...")
#         try:
#             df = pd.read_csv(file_path)
#             if df.empty:
#                 print(f"  Warning: File '{os.path.basename(file_path)}' is empty. Skipping.")
#                 continue

#             required_cols = [indicator_col, r2_col, image_name_col]
#             if not all(col in df.columns for col in required_cols):
#                 print(f"  Warning: File '{os.path.basename(file_path)}' is missing one or more required columns: {required_cols}. Skipping.")
#                 continue

#             df[indicator_col] = pd.to_numeric(df[indicator_col], errors='coerce')
#             df[r2_col] = pd.to_numeric(df[r2_col], errors='coerce')
#             df.dropna(subset=[indicator_col], inplace=True)

#             for name_of_image, group in df.groupby(image_name_col):
#                 row_ind2 = group[group[indicator_col] == 2].copy()
#                 row_ind1 = group[group[indicator_col] == 1].copy()

#                 r2_to_report = np.nan
#                 flag_for_report = np.nan
#                 image_name_for_report = name_of_image

#                 if not row_ind2.empty:
#                     ind2_data = row_ind2.iloc[0]
#                     r2_original_ind2 = ind2_data[r2_col]
#                     r2_check_ind2 = round(r2_original_ind2, 2) if pd.notna(r2_original_ind2) else np.nan
                    
#                     ind2_passes_threshold = pd.notna(r2_check_ind2) and r2_check_ind2 >= r2_threshold

#                     if ind2_passes_threshold:
#                         r2_to_report = r2_original_ind2
#                         flag_for_report = 1
#                     else: 
#                         if not row_ind1.empty:
#                             ind1_data = row_ind1.iloc[0]
#                             r2_original_ind1 = ind1_data[r2_col]
#                             r2_check_ind1 = round(r2_original_ind1, 2) if pd.notna(r2_original_ind1) else np.nan
                            
#                             ind1_passes_threshold = pd.notna(r2_check_ind1) and r2_check_ind1 >= r2_threshold

#                             if ind1_passes_threshold:
#                                 r2_to_report = r2_original_ind1
#                                 flag_for_report = 1
#                             else: 
#                                 r2_to_report = r2_original_ind2 
#                                 flag_for_report = 0 if pd.notna(r2_check_ind2) else np.nan
#                         else: 
#                             r2_to_report = r2_original_ind2
#                             flag_for_report = 0 if pd.notna(r2_check_ind2) else np.nan
                    
#                     results_to_save.append({
#                         image_name_col: image_name_for_report,
#                         r2_col: r2_to_report,
#                         flag_col_name: flag_for_report
#                     })

#         except pd.errors.EmptyDataError:
#             print(f"  Warning: File '{os.path.basename(file_path)}' became empty after initial processing. Skipping.")
#         except Exception as e:
#             print(f"  Error processing file '{os.path.basename(file_path)}': {e}. Skipping.")

#     if results_to_save:
#         output_df = pd.DataFrame(results_to_save, columns=output_column_names)
#         try:
#             output_dir = os.path.dirname(output_csv_path)
#             if output_dir and not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             output_df.to_csv(output_csv_path, index=False, float_format='%.4f')
#             print(f"\nSuccessfully saved {len(output_df)} entries to '{output_csv_path}'.")
#         except Exception as e:
#             print(f"\nError saving output CSV to '{output_csv_path}': {e}")
#             return False
#     else:
#         print("\nNo Image Names with an Indicator 2 row were found, or no data to report based on logic.")
#         try:
#             output_dir = os.path.dirname(output_csv_path)
#             if output_dir and not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             pd.DataFrame(columns=output_column_names).to_csv(output_csv_path, index=False)
#             print(f"Empty results file with headers created at '{output_csv_path}'.")
#         except Exception as e:
#             print(f"\nError saving empty output CSV to '{output_csv_path}': {e}")
#             return False
#     return True

# # --- How to use with your specific paths ---
# if __name__ == "__main__":
#     # Your provided paths
#     input_data_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results"
#     output_save_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results\Flagged Results"

#     # Define the full path for the output CSV file
#     output_filename = "Flagged R^2 Summary.csv" # You can change this name
#     full_output_csv_path = os.path.join(output_save_folder, output_filename)

#     # Make sure the output directory exists
#     if not os.path.exists(output_save_folder):
#         os.makedirs(output_save_folder)
#         print(f"Created output directory: {output_save_folder}")

#     # Call the function
#     success = process_r2_with_indicator_fallback(
#         input_folder_path=input_data_folder,
#         output_csv_path=full_output_csv_path,
#         r2_threshold=0.7,   ### R2 >= 0.7 is considered "meeting threshold" ###
#     )

#     if success:
#         print("\nProcess completed successfully.")
#         if os.path.exists(full_output_csv_path):
#             try:
#                 summary_df = pd.read_csv(full_output_csv_path)
#                 if not summary_df.empty:
#                     print(f"\nFirst 5 rows of the output file ('{full_output_csv_path}'):")
#                     print(summary_df.head())
#                     print(f"\nValue counts for '{summary_df.columns[-1]}' (the flag column):") # Assumes flag is last
#                     print(summary_df[summary_df.columns[-1]].value_counts(dropna=False))
#                 else:
#                     print(f"\nOutput file '{full_output_csv_path}' is empty.")
#             except pd.errors.EmptyDataError:
#                   print(f"\nOutput file '{full_output_csv_path}' is empty (pd.errors.EmptyDataError).")
#             except Exception as e:
#                 print(f"Could not read or display output CSV: {e}")
#     else:
#         print("\nProcess encountered an error.")


###################################################################################################################################################
##################################################################################################################################################


"""Create SDB red and green with constants from linear regression"""

def create_sdb_rasters(raster_folder, csv_folder, output_folder,
                       apply_r2_filter=False,
                       r2_filter_threshold=0.7,
                       indicator_col="Indicator",
                       r2_col="R2 Value",                    
                       m1_col="m1",
                       m0_col="m0",
                       nodata_value=-9999):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Output SDB rasters will be saved to: {output_folder}")
    if apply_r2_filter:
        print(f"R2 filter is ON. Threshold for SDB creation: {r2_col} >= {r2_filter_threshold}")
    else:
        print("R2 filter is OFF. Using first 'Indicator == 2' row if available from matched stats CSV.")

    stats_csv_filenames_in_folder = [f for f in os.listdir(csv_folder) if f.lower().endswith('.csv')]
    if not stats_csv_filenames_in_folder:
        print(f"CRITICAL DEBUG: No CSV files found in stats folder {csv_folder}. Cannot proceed.")
        return

    print(f"DEBUG: Found {len(stats_csv_filenames_in_folder)} CSV files in stats folder: {csv_folder}")
    processed_rasters_count = 0
    processed_at_all_count = 0 # To see if any raster even starts processing

    for pSDB_raster_filename in os.listdir(raster_folder):
        processed_at_all_count +=1
        print(f"\nDEBUG: Checking input raster file [{processed_at_all_count}]: {pSDB_raster_filename}")
        pSDB_raster_filename_lower = pSDB_raster_filename.lower()
        
        current_raster_path = os.path.join(raster_folder, pSDB_raster_filename) # Defined early

        current_pSDB_type_suffix_in_raster = None
        base_for_stats_csv_search = None
        output_sdb_suffix = None
        
        if pSDB_raster_filename_lower.endswith('_psdbgreen.tif'):
            current_pSDB_type_suffix_in_raster = '_psdbgreen.tif'
            base_for_stats_csv_search = pSDB_raster_filename_lower.replace(current_pSDB_type_suffix_in_raster, '_pSDBgreen_extracted')
            output_sdb_suffix = "_SDBgreen.tif"
        elif pSDB_raster_filename_lower.endswith('_psdbred.tif'):
            current_pSDB_type_suffix_in_raster = '_psdbred.tif'
            base_for_stats_csv_search = pSDB_raster_filename_lower.replace(current_pSDB_type_suffix_in_raster, '_pSDBred_extracted')
            output_sdb_suffix = "_SDBred.tif"
        else:
            if pSDB_raster_filename_lower.endswith('.tif'):
                print(f"  DEBUG SKIP: {pSDB_raster_filename} - not a recognized _psdbgreen.tif or _psdbred.tif file.")
            continue # Skip non-target files

        if not base_for_stats_csv_search or not output_sdb_suffix : # Should be caught by above continue
             print(f"  DEBUG ERROR: Could not determine base_for_stats_csv_search or output_sdb_suffix for {pSDB_raster_filename}. Skipping.")
             continue

        expected_stats_csv_name = f"{base_for_stats_csv_search}_LR_Stats_iterations.csv"
        print(f"Processing pSDB raster: {pSDB_raster_filename} (Derived base for stats CSV: '{base_for_stats_csv_search}', Expected stats CSV: '{expected_stats_csv_name}')")

        matched_stats_csv_name = None
        for name_in_list in stats_csv_filenames_in_folder:
            if name_in_list.lower() == expected_stats_csv_name.lower():
                matched_stats_csv_name = name_in_list
                print(f"  DEBUG: Exact match found for stats CSV: {matched_stats_csv_name}")
                break
        
        if not matched_stats_csv_name:
            print(f"  DEBUG: No exact match for stats CSV '{expected_stats_csv_name}'. Trying get_close_matches...")
            close_matches = get_close_matches(expected_stats_csv_name, stats_csv_filenames_in_folder, n=1, cutoff=0.85) # Stricter cutoff
            if close_matches:
                matched_stats_csv_name = close_matches[0]
                print(f"  DEBUG: Used get_close_matches. Found stats CSV: {matched_stats_csv_name}")
        
        if not matched_stats_csv_name:
            print(f"  DEBUG FAIL (Stage 1): No matching stats CSV file found for pSDB raster '{pSDB_raster_filename}'. Expected pattern was '{expected_stats_csv_name}'. Skipping this raster.")
            continue
            
        stats_csv_path = os.path.join(csv_folder, matched_stats_csv_name)
        print(f"  Using stats CSV: {matched_stats_csv_name} at path: {stats_csv_path}")

        try:
            if not os.path.exists(stats_csv_path):
                print(f"  DEBUG FAIL (Stage 2a): Stats CSV file does not exist: {stats_csv_path}. Skipping.")
                continue

            stats_df = pd.read_csv(stats_csv_path)
            if stats_df.empty:
                print(f"  DEBUG FAIL (Stage 2b): Stats CSV {matched_stats_csv_name} is empty. Skipping pSDB raster {pSDB_raster_filename}.")
                continue
            # print(f"    DEBUG: Stats CSV head:\n{stats_df.head()}") # For deep debugging
            
            essential_cols = [indicator_col, m1_col, m0_col]
            if apply_r2_filter:
                essential_cols.append(r2_col)
            
            missing_cols = [col for col in essential_cols if col not in stats_df.columns]
            if missing_cols:
                print(f"  DEBUG FAIL (Stage 3): Stats CSV {matched_stats_csv_name} is missing required columns: {missing_cols}. Skipping.")
                continue
            
            # Convert to numeric and clean NA for these essential columns from the entire stats_df
            stats_df[indicator_col] = pd.to_numeric(stats_df[indicator_col], errors='coerce')
            stats_df[m1_col] = pd.to_numeric(stats_df[m1_col], errors='coerce')
            stats_df[m0_col] = pd.to_numeric(stats_df[m0_col], errors='coerce')
            
            # Drop rows if these absolutely essential cols became NaN; R2 can be NaN if filter is off
            cleaned_stats_df = stats_df.dropna(subset=[indicator_col, m1_col, m0_col]).copy()

            if apply_r2_filter:
                cleaned_stats_df[r2_col] = pd.to_numeric(cleaned_stats_df[r2_col], errors='coerce')
                # For R2 filter, if R2 is NaN, that row cannot pass. No need to drop here, logic below handles NaN R2.

            if cleaned_stats_df.empty:
                print(f"  DEBUG FAIL (Stage 4): No valid coefficient rows in {matched_stats_csv_name} after initial cleaning for {pSDB_raster_filename}. Skipping SDB creation.")
                continue

            m1_to_use, m0_to_use = np.nan, np.nan
            coeffs_were_selected = False
            chosen_coeffs_description = "None (default)"

            if apply_r2_filter:
                print(f"    DEBUG: Applying R2 filter. Threshold: {r2_filter_threshold}")
                # Rows from the *entire cleaned_stats_df* that match indicator 2 or 1
                rows_ind2_all = cleaned_stats_df[cleaned_stats_df[indicator_col] == 2]
                rows_ind1_all = cleaned_stats_df[cleaned_stats_df[indicator_col] == 1]
                selected_row_from_stats = None

                if not rows_ind2_all.empty:
                    # If multiple Ind2 rows, pick the one with highest R2 (if R2 column exists and is valid)
                    # or just the first one if R2 is not for sorting here.
                    # Assuming the previous script correctly marked ONE Ind2 row as the "best" for that type.
                    temp_row_ind2 = rows_ind2_all.iloc[0]
                    if r2_col in temp_row_ind2 and pd.notna(temp_row_ind2[r2_col]):
                        r2_val_ind2 = temp_row_ind2[r2_col]
                        r2_check_ind2 = round(r2_val_ind2, 2)
                        print(f"      DEBUG: Ind2 candidate R2={r2_check_ind2:.2f} (Original: {r2_val_ind2})")
                        if r2_check_ind2 >= r2_filter_threshold:
                            selected_row_from_stats = temp_row_ind2
                            chosen_coeffs_description = f"Indicator 2 (R2={r2_check_ind2:.2f})"
                        else:
                            print(f"      Indicator 2 R2 ({r2_check_ind2:.2f}) < {r2_filter_threshold}. Checking Ind1.")
                    else: # R2 value is NaN or R2 column missing for this Ind2 row
                        r2_check_ind2 = np.nan # Ensure it's marked as NaN for print
                        print("      Indicator 2 R2 is NaN/missing. Checking Ind1.")

                    if selected_row_from_stats is None and not rows_ind1_all.empty: # Ind2 failed or had NaN R2, try Ind1
                        temp_row_ind1 = rows_ind1_all.iloc[0] # Similar logic for multiple Ind1 rows
                        if r2_col in temp_row_ind1 and pd.notna(temp_row_ind1[r2_col]):
                            r2_val_ind1 = temp_row_ind1[r2_col]
                            r2_check_ind1 = round(r2_val_ind1, 2)
                            print(f"        DEBUG: Ind1 candidate R2={r2_check_ind1:.2f} (Original: {r2_val_ind1})")
                            if r2_check_ind1 >= r2_filter_threshold:
                                selected_row_from_stats = temp_row_ind1
                                chosen_coeffs_description = f"Indicator 1 (Fallback, R2={r2_check_ind1:.2f})"
                            else:
                                print(f"        Indicator 1 R2 ({r2_check_ind1:.2f}) also < {r2_filter_threshold}.")
                        else: # R2 value is NaN or R2 column missing for this Ind1 row
                            print("        Indicator 1 R2 is NaN/missing.")
                    elif selected_row_from_stats is None: # Ind2 failed and no Ind1 rows
                         print("      No Indicator 1 rows found for fallback.")
                else:
                     print(f"    DEBUG: No Indicator 2 rows found in {matched_stats_csv_name} for R2 check.")

                if selected_row_from_stats is not None:
                    m1_to_use = selected_row_from_stats[m1_col]
                    m0_to_use = selected_row_from_stats[m0_col]
                    coeffs_were_selected = True
                else:
                    print(f"  DEBUG FAIL (Stage 5a - R2 Filter ON): No coefficients passed threshold for {pSDB_raster_filename}.")
            
            else: # R2 filter is OFF
                print(f"    DEBUG: R2 filter OFF for {pSDB_raster_filename}")
                rows_ind2_no_filter = cleaned_stats_df[cleaned_stats_df[indicator_col] == 2]
                if not rows_ind2_no_filter.empty:
                    best_fit_row_no_filter = rows_ind2_no_filter.iloc[0]
                    m1_to_use = best_fit_row_no_filter[m1_col]
                    m0_to_use = best_fit_row_no_filter[m0_col]
                    coeffs_were_selected = True
                    chosen_coeffs_description = "Indicator 2 (R2 filter OFF)"
                else:
                    print(f"  DEBUG FAIL (Stage 5b - R2 Filter OFF): No Indicator 2 coefficients found in {matched_stats_csv_name}.")

            if not coeffs_were_selected:
                print(f"  Final decision: Skipping SDB creation for {pSDB_raster_filename} as no suitable coefficients were selected.")
                continue

            print(f"  Using Coeffs for {pSDB_raster_filename}: {chosen_coeffs_description}. m1={m1_to_use:.4f}, m0={m0_to_use:.4f}")
            
            # SDB raster creation
            with rasterio.open(current_raster_path) as src: # current_raster_path is defined
                pSDB_array = src.read(1); src_profile = src.profile.copy(); src_nodata = src.nodata
                pSDB_for_calc = pSDB_array.astype(np.float32)
                if src_nodata is not None: pSDB_for_calc[pSDB_for_calc == src_nodata] = np.nan
                result = m1_to_use * pSDB_for_calc + m0_to_use
                result = result.astype(np.float32)
                result_filled = np.where(np.isnan(result) | (result < 0), nodata_value, result)
                result_filled[np.isnan(pSDB_for_calc)] = nodata_value
                
                # Construct output name by replacing the pSDB part with SDB
                output_raster_base_name = pSDB_raster_filename_lower.replace(current_pSDB_type_suffix_in_raster, "")
                output_filename_sdb = f"{output_raster_base_name}{output_sdb_suffix}" # output_sdb_suffix is _SDBgreen.tif or _SDBred.tif
                
                output_raster_path = os.path.join(output_folder, output_filename_sdb)
                dst_profile = src_profile
                dst_profile.update(dtype=result.dtype, count=1, nodata=nodata_value)
                with rasterio.open(output_raster_path, 'w', **dst_profile) as dst:
                    dst.write(result_filled, 1)
                print(f"  SUCCESS: Saved SDB raster: {output_raster_path}")
                processed_rasters_count += 1

        except FileNotFoundError:
            print(f"  Error: Stats CSV file {stats_csv_path} was not found (unexpected after matching).")
        except pd.errors.EmptyDataError:
            print(f"  Warning: Stats CSV file {stats_csv_path} is empty after loading.")
        except KeyError as e_key:
            print(f"  KeyError: Column ({e_key}) missing in {matched_stats_csv_name} or during processing of {pSDB_raster_filename}.")
        except Exception as e_main:
            print(f"  An unexpected error occurred processing {pSDB_raster_filename} with {matched_stats_csv_name}: {e_main}")
            import traceback
            traceback.print_exc()

    if processed_rasters_count == 0:
        print("\nFINAL RESULT: No SDB rasters were generated in this run based on the criteria and processing steps.")
    else:
        print(f"\n--- SDB Raster Creation Finished --- Successfully generated {processed_rasters_count} SDB rasters.")


# ############## CHECK DIRECTORIES/INPUTS ###########################
raster_folder = r"E:\Thesis Stuff\pSDB"
csv_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results" 
output_folder = r"E:\Thesis Stuff\SDB" # Use a fresh, distinct output folder for this test

create_sdb_rasters(raster_folder, csv_folder, output_folder,
                   apply_r2_filter=True,
                   r2_filter_threshold=0.0,     ### !! CHANGE BACK TO 0.7 WHEN DONE !!! ####
                   indicator_col="Indicator",
                   r2_col="R2 Value",
                   # image_name_col="Image Name", # Not used for filtering within stats CSV
                   m1_col="m1", 
                   m0_col="m0" 
                   )


##############################################################################################################
##############################################################################################################

"""Merge SDB red and green together"""


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
    sdb_red = np.nan_to_num(sdb_red, nan=np.nan) 
    sdb_green = np.nan_to_num(sdb_green, nan=np.nan)  


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
        #sdb_merged_filled = np.where(np.isnan(sdb_merged), nodata_value, sdb_merged).astype(np.float32)
        
        # Save the new merged raster
        with rasterio.open(output_path, 'w', driver='GTiff', count=1, dtype=sdb_merged.dtype,
                            height=sdb_merged.shape[0], width=sdb_merged.shape[1],
                            crs=red_dataset.crs, transform=red_dataset.transform, nodata=nodata_value) as dst:
            dst.write(sdb_merged, 1)

        print(f"Saved merged SDB raster: {output_path}")

### Save Results Path ### 
#input_folder = r"B:\Thesis Project\SDB_Time\Results_main\Homer\SuperDove\SDB"

### Workspace Path ###
input_folder = r"E:\Thesis Stuff\SDB"  # Folder with input rasters


process_sdb_folder(input_folder)



##############################################################################################################
##############################################################################################################

"""Extract SDB values at reference point locations"""


def extract_raster_values_optimized(val_csv_file, raster_folder, output_folder):
    """
    Extracts raster values at specified locations using optimized vectorized operations.

    Args:
        val_csv_file (str): Path to the CSV file with point coordinates.
        raster_folder (str): Path to the folder containing GeoTIFF raster files.
        output_folder (str): Path to the folder for saving results.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load and Prepare Point Data (Done Once) ---
    print("Loading and preparing validation points...")
    df = pd.read_csv(val_csv_file)
    
    # Assign clear column names
    df.columns = ['Easting(m)', 'Northing(m)', 'Geoid_Corrected_Ortho_Height']
    height_col_name = 'Geoid_Corrected_Ortho_Height'

    # Convert height to depth: only multiply negative values by -1
    print(f"Converting negative values in '{height_col_name}' to positive depth.")
    df[height_col_name] = pd.to_numeric(df[height_col_name], errors='coerce')
    # Use .loc to ensure we are modifying the actual DataFrame
    df.loc[df[height_col_name] < 0, height_col_name] *= -1
    
    # Create a GeoDataFrame from the points
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['Easting(m)'], df['Northing(m)'])
    )

    # Process Each Raster File
    for raster_file in os.listdir(raster_folder):
        if raster_file.lower().endswith(".tif"):
            raster_path = os.path.join(raster_folder, raster_file)
            print(f"\nProcessing raster: {raster_file}")

            with rasterio.open(raster_path) as src:
                # Set the CRS of the points to match the raster's CRS
                if gdf.crs != src.crs:
                    print(f"Assigning raster CRS ({src.crs}) to points.")
                    print(gdf.crs)
                    gdf.set_crs(src.crs, inplace=True)

                # Fast Filtering with a Spatial Index
                raster_bbox = box(*src.bounds)
                gdf_in_bounds = gdf.clip(raster_bbox)

                if gdf_in_bounds.empty:
                    print(f"No points overlap with raster: {raster_file}")
                    continue
                
                print(f"Found {len(gdf_in_bounds)} points within the raster bounds.")

                # Vectorized Value Extraction 
                coords = [(p.x, p.y) for p in gdf_in_bounds.geometry]
                raster_values = [val[0] for val in src.sample(coords)]

                # Add the extracted values back to the GeoDataFrame
                gdf_in_bounds = gdf_in_bounds.copy()
                gdf_in_bounds.loc[:, 'Raster_Value'] = raster_values
                
                # Clean and Save Results
                # Remove points where the raster value is the NoData value or NaN
                # This handles cases where sample_gen might return the nodata value
                if src.nodata is not None:
                    gdf_in_bounds = gdf_in_bounds[gdf_in_bounds['Raster_Value'] != src.nodata]
                gdf_in_bounds.dropna(subset=['Raster_Value'], inplace=True)

                if gdf_in_bounds.empty:
                    print(f"No valid raster values found for raster: {raster_file}")
                    continue

                # Save the results to a new CSV
                output_filename = os.path.splitext(raster_file)[0] + "_extracted.csv"
                output_path = os.path.join(output_folder, output_filename)
                gdf_in_bounds.to_csv(output_path, index=False)
                print(f"Results saved to: {output_path}")

# ######################  CHECK DIRECTORIES/INPUTS #####################
#!!!
val_csv_file = r"B:\Thesis Project\Reference Data\Processed_ICESat\Anegada_acc.csv"
raster_folder = r"E:\Thesis Stuff\SDB"
output_folder = r"E:\Thesis Stuff\SDB_ExtractedPts"

# Run the optimized function
extract_raster_values_optimized(val_csv_file, raster_folder, output_folder)

##############################################################################################################
##############################################################################################################

"""Perform linear regressions between SDB and other reference data for accuracy"""


# def process_csv_files(input_folder, output_folder):
#     """
#     Processes CSV files in the input folder, performs linear regression, 
#     and saves the results in the output folder.
#     """
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # Loop through all CSV files in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".csv"):
#             input_path = os.path.join(input_folder, filename)
#             print(f"Processing {filename}...")

#             # Read the CSV file
#             data = pd.read_csv(input_path)

#             # Drop rows where 'Raster_Value' is blank
#             data = data.dropna(subset=['Raster_Value'])

#             # Initialize a results list
#             results = []

#             # Perform linear regression
#             x = data[['Raster_Value']].values
#             y = data['Geoid_Corrected_Ortho_Height'].values
#             model = LinearRegression()
#             model.fit(x, y)

#             # Calculate regression metrics
#             y_pred = model.predict(x)
#             r2 = r2_score(y, y_pred)
#             rmse = np.sqrt(mean_squared_error(y, y_pred))
#             coef = model.coef_[0]
#             intercept = model.intercept_

#             # Create the line of best fit equation
#             equation = f"y = {coef:.4f}x + {intercept:.4f}"

#             # Calculate regression metrics
#             y_pred = model.predict(x)
#             r2 = r2_score(y, y_pred)  # Scikit-learn R² calculation
#             rmse = np.sqrt(mean_squared_error(y, y_pred))

#             # Calculate perpendicular distances
#             distances = np.abs(coef * x.flatten() - y + intercept) / np.sqrt(coef**2 + 1)
            
#             # Compute statistics for distances
#             min_dist = np.min(distances)
#             max_dist = np.max(distances)
#             mean_dist = np.mean(distances)
#             std_dist = np.std(distances)
            
#             data["Perpendicular distances to line"] = distances


#             # Append the results to the list
#             results.append({
#                 "Image Name": filename,
#                 "R^2": r2,
#                 "RMSE": rmse,        
#                 "Line of Best Fit": equation,
#                 "m1": coef,
#                 "m0": intercept,
#                 "min perp dist": min_dist,
#                 "max perp dist": max_dist,
#                 "mean perp dist": mean_dist,
#                 "std dev perp dist": std_dist 
#             })
            
#             min_x = np.min(x)
#             mean_y = np.mean(y)

#             # Compute the x-intercept (where y = 0)
#             x_intercept = -intercept / coef if coef != 0 else np.min(x)

#             # Generate extended x values from x_intercept to max x
#             x_ext = np.linspace(x_intercept, np.max(x), 100)

#             # Compute corresponding y values
#             y_ext = coef * x_ext + intercept

#             # Regression Plot
#             plt.figure(figsize=(8, 6))
#             plt.scatter(x, y, color='blue', label='Data Points', alpha=0.7)
#             plt.plot(x_ext, y_ext, color='red', label='Best Fit Line', linewidth=2)
#             plt.title(f"Linear Regression for {filename}")
#             plt.xlabel("SDB Value (m)")
#             plt.ylabel("Reference Depths (m)")
#             plt.legend()
#             plt.grid(True)    
#             plt.xlim(0, None)
#             plt.ylim(0, None)
            
#             # Add R^2 and RMSE as text on the plot
#             plt.text(min_x, mean_y, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nIntercept = {intercept:0.2f}", fontsize=10, 
#                      bbox=dict(facecolor='white', alpha=0.5), ha='left')

#             # Save the regression plot in the output folder
#             plot_filename = f"{os.path.splitext(filename)[0]}_LR_plot.png"
#             plot_path = os.path.join(output_folder, plot_filename)
#             plt.savefig(plot_path)
#             plt.close()

#             # Convert the results into a DataFrame
#             results_df = pd.DataFrame(results)

#             # Generate the output file name
#             output_filename = f"{os.path.splitext(filename)[0]}_LR_stats.csv"
#             output_path = os.path.join(output_folder, output_filename)


#             # Save the results to the output CSV file
#             results_df.to_csv(output_path, index=False)
#             #print(f"Results saved to {output_path}")


#       #################  CHECK DIRECTORIES/INPUTS #####################

# ### Save Results Path ###
# #input_folder = r"B:\Thesis Project\SDB_Time\Results_main\Homer\SuperDove\Extracted Pts\SDB"

# ### Workspace Path ###
# input_folder = r"E:\Thesis Stuff\SDB_ExtractedPts"  


# ### Save Results Path ###
# #output_folder = r"B:\Thesis Project\SDB_Time\Results_main\Homer\SuperDove\Figures\SDB"

# ### Workspace Path ###
# output_folder = r"E:\Thesis Stuff\SDB_ExtractedPts_Results" 


# process_csv_files(input_folder, output_folder)



##############################################################################################################
##############################################################################################################

"""Better Optically Deep finder: Linear regression to find the best line of fit with highest R^2 value, and the 
   next best line that is still within 5% of highest R^2 vale """


### Save Results Path ###
#data_folder_path = r"B:\Thesis Project\SDB_Time\Results_main\Homer\SuperDove\Extracted Pts\pSDB"

### Workspace Path ###
data_folder_path = r"E:\Thesis Stuff\SDB_ExtractedPts" 


### Save Results Path ###
#output_save_folder_path = r"B:\Thesis Project\SDB_Time\Results_main\Homer\SuperDove\Figures\pSDB_maxR2"

### Workspace Path ###
output_save_folder_path = r"E:\Thesis Stuff\SDB_ExtractedPts_maxR2_results"



# Create the output folder if it doesn't exist
if not os.path.exists(output_save_folder_path):
    os.makedirs(output_save_folder_path)
    print(f"Created output folder: {output_save_folder_path}")

csv_files = glob(os.path.join(data_folder_path, "*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in the specified folder: {data_folder_path}")

print(f"Found {len(csv_files)} CSV files to process in: {data_folder_path}")

for data_name_full_path in csv_files:
    print(f"\n--- Processing file: {data_name_full_path} ---")
    current_file_name_for_output = os.path.basename(data_name_full_path)
    just_the_filename_for_output_csv = os.path.splitext(current_file_name_for_output)[0]

    fig = None
    current_file_iterations_data = []

    try:
        # Load Data
        if not os.path.isfile(data_name_full_path):
            print(f'Warning: Data CSV file not found: {data_name_full_path}')
            continue
        try:
            data_df = pd.read_csv(data_name_full_path)
            if data_df.shape[1] < 5:
                raise ValueError('CSV file does not have enough columns. Expecting at least 5.')

            y_original = data_df["Geoid_Corrected_Ortho_Height"].values.astype(float)
            x_original = data_df["Raster_Value"].values.astype(float)

            nan_mask = ~ (np.isnan(x_original) | np.isnan(y_original))
            x_original = x_original[nan_mask]
            y_original = y_original[nan_mask]

            if x_original.size == 0 or y_original.size == 0:
                raise ValueError('No valid data points after removing NaNs from x or y.')

            valid_x_idx = (x_original >= 0) & (x_original <= 40)
            x_original = x_original[valid_x_idx]
            y_original = y_original[valid_x_idx]

            if x_original.size == 0 or y_original.size == 0:
                raise ValueError('No data points remain after filtering x values between 0 and 5.')

        except Exception as e:
            print(f'Failed to read or process input CSV file {data_name_full_path}. Error: {e}')
            if fig and plt.fignum_exists(fig.number): plt.close(fig)
            continue

        data_name_for_conditions = str(just_the_filename_for_output_csv).lower()

        # Initial Plot Setup
        fig, ax = plt.subplots(figsize=(10, 7))
        if "green" in data_name_for_conditions: depth_min_limit_for_plot = 0.0
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
        ax.set_xlabel('SDB Value (m)')
        ax.set_ylabel('Reference Depth (m)')
        ax.set_title(f'SDB Regression: {just_the_filename_for_output_csv.replace("_", " ")}') # Slightly shorter title
        ax.grid(True)


        # Plot 1:1 diagonal line 
        # Get the current axis limits to determine the span for the 1:1 line
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        
        # Find the overall min and max to make the line span the whole plot
        line_min = min(0, 0)
        line_max = max(current_xlim[1], current_ylim[1])
        
        # Plot the 1:1 line
        one_to_one_handle = ax.plot([line_min, line_max], [line_min, line_max], 
                                    color='gray', 
                                    linestyle=':', 
                                    linewidth=1.5, 
                                    zorder=0)[0] 
        
        # Optionally, force the axes to be equal
        ax.set_aspect('equal', adjustable='box')
        
        # Reset limits after setting aspect ratio, as it might change them
        ax.set_xlim(current_xlim)
        ax.set_ylim(current_ylim)

        iteration_results_for_plot_obj = []

        # Define the two sets of depth_min_limit_regr ###
        depth_min_regr_sets = [1.0, 1.25, 1.5, 1.75, 2.0]
        step = 0.25 # Common step

        if y_original.size > 0:
            overall_max_depth_data = np.max(y_original) # Max depth from this specific file
            print(f"DEBUG: Dynamically set overall_max_depth_data to: {overall_max_depth_data:.2f} m")
        else:
            print("Warning: y_original is empty for regression. Skipping iterations.")
            # Save empty plot if needed and continue
            if fig: plt.savefig(os.path.join(output_save_folder_path, f"{just_the_filename_for_output_csv}_plot_no_data_for_regr.png")); plt.close(fig)
            continue

        # Loop through each depth_min_limit_regr set ###
        for depth_min_limit_regr in depth_min_regr_sets:
            print(f"\n--- Iterating with Min Depth Range starting at: {depth_min_limit_regr:.2f} m ---")

            # Determine initial_depth_max based on the current depth_min_limit_regr
            initial_depth_max = depth_min_limit_regr + step # Start one step above the min
            if initial_depth_max < depth_min_limit_regr: # handles if step is 0 or negative, though unlikely
                initial_depth_max = depth_min_limit_regr

            # Cap initial_depth_max at overall_max_depth_data
            initial_depth_max = min(initial_depth_max, overall_max_depth_data)

            if initial_depth_max >= overall_max_depth_data:
                if overall_max_depth_data >= depth_min_limit_regr:
                    depth_max_limits_to_test = np.array([overall_max_depth_data])
                else:
                    depth_max_limits_to_test = np.array([])
            else:
                start_arange = max(initial_depth_max, depth_min_limit_regr) # Ensure arange starts correctly
                if start_arange >= overall_max_depth_data:
                    if overall_max_depth_data >= depth_min_limit_regr:
                        depth_max_limits_to_test = np.array([overall_max_depth_data])
                    else:
                        depth_max_limits_to_test = np.array([])
                else:
                    depth_max_limits_to_test = np.arange(start_arange, overall_max_depth_data + step, step)
                    if not np.isclose(depth_max_limits_to_test[-1], overall_max_depth_data) and \
                       depth_max_limits_to_test[-1] < overall_max_depth_data and \
                       overall_max_depth_data >= depth_min_limit_regr:
                        depth_max_limits_to_test = np.append(depth_max_limits_to_test, overall_max_depth_data)
            
            # Ensure all test limits are valid for the current depth_min_limit_regr
            if len(depth_max_limits_to_test) > 0:
                depth_max_limits_to_test = depth_max_limits_to_test[depth_max_limits_to_test >= depth_min_limit_regr]
                depth_max_limits_to_test = np.unique(depth_max_limits_to_test) # Remove duplicates and sort

            if len(depth_max_limits_to_test) == 0:
                print(f'Warning: No valid depth ranges for Min Depth {depth_min_limit_regr:.2f}m. Skipping this set.')
                continue

            print(f'Calculating regression for {len(depth_max_limits_to_test)} depth ranges (Min Depth: {depth_min_limit_regr:.2f}m) for {current_file_name_for_output}...')
            for k_loop_idx, current_depth_max_limit in enumerate(depth_max_limits_to_test):
                plot_result_entry = {'depth_limit': current_depth_max_limit, 'R2': np.nan, 'params': None,
                                     'point_count': 0, 'x_min_fit': np.nan, 'x_max_fit': np.nan, 'slope_m': np.nan,
                                     'original_k_index': k_loop_idx, # You can use this or a combined index later
                                     'min_depth_regr_setting': depth_min_limit_regr} # Store which min_depth this iteration belongs to

                m_for_iteration, b_for_iteration, R2_for_iteration, rmse_for_iteration = np.nan, np.nan, np.nan, np.nan
                equation_for_iteration = "N/A"
                
                # range_idx uses the current depth_min_limit_regr from the outer loop
                range_idx = (y_original >= depth_min_limit_regr) & (y_original <= current_depth_max_limit)
                x_iter_range, y_iter_range = x_original[range_idx], y_original[range_idx]
                num_points = len(x_iter_range)
                plot_result_entry['point_count'] = num_points

                if num_points > 1:
                    params = np.polyfit(x_iter_range, y_iter_range, 1)
                    y_fit_iter_range = np.polyval(params, x_iter_range)
                    m_for_iteration, b_for_iteration = params[0], params[1]
                    plot_result_entry['slope_m'] = m_for_iteration
                    if len(np.unique(y_iter_range)) > 1:
                        R2_for_iteration = r2_score(y_iter_range, y_fit_iter_range)
                        if R2_for_iteration < 0: R2_for_iteration = 0.0
                    elif len(x_iter_range) > 0:
                        R2_for_iteration = 1.0 if np.allclose(y_iter_range, y_fit_iter_range) else 0.0
                    
                    plot_result_entry['R2'] = R2_for_iteration
                    plot_result_entry['params'] = params
                    if len(x_iter_range) > 0:
                        plot_result_entry['x_min_fit'], plot_result_entry['x_max_fit'] = np.min(x_iter_range), np.max(x_iter_range)
                    equation_for_iteration = f"y = {m_for_iteration:.4f}x + {b_for_iteration:.4f}"
                    rmse_for_iteration = np.sqrt(mean_squared_error(y_iter_range, y_fit_iter_range))
                
                iteration_results_for_plot_obj.append(plot_result_entry) # Appends to the list for the plot object
                
                # Append to the CSV data list, including the Min Depth Range setting
                current_file_iterations_data.append({
                    'Image Name': current_file_name_for_output,
                    'Min Depth Range': depth_min_limit_regr, # This now reflects 1.0 or 2.0
                    'Max Depth Range': current_depth_max_limit,
                    'R2 Value': R2_for_iteration, 'RMSE': rmse_for_iteration,
                    'Line of Best Fit': equation_for_iteration, 'm1': m_for_iteration, 'm0': b_for_iteration,
                    'Pt Count': num_points,
                    'Indicator': 0 # Initialize Indicator to 0, will update later
                })
        
        
        results_df_for_plot = pd.DataFrame(iteration_results_for_plot_obj)
        # Ensure sorted by depth_limit for the subsequent logic, though arange should provide this.
        results_df_for_plot = results_df_for_plot.sort_values(by='depth_limit').reset_index(drop=True)


        peak_R2_iteration_details = None
        peak_R2_value = -np.inf
        rmse_for_peak_R2_fit = np.nan

        deepest_tolerable_iteration_details = None
        rmse_for_deepest_tolerable_fit = np.nan
        R2_threshold = -np.inf

        if not results_df_for_plot.empty:
            positive_slope_results_df = results_df_for_plot[results_df_for_plot['slope_m'] > 1e-9].copy()

            if not positive_slope_results_df.empty and not positive_slope_results_df['R2'].isna().all():
                # Find Peak R² Line
                peak_idx_in_positive_df = positive_slope_results_df['R2'].idxmax()
                peak_R2_iteration_details = positive_slope_results_df.loc[peak_idx_in_positive_df]
                peak_R2_value = peak_R2_iteration_details['R2']

                if peak_R2_iteration_details['params'] is not None and len(peak_R2_iteration_details['params']) == 2:
                    fit_range_idx = (y_original >= depth_min_limit_regr) & (y_original <= peak_R2_iteration_details['depth_limit'])
                    if np.sum(fit_range_idx) > 1:
                        y_pred = np.polyval(peak_R2_iteration_details['params'], x_original[fit_range_idx])
                        rmse_for_peak_R2_fit = np.sqrt(mean_squared_error(y_original[fit_range_idx], y_pred))

                # Find Deepest Tolerable R² Line
                R2_threshold = peak_R2_value * 0.90
                
                # Iterate through positive slope results to find the last one meeting the threshold
                for index, row in positive_slope_results_df.iterrows():
                    if pd.notna(row['R2']) and row['R2'] >= R2_threshold:
                        deepest_tolerable_iteration_details = row # Keep updating, last one will be the deepest
                
                if deepest_tolerable_iteration_details is not None and \
                   deepest_tolerable_iteration_details['params'] is not None and \
                   len(deepest_tolerable_iteration_details['params']) == 2:
                    fit_range_idx_deep = (y_original >= depth_min_limit_regr) & (y_original <= deepest_tolerable_iteration_details['depth_limit'])
                    if np.sum(fit_range_idx_deep) > 1:
                        y_pred_deep = np.polyval(deepest_tolerable_iteration_details['params'], x_original[fit_range_idx_deep])
                        rmse_for_deepest_tolerable_fit = np.sqrt(mean_squared_error(y_original[fit_range_idx_deep], y_pred_deep))
            else:
                print(f"No iterations with a positive slope and valid R2 found for file: {current_file_name_for_output}")
        else:
            print(f"No iteration results to process for R2 for file: {current_file_name_for_output}")


        # # Plot All Regression Lines 
        # if not results_df_for_plot.empty:
        #     for index, row_data in results_df_for_plot.iterrows():
        #         if pd.notna(row_data['R2']) and row_data['params'] is not None and \
        #            pd.notna(row_data['x_min_fit']) and pd.notna(row_data['x_max_fit']):
        #             p_current = row_data['params']
        #             current_x_min, current_x_max = row_data['x_min_fit'], row_data['x_max_fit']
        #             if current_x_min == current_x_max:
        #                 if len(x_for_scatter) > 0:
        #                     overall_x_min_scatter, overall_x_max_scatter = np.min(x_for_scatter), np.max(x_for_scatter)
        #                     x_line_segment = np.array([overall_x_min_scatter, overall_x_max_scatter]) if overall_x_min_scatter != overall_x_max_scatter else np.array([current_x_min -0.1, current_x_min + 0.1])
        #                 else: x_line_segment = np.array([current_x_min - 0.1, current_x_min + 0.1])
        #             else: x_line_segment = np.linspace(current_x_min, current_x_max, 20)
        #             y_plot_fit_segment = np.polyval(p_current, x_line_segment)
        #             ax.plot(x_line_segment, y_plot_fit_segment, color=[0.7, 0.7, 0.7], linewidth=0.5)

        peak_R2_line_handle = None
        deepest_tolerable_line_handle = None
        color_deepest_tolerable = 'blue' # Choose a distinct color

        # Plot Peak R² Line (Red)
        if peak_R2_iteration_details is not None and peak_R2_iteration_details['params'] is not None:
            params = peak_R2_iteration_details['params']
            x_min = peak_R2_iteration_details['x_min_fit']
            x_max = peak_R2_iteration_details['x_max_fit']
            if pd.notna(x_min) and pd.notna(x_max):
                if x_min == x_max:
                    if len(x_for_scatter) > 0:
                        overall_x_min_scatter, overall_x_max_scatter = np.min(x_for_scatter), np.max(x_for_scatter)
                        x_segment = np.array([overall_x_min_scatter, overall_x_max_scatter]) if overall_x_min_scatter != overall_x_max_scatter else np.array([x_min -0.1, x_min + 0.1])
                    else: x_segment = np.array([x_min - 0.1, x_min + 0.1])
                else: x_segment = np.linspace(x_min, x_max, 20)
                y_segment = np.polyval(params, x_segment)
                label = ("Peak R² Fit")
                line_plots = ax.plot(x_segment, y_segment, color='r', linewidth=2.5, label=label)
                if line_plots: peak_R2_line_handle = line_plots[0]

        # Plot Deepest Tolerable R² Line (New Color)
        if deepest_tolerable_iteration_details is not None and deepest_tolerable_iteration_details['params'] is not None:
            # Avoid plotting twice if it's the same as the peak R2 line
            # Check based on original_k_index or a combination of R2 and depth_limit
            is_same_as_peak = False
            if peak_R2_iteration_details is not None:
                 if deepest_tolerable_iteration_details['original_k_index'] == peak_R2_iteration_details['original_k_index'] and \
                    np.isclose(deepest_tolerable_iteration_details['min_depth_regr_setting'], peak_R2_iteration_details['min_depth_regr_setting']):
                    is_same_as_peak = True
            
            if not is_same_as_peak:
                params = deepest_tolerable_iteration_details['params']
                x_min = deepest_tolerable_iteration_details['x_min_fit']
                x_max = deepest_tolerable_iteration_details['x_max_fit']
                if pd.notna(x_min) and pd.notna(x_max):
                    if x_min == x_max:
                        if len(x_for_scatter) > 0:
                            overall_x_min_scatter, overall_x_max_scatter = np.min(x_for_scatter), np.max(x_for_scatter)
                            x_segment = np.array([overall_x_min_scatter, overall_x_max_scatter]) if overall_x_min_scatter != overall_x_max_scatter else np.array([x_min -0.1, x_min + 0.1])
                        else: x_segment = np.array([x_min - 0.1, x_min + 0.1])
                    else: x_segment = np.linspace(x_min, x_max, 20)
                    y_segment = np.polyval(params, x_segment)
                    label = ("Deepest Tolerable R² Fit")
                    line_plots = ax.plot(x_segment, y_segment, color=color_deepest_tolerable, linewidth=2.0, linestyle='--', label=label)
                    if line_plots: deepest_tolerable_line_handle = line_plots[0]
            elif peak_R2_line_handle: # If same, update label of peak line to include this info or just let it be
                    current_label = peak_R2_line_handle.get_label()
                    peak_R2_line_handle.set_label(f"{current_label}\n(Also Deepest Tolerable)")
                    print(f"INFO: Peak R2 line is also the Deepest Tolerable R2 line for {current_file_name_for_output}")

            # --- Text Annotation for Highlighted Fits (on Plot) ---

            # Ensure text_x_pos is defined
            if len(x_for_scatter) > 0:
                plot_x_min_s, plot_x_max_s = np.min(x_for_scatter), np.max(x_for_scatter)
                text_x_pos = 0.1 # Your current hardcoded x-position (adjust as needed)
            else:
                text_x_pos = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05

            # Get current y-axis limits
            current_ymin_plot_extent, current_ymax_plot_extent = ax.get_ylim()

            # Determine y position and vertical alignment for the entire block
            if ax.get_yaxis().get_inverted():
                text_y_base = current_ymin_plot_extent + (current_ymax_plot_extent - current_ymin_plot_extent) * 0.02
                vertical_align = 'bottom'
            else:
                text_y_base = current_ymax_plot_extent - (current_ymax_plot_extent - current_ymin_plot_extent) * 0.02
                vertical_align = 'top'

            # --- MODIFICATION FOR SINGLE TEXT BOX (ALL SAME COLOR) ---

            final_annotation_string = []
            display_color = 'k' # Default to black, or choose 'r', 'blue', etc. for the whole box

            # --- Peak R^2 Fit Details ---
            if peak_R2_iteration_details is not None and peak_R2_iteration_details['params'] is not None:
                m_p, b_p = peak_R2_iteration_details['params'][0], peak_R2_iteration_details['params'][1]
                eq_p = f"y = {m_p:.2f}x + {b_p:.2f}"
                
                final_annotation_string.append("Peak R² Fit:")
                final_annotation_string.append(f"  Range: {peak_R2_iteration_details['min_depth_regr_setting']:.2f}-{peak_R2_iteration_details['depth_limit']:.2f} m")
                #final_annotation_string.append(f"  Equation: {eq_p}")
                final_annotation_string.append(f"  R² = {peak_R2_iteration_details['R2']:.2f}, RMSE = {rmse_for_peak_R2_fit:.2f}")
            
            # Deepest Tolerable R^2 Fit Details (if distinct)
            is_deepest_distinct = False
            if deepest_tolerable_iteration_details is not None and \
               deepest_tolerable_iteration_details['params'] is not None:
                if peak_R2_iteration_details is None or \
                   not (deepest_tolerable_iteration_details['original_k_index'] == peak_R2_iteration_details['original_k_index'] and
                        np.isclose(deepest_tolerable_iteration_details['min_depth_regr_setting'], peak_R2_iteration_details['min_depth_regr_setting'])):
                    is_deepest_distinct = True

            if is_deepest_distinct:
                m_d, b_d = deepest_tolerable_iteration_details['params'][0], deepest_tolerable_iteration_details['params'][1]
                eq_d = f"y = {m_d:.2f}x + {b_d:.2f}"
                
                # Add a blank line for visual separation if both exist
                if final_annotation_string: # Check if there's content from Peak R2
                    final_annotation_string.append("") 

                final_annotation_string.append("Deepest Tolerable R² Fit:")
                final_annotation_string.append(f"  Range: {deepest_tolerable_iteration_details['min_depth_regr_setting']:.2f}-{deepest_tolerable_iteration_details['depth_limit']:.2f} m")
                #final_annotation_string.append(f"  Equation: {eq_d}")
                final_annotation_string.append(f"  R² = {deepest_tolerable_iteration_details['R2']:.2f}, RMSE = {rmse_for_deepest_tolerable_fit:.2f}")
            elif peak_R2_line_handle: # If same, update label of peak line
                current_label = peak_R2_line_handle.get_label()
                if "(Also Deepest Tolerable)" not in current_label:
                    peak_R2_line_handle.set_label(f"{current_label}\n(Also Deepest Tolerable)")
                print(f"INFO: Peak R2 line is also the Deepest Tolerable R2 line for {current_file_name_for_output}")

            # If there's any text to display, create the single annotation
            if final_annotation_string:
                ax.text(text_x_pos, text_y_base, "\n".join(final_annotation_string),
                        color=display_color, fontsize=8, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.8),
                        verticalalignment=vertical_align)


        # Legend
        handles_for_legend = []
        if scatter_handle and hasattr(scatter_handle, 'get_offsets') and len(scatter_handle.get_offsets()) > 0 :
            handles_for_legend.append(scatter_handle)
        if peak_R2_line_handle: # Add only if it was plotted
            handles_for_legend.append(peak_R2_line_handle)
        if deepest_tolerable_line_handle: # Add only if it was plotted and is distinct
              # Check if it's truly a different handle than peak_R2_line_handle
            if peak_R2_line_handle is None or (deepest_tolerable_line_handle != peak_R2_line_handle):
                handles_for_legend.append(deepest_tolerable_line_handle)
            elif peak_R2_line_handle and (deepest_tolerable_iteration_details is not None and peak_R2_iteration_details is not None and \
                  deepest_tolerable_iteration_details['original_k_index'] == peak_R2_iteration_details['original_k_index'] and \
                  np.isclose(deepest_tolerable_iteration_details['min_depth_regr_setting'], peak_R2_iteration_details['min_depth_regr_setting'])):
                  # If they are the same and label was updated, we don't need a duplicate handle
                  pass


        if handles_for_legend:
            # Remove duplicate handles if any (e.g. if peak and deepest were the same and plotted with same object but updated label)
            unique_handles = []
            seen_labels = set()
            for h in handles_for_legend:
                label = h.get_label()
                if label not in seen_labels:
                    unique_handles.append(h)
                    seen_labels.add(label)
            ax.legend(handles=unique_handles, loc='best', fontsize=8)

        # Get the limits that matplotlib automatically decided on after all plotting
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()

        # Find the largest maximum value from either axis
        max_limit = max(current_xlim[1], current_ylim[1])

        # Set both axes to have the same limits, from 0 to the new max_limit
        ax.set_xlim(0, max_limit)
        ax.set_ylim(0, max_limit)

        # This ensures the plot is visually square (1 unit on x-axis = 1 unit on y-axis)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        
        plot_filename = f"{just_the_filename_for_output_csv}_plot_dual_highlight.png" # New filename
        plot_save_path = os.path.join(output_save_folder_path, plot_filename)
        
        if fig:
            plt.savefig(plot_save_path)
            print(f"Plot for {current_file_name_for_output} saved to: {plot_save_path}")
            #plt.show() 
            plt.close(fig)

        # --- Save Iteration Data for the Current File to its own CSV ---
        if current_file_iterations_data:
            file_summary_df = pd.DataFrame(current_file_iterations_data)

            # --- Add Indicator Column Logic ---
            # Set all to 0 initially as per initialization in current_file_iterations_data.append
            # Then update based on conditions.
            if peak_R2_iteration_details is not None:
                # Find the row corresponding to the peak R^2 line and set Indicator to 1
                peak_R2_mask = (file_summary_df['Min Depth Range'] == peak_R2_iteration_details['min_depth_regr_setting']) & \
                               (file_summary_df['Max Depth Range'] == peak_R2_iteration_details['depth_limit']) & \
                               (file_summary_df['R2 Value'] == peak_R2_iteration_details['R2'])
                
                # Check if peak_R2_iteration_details refers to a row in current_file_iterations_data before assigning
                # This handles cases where positive_slope_results_df might be empty or peak_R2_iteration_details might be from a different source
                if not peak_R2_mask.empty and peak_R2_mask.any():
                    file_summary_df.loc[peak_R2_mask, 'Indicator'] = 1

            if deepest_tolerable_iteration_details is not None:
                # Find the row corresponding to the deepest tolerable R^2 line
                deepest_tolerable_mask = (file_summary_df['Min Depth Range'] == deepest_tolerable_iteration_details['min_depth_regr_setting']) & \
                                         (file_summary_df['Max Depth Range'] == deepest_tolerable_iteration_details['depth_limit']) & \
                                         (file_summary_df['R2 Value'] == deepest_tolerable_iteration_details['R2'])
                
                # Check if deepest_tolerable_iteration_details refers to a row in current_file_iterations_data
                if not deepest_tolerable_mask.empty and deepest_tolerable_mask.any():
                    # Check if it's the same line as the peak R^2 line
                    is_same_line = False
                    if peak_R2_iteration_details is not None:
                        is_same_line = (deepest_tolerable_iteration_details['original_k_index'] == peak_R2_iteration_details['original_k_index'] and \
                                        np.isclose(deepest_tolerable_iteration_details['min_depth_regr_setting'], peak_R2_iteration_details['min_depth_regr_setting']))
                    
                    if is_same_line:
                        # If only one regression line, mark as 2
                        file_summary_df.loc[deepest_tolerable_mask, 'Indicator'] = 2
                    else:
                        # If different, mark as 2 (deepest tolerable)
                        file_summary_df.loc[deepest_tolerable_mask, 'Indicator'] = 2

            output_csv_filename = f"{just_the_filename_for_output_csv}_LR_Stats_iterations.csv" # Slightly different name
            output_csv_save_path = os.path.join(output_save_folder_path, output_csv_filename)
            file_summary_df.to_csv(output_csv_save_path, index=False, float_format='%.4f')
            print(f"Iterations summary for {current_file_name_for_output} saved to: {output_csv_save_path}")
        else:
            print(f"No iteration data to save for {current_file_name_for_output}.")

    except Exception as e_outer:
        print(f"An unexpected error occurred while processing file {data_name_full_path}: {e_outer}")
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
        continue

print("\n--- All CSV files processed. ---")










