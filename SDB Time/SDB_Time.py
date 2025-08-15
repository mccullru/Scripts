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
from sklearn.metrics import mean_squared_error, r2_score
from difflib import get_close_matches
from shapely.geometry import box

##############################################################################################################
##############################################################################################################

"""Composite R, G, B tiffs from Acolite into one RGB tiff"""

"""Inputs: Acolite R,G, and B tiffs
   Outputs: RGB composites in RGBCompositOutput folder"""


def extract_rrs_number(file_name):
    """Extract the band number after 'Rrs_' in the file name."""
    match = re.search(r'Rrs_(\d+)', file_name)
    if match:
        return int(match.group(1))
    return None  # Return None if 'Rrs_' is not found


def is_close_to(value, target, tolerance):
    return abs(value - target) <= tolerance


def combine_bands_to_rgb(input_folder, output_folder, config):
    
    target_wavelengths = config['target_wavelengths']
    tolerance = config['wavelength_tolerance']
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Recursively find all .tif files
    files = list(Path(input_folder).rglob("*.tif"))
    
    # Filter bands based on proximity to target wavelengths
    red_bands = sorted([f for f in files if (num := extract_rrs_number(f.name)) is not None and is_close_to(num, target_wavelengths['red'], tolerance)])
    green_bands = sorted([f for f in files if (num := extract_rrs_number(f.name)) is not None and is_close_to(num, target_wavelengths['green'], tolerance)])
    blue_bands = sorted([f for f in files if (num := extract_rrs_number(f.name)) is not None and is_close_to(num, target_wavelengths['blue'], tolerance)])

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


##############################################################################################################
##############################################################################################################

"""Optically Deep Finder (ODF) where blue/green values < 0.003 sr^-1 are omitted. Mostly effective with very clear
   and deep waters, sometimes doesn't remove any pixels"""

"""Inputs: RGB composites from RGBCompositOutput folder
   Outputs: optically deep masked RGB tiffs with _m1 suffix in RGBCompositOutput folder"""


def mask_optically_deep_water(input_rgb_folder, output_masked_folder, output_binary_mask_folder, config):

    """
    Processes RGB GeoTIFF files to mask optically deep water pixels.
    Pixels where either the green OR blue band value is <= threshold are set to NaN.
    Saves the masked RGB to a NEW FILE in output_masked_folder.
    Optionally saves a separate binary mask of the changed pixels.

    """
    # --- Unpack threshold from the config dictionary ---
    threshold = config['odw_threshold']
    
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


##############################################################################################################
##############################################################################################################

"""Create pSDB red and green"""

"""Inputs: optically deep masked RGB tiffs with _m1 suffix in RGBCompositOutput folder
   Outputs: pSDBred and pSDBgreen in the pSDB folder"""


def process_rgb_geotiffs(input_folder, output_folder, config):
    
    # Get file deletion setting from the config dictionary (should be false if you want to keep input files)
    delete_input_files = config['delete_intermediate_files']
    
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
                scale_factor = 1000 * np.pi
                scaled_red = red_band * scale_factor     
                scaled_green = green_band * scale_factor  
                scaled_blue = blue_band * scale_factor    

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


    # Optional deletion block (only runs if delete_intermediate_files = TRUE in main script)
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


##############################################################################################################
##############################################################################################################
"""Extract pSDB/SDB values at reference point locations"""

"""Inputs: pSDBred/SDBred, pSDBgreen/SDBgreen, and SDBmerged in the pSDB or SDB folder, AND reference data in 
           point form with the columns: easting, northing, and depth
   Outputs: pSDBred/SDBred, pSDBgreen/SDBgreen, and SDBmerged extracted points in either the pSDB_ExtratedPts 
            or SDB_extracted points folder"""


def extract_raster_values_optimized(points_csv_file, raster_folder, output_folder, points_type='Reference', apply_min_depth_filter=False):
    """
    Extracts raster values at specified locations.
    (Corrected version to filter out nodata values)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Loading and preparing {points_type} points from: {os.path.basename(points_csv_file)}")
    df = pd.read_csv(points_csv_file)
    easting_col, northing_col, elev_col = df.columns[0], df.columns[1], df.columns[2]
    
    # Ensures that all values in the elevation/depth column are positive
    df[elev_col] = pd.to_numeric(df[elev_col], errors='coerce')
    if (df[elev_col] < 0).any():
        df.loc[df[elev_col] < 0, elev_col] *= -1
    
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[easting_col], df[northing_col])
    )

    for raster_file in os.listdir(raster_folder):
        if raster_file.lower().endswith(".tif"):
            raster_path = os.path.join(raster_folder, raster_file)
            print(f"\nProcessing raster: {raster_file}")

            try:
                with rasterio.open(raster_path) as src:
                    if gdf.crs != src.crs:
                        gdf.set_crs(src.crs, inplace=True)

                    gdf_in_bounds = gdf.clip(box(*src.bounds))

                    if gdf_in_bounds.empty:
                        print(f"No points overlap with raster: {raster_file}")
                        continue
                    
                    coords = [(p.x, p.y) for p in gdf_in_bounds.geometry]
                    
                    # Extract the raster values
                    raster_values = [val[0] for val in src.sample(coords)]

                    gdf_in_bounds = gdf_in_bounds.copy()
                    gdf_in_bounds.loc[:, 'Raster_Value'] = raster_values
                    
                    # 1. Get the nodata value directly from the raster file's metadata.
                    nodata_val_from_raster = src.nodata

                    # 2. If a nodata value exists, remove all rows with that value.
                    if nodata_val_from_raster is not None:
                        print(f"Filtering out NoData value: {nodata_val_from_raster}")
                        initial_count = len(gdf_in_bounds)
                        gdf_in_bounds = gdf_in_bounds[gdf_in_bounds['Raster_Value'] != nodata_val_from_raster]
                        print(f"Removed {initial_count - len(gdf_in_bounds)} NoData points.")

                    # 3. Also remove any other NaN values for safety.
                    gdf_in_bounds.dropna(subset=['Raster_Value'], inplace=True)


                    # --- Filter by depth < 40m ---
                    print("Applying depth filter (SDB & Reference < 40m)...")
                    initial_count_depth = len(gdf_in_bounds)
                    
                    # Keep rows where BOTH raster value AND reference elevation are less than 40
                    gdf_in_bounds = gdf_in_bounds[
                        (gdf_in_bounds['Raster_Value'] < 40) & (gdf_in_bounds[elev_col] < 40)
                    ]
                    
                    # Only apply the minimum depth filter if explicitly told to do so
                    if apply_min_depth_filter:
                        print("Applying minimum depth filter (>= 0m) for SDB points...")
                        gdf_in_bounds = gdf_in_bounds[
                            (gdf_in_bounds['Raster_Value'] >= 0) & (gdf_in_bounds[elev_col] >= 0)
                        ]
                    
                    removed_count = initial_count_depth - len(gdf_in_bounds)
                    print(f"Removed {removed_count} points with depths >= 40m.") 

                    if gdf_in_bounds.empty:
                        print("No valid data points remain after filtering. Skipping save.")
                        continue

                    output_filename = os.path.splitext(raster_file)[0] + "_extracted.csv"
                    output_path = os.path.join(output_folder, output_filename)
                    gdf_in_bounds.to_csv(output_path, index=False)
                    print(f"Results saved to: {output_path}")
            except Exception as e:
                print(f"ERROR processing {raster_file}: {e}")


##############################################################################################################
##############################################################################################################

"""Performs linear regressions to find two regression lines: 1) the line with the peak R^2 fit, and 2) the line
that has an R^2 value within 5% of the peak line, but includes the maximum number of points possible. It is
an iterative process that tries many different depth ranges until those two lines are found. """


"""Inputs: pSDBred/SDBred, pSDBgreen/SDBgreen, and SDBmerged extracted points in either the pSDB_ExtratedPts 
           or SDB_ExtractedPts folder
   Outputs: pngs of linear regression plots, and csv files with all linear regression iterations where the "Indicator"
            column has a 1 for peak R^2 line, and a 2 for maximum depth range with line within 5% of peak line
            in the pSDB_ExtractedPts_maxR2_results or SDB_ExtractedPts_maxR2_results"""


def perform_regression_analysis(input_folder, output_folder, plot_title_prefix, xlabel, is_validation_data=False):

    
    data_folder_path = input_folder
    output_save_folder_path = output_folder

    # the tolerable R^2 threshold that the second regression line is within from the peak R^2 threshold
    threshold_percent = 0.95
    print(f"\nR2 Tolerable threshold is within {threshold_percent*100}% of peak threshold")

    if not os.path.exists(output_save_folder_path):
        os.makedirs(output_save_folder_path)

    csv_files = glob(os.path.join(data_folder_path, "*.csv"))
    if not csv_files:
        print(f"Warning: No CSV files in {data_folder_path} for analysis."); return

    for data_name_full_path in csv_files:
        print(f"\n--- {plot_title_prefix}: Analyzing {os.path.basename(data_name_full_path)} ---")
        current_file_name_for_output = os.path.basename(data_name_full_path)
        just_the_filename_for_output_csv = os.path.splitext(current_file_name_for_output)[0]
        fig = None

        try:
            data_df = pd.read_csv(data_name_full_path)
            y_original = data_df["Geoid_Corrected_Ortho_Height"].values.astype(float)
            x_original = data_df["Raster_Value"].values.astype(float)
            nan_mask = ~ (np.isnan(x_original) | np.isnan(y_original))
            x_original, y_original = x_original[nan_mask], y_original[nan_mask]

            if x_original.size < 2:
                print("Fewer than 2 valid data points. Skipping analysis."); continue

            # Full Iterative Regression
            current_file_iterations_data = []
            depth_min_regr_sets = [1.0, 1.25, 1.5, 1.75, 2.0]
            step = 0.25
            overall_max_depth_data = np.max(y_original)

            for depth_min_limit_regr in depth_min_regr_sets:
                initial_depth_max = min(depth_min_limit_regr + step, overall_max_depth_data)
                
                if initial_depth_max <= depth_min_limit_regr:
                    depth_max_limits_to_test = np.array([overall_max_depth_data]) if overall_max_depth_data > depth_min_limit_regr else np.array([])
                
                else:
                    depth_max_limits_to_test = np.arange(initial_depth_max, overall_max_depth_data, step)
                    
                    # Ensure array is not empty before attempting to access [-1]
                    if depth_max_limits_to_test.size > 0 and not np.isclose(depth_max_limits_to_test[-1], overall_max_depth_data):
                        depth_max_limits_to_test = np.append(depth_max_limits_to_test, overall_max_depth_data)
                
                # This check is still necessary if the above logic results in an empty array
                # (e.g., if overall_max_depth_data is very small or equal to depth_min_limit_regr)
                depth_max_limits_to_test = np.unique(depth_max_limits_to_test[depth_max_limits_to_test >= depth_min_limit_regr])
                
                
                if len(depth_max_limits_to_test) == 0: continue

                for k_loop_idx, current_depth_max_limit in enumerate(depth_max_limits_to_test):
                    range_idx = (y_original >= depth_min_limit_regr) & (y_original <= current_depth_max_limit)
                    x_iter, y_iter = x_original[range_idx], y_original[range_idx]
                    num_points, m, b, r2, rmse, equation, params = len(x_iter), np.nan, np.nan, np.nan, np.nan, "N/A", None
                    if num_points > 1:
                        params = np.polyfit(x_iter, y_iter, 1); m, b = params[0], params[1]
                        y_fit = np.polyval(params, x_iter)
                        if len(np.unique(y_iter)) > 1: r2 = r2_score(y_iter, y_fit)
                        else: r2 = 1.0
                        if r2 < 0: r2 = 0.0
                        rmse = np.sqrt(mean_squared_error(y_iter, y_fit))
                        equation = f"y = {m:.4f}x + {b:.4f}"
                    current_file_iterations_data.append({'Image Name': current_file_name_for_output, 
                                                         'Min Depth Range': depth_min_limit_regr, 
                                                         'Max Depth Range': current_depth_max_limit, 
                                                         'R2 Value': r2, 'RMSE': rmse, 
                                                         'Line of Best Fit': equation, 
                                                         'm1': m, 'm0': b, 
                                                         'Pt Count': num_points, 
                                                         'Indicator': 0, 
                                                         'params': params})

            summary_df = pd.DataFrame(current_file_iterations_data)
            
            # Plotting Setup
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(x_original, y_original, s=36, c='k', alpha=0.3, label='Data Points')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Reference Depth (m)')
            ax.set_title(f'{plot_title_prefix}: {just_the_filename_for_output_csv.replace("_", " ")}')
            ax.grid(True)
            
            # Find Best-Fit Lines & Set Indicators
            positive_slope_df = summary_df[summary_df['m1'] > 0].copy()
            
            # Create a new DataFrame of only the eligible models for best-fit selection.
            # Exclude models where R^2 is 1, often an artifact of too few points which is why a 10 point minimum
            # is another condition
            eligible_fits_df = positive_slope_df[
                (positive_slope_df['R2 Value'] != 1.0) &
                (positive_slope_df['Pt Count'] > 10)
            ].copy()
            print(f"  Found {len(positive_slope_df)} positive slope models. {len(eligible_fits_df)} are eligible for selection.")

            
            # Peak R^2 fit will be indicated by a 1
            if not eligible_fits_df.empty and not eligible_fits_df['R2 Value'].isna().all():
                peak_idx = eligible_fits_df['R2 Value'].idxmax()
                peak_R2_details = eligible_fits_df.loc[peak_idx]
                summary_df.loc[peak_idx, 'Indicator'] = 1
                
                
                r2_threshold = peak_R2_details['R2 Value'] * threshold_percent
                tolerable_fits = eligible_fits_df[eligible_fits_df['R2 Value'] >= r2_threshold]
                
                # Deepest tolerable fit will be indicated by a 2
                if not tolerable_fits.empty:
                    deepest_idx = tolerable_fits['Max Depth Range'].idxmax()
                    deepest_details = tolerable_fits.loc[deepest_idx]
                    summary_df.loc[deepest_idx, 'Indicator'] = 2
                    
                    # Plot the calculated best-fit lines on the chart
                    x_fit_domain = np.array([np.min(x_original), np.max(x_original)])
                    if peak_R2_details['params'] is not None:
                        ax.plot(x_fit_domain, np.polyval(peak_R2_details['params'], x_fit_domain), 'r-', label="Peak R² Fit")
                    if not deepest_details.equals(peak_R2_details) and deepest_details['params'] is not None:
                        ax.plot(x_fit_domain, np.polyval(deepest_details['params'], x_fit_domain), 'b--', label="Deepest Fit")

            # --- Apply Special Formatting ONLY for SDB Validation Plot ---
            if is_validation_data:
                line_max = max(np.max(x_original), np.max(y_original)) * 1.1
                ax.plot([0, line_max], [0, line_max], color='gray', linestyle=':', linewidth=1.5, zorder=0)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(0, line_max)
                ax.set_ylim(0, line_max)


            # --- Build and add the detailed text box ---
            annotation_lines = []
            if peak_R2_details is not None:
                annotation_lines.append("Peak R² Fit:")
                annotation_lines.append(f"  Range: {peak_R2_details['Min Depth Range']:.2f} - {peak_R2_details['Max Depth Range']:.2f} m")
                annotation_lines.append(f"  R² = {peak_R2_details['R2 Value']:.2f}")
                annotation_lines.append(f"  RMSE = {peak_R2_details['RMSE']:.2f} m")

            if deepest_details is not None and not deepest_details.equals(peak_R2_details):
                annotation_lines.append("") # Add a blank line for separation
                annotation_lines.append("Deepest Tolerable Fit:")
                annotation_lines.append(f"  Range: {deepest_details['Min Depth Range']:.2f} - {deepest_details['Max Depth Range']:.2f} m")
                annotation_lines.append(f"  R² = {deepest_details['R2 Value']:.2f}")
                annotation_lines.append(f"  RMSE = {deepest_details['RMSE']:.2f} m")
            
            if annotation_lines:
                full_annotation_text = "\n".join(annotation_lines)
                ax.text(0.75, 0.05, full_annotation_text, transform=ax.transAxes, fontsize=12,
                        ha='left', va='bottom', bbox=dict(boxstyle='round', fc='white', alpha=0.8))


            # --- Save Outputs ---
            summary_df_to_save = summary_df.drop(columns=['params'])
            csv_path = os.path.join(output_save_folder_path, f"{just_the_filename_for_output_csv}_LR_Stats_iterations.csv")
            summary_df_to_save.to_csv(csv_path, index=False, float_format='%.4f')
            print(f"Iterations summary saved to: {csv_path}")

            ax.legend()
            plt.tight_layout()
            plot_filename = f"{just_the_filename_for_output_csv}_plot.png"
            plot_save_path = os.path.join(output_save_folder_path, plot_filename)
            plt.savefig(plot_save_path)
            plt.close(fig)
            print(f"Plot saved to: {plot_save_path}")

        except Exception as e:
            print(f"An error occurred while processing {data_name_full_path}: {e}")
            if fig: plt.close(fig)
            continue

##############################################################################################################
##############################################################################################################

"""Create SDB red and green with constants from linear regression lines"""


"""Inputs: csv files with all linear regression iterations where the "Indicator" column has a 1 for peak R^2 line, 
           and a 2 for maximum depth range with line within 10% of peak line in the pSDB_ExtractedPts_maxR2_results 
           or SDB_ExtractedPts_maxR2_results folder
   Outputs: SDBred and SDBgreen in SDB folder"""


def create_sdb_rasters(raster_folder, csv_folder, output_folder, config):
    
    # Get parameters from the config dictionary
    r2_filter_threshold = config['r2_threshold']
    
    # Define column names and nodata value from config for consistency, if needed,
    # or keep them as defaults like the original function. We'll use defaults for now.
    indicator_col = "Indicator"
    r2_col = "R2 Value"
    m1_col = "m1"
    m0_col = "m0"
    nodata_value = -9999
    apply_r2_filter = False
    
    
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Output SDB rasters will be saved to: {output_folder}")
    if apply_r2_filter:
        print(f"R2 filter is ON. Threshold for SDB creation: {r2_col} >= {r2_filter_threshold}")
    else:
        print("R2 filter is OFF. Using Deepest Tolerable fit (where Indicator=2) from matched stats CSV.")

    stats_csv_filenames_in_folder = [f for f in os.listdir(csv_folder) if f.lower().endswith('.csv')]
    if not stats_csv_filenames_in_folder:
        print(f"No CSV files found in stats folder {csv_folder}. Cannot proceed.")
        return

    print(f"Found {len(stats_csv_filenames_in_folder)} CSV files in stats folder: {csv_folder}")
    processed_rasters_count = 0
    processed_at_all_count = 0 # To see if any raster even starts processing

    for pSDB_raster_filename in os.listdir(raster_folder):
        processed_at_all_count +=1
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
                     print(f"    No Indicator 2 rows found in {matched_stats_csv_name} for R2 check.")

                if selected_row_from_stats is not None:
                    m1_to_use = selected_row_from_stats[m1_col]
                    m0_to_use = selected_row_from_stats[m0_col]
                    coeffs_were_selected = True
                else:
                    print(f"  No coefficients passed threshold for {pSDB_raster_filename}.")
            
            else: # R2 filter is OFF
                rows_ind2_no_filter = cleaned_stats_df[cleaned_stats_df[indicator_col] == 2]
                if not rows_ind2_no_filter.empty:
                    best_fit_row_no_filter = rows_ind2_no_filter.iloc[0]
                    m1_to_use = best_fit_row_no_filter[m1_col]
                    m0_to_use = best_fit_row_no_filter[m0_col]
                    coeffs_were_selected = True
                    chosen_coeffs_description = "Indicator 2"
                else:
                    print(f"  No Indicator 2 coefficients found in {matched_stats_csv_name}.")

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


##############################################################################################################
##############################################################################################################

"""Merge SDB red and green together"""

"""Inputs: SDBred and SDBgreen in SDB folder
   Outputs: SDBmerged in SDB folder """


def merge_sdb_raster(sdb_red, sdb_green, config):
    """
    Create a merged SDB raster with the following rules:
    - If 0 <= SDBred <= 2, use SDBred.
    - If 2 < SDBred <= 3.5, use a weighted average of SDBred and SDBgreen.
    - If SDBred > 3.5, use SDBgreen.
    """
    
    
    # Get merge limits from the config dictionary
    lower_limit = config['merge_lower_limit']
    upper_limit = config['merge_upper_limit']
    
    
    # Initialize output array
    sdb_merged = np.full_like(sdb_red, np.nan)

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


def process_sdb_folder(input_folder, config):
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
        sdb_merged = merge_sdb_raster(sdb_red, sdb_green, config)
        
        # Replace np.nan with NoData value (e.g., -9999)
        nodata_value = -9999
       
        
        # Save the new merged raster
        with rasterio.open(output_path, 'w', driver='GTiff', count=1, dtype=sdb_merged.dtype,
                            height=sdb_merged.shape[0], width=sdb_merged.shape[1],
                            crs=red_dataset.crs, transform=red_dataset.transform, nodata=nodata_value) as dst:
            dst.write(sdb_merged, 1)

        print(f"Saved merged SDB raster: {output_path}")


##############################################################################################################
##############################################################################################################

def run_full_pipeline(config):
    """
    Runs the entire raster processing pipeline from start to finish.
    """
    print("--- STARTING FULL RASTER PROCESSING PIPELINE ---")
    
    # Define and create output sub-folders
    folders = config['output_folders']
    
    for f in folders.values(): os.makedirs(f, exist_ok=True)

    # Execute Pipeline
    print("\n\n[Step 1/8] Combining single bands into RGB TIFFs...\n")
    combine_bands_to_rgb(config['raw_bands_folder'], 
                         folders['rgb_composites'], 
                         config)

    print("\n\n[Step 2/8] Masking Optically Deep Water...\n")
    mask_optically_deep_water(folders['rgb_composites'], 
                              folders['masked_odw'], 
                              None, 
                              config)

    print("\n\n[Step 3/8] Creating pSDB rasters...\n")
    process_rgb_geotiffs(folders['masked_odw'], 
                         folders['psdb_rasters'], 
                         config)

    print("\n\n[Step 4/8] Extracting pSDB values at Calibration points...\n")
    extract_raster_values_optimized(config['cal_points_csv'], 
                                    folders['psdb_rasters'], 
                                    folders['cal_extract'], 
                                    'Calibration')
    
    print("\n\n[Step 5/9] Performing pSDB regression for coefficients...\n")
    perform_regression_analysis(
        input_folder=folders['cal_extract'],
        output_folder=folders['regr_results'],
        plot_title_prefix="pSDB Regression",
        xlabel="pSDB Value",
        is_validation_data=False  # This will use auto-scaled axes
    )

    print("\n\n[Step 6/9] Applying coefficients to create final SDB rasters...\n")
    create_sdb_rasters(folders['psdb_rasters'], 
                       folders['regr_results'], 
                       folders['sdb_final'], 
                       config)

    print("\n\n[Step 7/9] Merging SDB Red and Green rasters...\n")
    process_sdb_folder(folders['sdb_final'], config)

    print("\n\n[Step 8/9] Extracting final SDB values at Validation points...\n")
    extract_raster_values_optimized(config['val_points_csv'], 
                                    folders['sdb_final'], 
                                    folders['val_extract'], 
                                    'Validation', 
                                    apply_min_depth_filter=True)

    print("\n\n[Step 9/9] Performing final SDB regression analysis...\n")
    perform_regression_analysis(
        input_folder=folders['val_extract'],
        output_folder=folders['val_analysis'],
        plot_title_prefix="SDB Validation",
        xlabel="SDB Derived Depth (m)",
        is_validation_data=True  # This will use the special 0,0 axis rules
    )

    print("\n--- SDB_TIME FINISHED ---")







