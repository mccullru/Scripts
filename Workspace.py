# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:25:38 2025

@author: mccullru
"""

import os
import glob
import rasterio
import numpy as np
import re
import pandas as pd

from pyproj import CRS, Transformer
from difflib import get_close_matches



#################################################################################################################
#################################################################################################################
""" paired t-test for max depth analysis"""


import pandas as pd
from scipy.stats import ttest_rel

# =============================================================================
# --- Configuration ---
# =============================================================================

# 1. Define the full path to your two data files
# This script assumes you have two separate CSVs, one for SDBred and one for SDBgreen
sdb_red_file = r"B:\Thesis Project\Stats_Figures_Results\Max Depth Analysis\statistical_analysis\analysis_summary_SDBred.csv"   # <-- IMPORTANT: UPDATE THIS PATH
sdb_green_file = r"B:\Thesis Project\Stats_Figures_Results\Max Depth Analysis\statistical_analysis\analysis_summary_SDBgreen.csv" # <-- IMPORTANT: UPDATE THIS PATH

# 2. Set the significance level (alpha)
alpha = 0.05

def perform_sensor_comparison_test(filepath, sdb_type):
    """
    Reads a CSV and performs a paired t-test between the Sentinel-2 and SuperDove
    depth retrieval columns to see if their means are significantly different.
    """
    print(f"--- Comparing SD vs S2 for {sdb_type} ---")
    
    try:
        df = pd.read_csv(filepath)
        
        required_columns = ['mean_Sentinel2', 'mean_SuperDove']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Required columns not found in {filepath}.")
            return

        # Drop rows where a NaN value exists in EITHER column to keep pairs aligned
        df_cleaned = df[required_columns].dropna()
        
        if len(df_cleaned) < 2:
             print("Fewer than 2 complete data pairs remain. Cannot perform test.")
             return

        # Extract the two columns to be compared
        sentinel2_depths = df_cleaned['mean_Sentinel2']
        superdove_depths = df_cleaned['mean_SuperDove']
        
        # --- ADDED: Calculate average depth differences ---
        differences = superdove_depths - sentinel2_depths
        avg_difference = differences.mean()
        std_dev_difference = differences.std()
        min_difference = differences.min()
        max_difference = differences.max()
        
        
        # --- Run the paired t-test ---
        t_statistic, p_value = ttest_rel(sentinel2_depths, superdove_depths)

        # --- Print the results and interpretation ---
        print(f"Number of paired AOIs: {len(df_cleaned)}")
        # --- ADDED: Print the difference stats ---
        print(f"Average Difference (SuperDove - Sentinel-2): {avg_difference:.4f} m")
        print(f"Std Dev of Differences: {std_dev_difference:.4f} m")
        print(f"Min of Differences: {min_difference:.2f} m")
        print(f"Max of Differences: {max_difference:.2f} m")
        print("\n--- Paired T-Test Results ---")
        print(f"T-statistic: {t_statistic:.4f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < alpha:
            print("Conclusion: There is a statistically significant difference between the sensors for this SDB type.")
        else:
            print("Conclusion: There is NOT a statistically significant difference between the sensors for this SDB type.")

    except FileNotFoundError:
        print(f"Error: Data file not found at '{filepath}'")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- Run the analysis for both SDBred and SDBgreen files ---
if __name__ == "__main__":
    perform_sensor_comparison_test(sdb_red_file, "SDBred")
    print("-" * 60)
    perform_sensor_comparison_test(sdb_green_file, "SDBgreen")



#################################################################################################################
#################################################################################################################

""" Only saves Slide Rule csv rows where confidence is greater than 0.70"""


# def filter_csvs_by_confidence(input_folder_path):
#     """
    

#     Parameters
#     ----------
#     input_folder_path : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     """
#     Processes all CSV files in a given folder. For each file, it filters rows
#     where the 'confidence' column is >= 0.7 and saves the result to a new
#     CSV file in the same folder with '_filtered' appended to the original name.

#     Args:
#         input_folder_path (str): The path to the folder containing CSV files.
#     """
#     # Check if the input folder exists
#     if not os.path.isdir(input_folder_path):
#         print(f"Error: Input folder not found at '{input_folder_path}'")
#         return

#     # Find all CSV files in the specified folder
#     csv_files = glob.glob(os.path.join(input_folder_path, "*.csv"))

#     if not csv_files:
#         print(f"No CSV files found in '{input_folder_path}'.")
#         return

#     print(f"Found {len(csv_files)} CSV files to process in '{input_folder_path}'.\n")

#     for file_path in csv_files:
#         try:
#             # Get the base name of the file and its extension
#             directory = os.path.dirname(file_path)
#             filename = os.path.basename(file_path)
#             name_part, ext_part = os.path.splitext(filename)

#             # Skip already processed files to avoid re-processing if script is run multiple times
#             if name_part.endswith("_filtered"):
#                 print(f"Skipping already filtered file: {filename}")
#                 continue
            
#             print(f"Processing file: {filename}...")

#             # Read the CSV file
#             df = pd.read_csv(file_path)

#             # Check if the 'confidence' column exists
#             if 'confidence' not in df.columns:
#                 print(f"  Warning: 'confidence' column not found in {filename}. Skipping this file.")
#                 continue

#             # Check if 'confidence' column is numeric, attempt conversion if not
#             if not pd.api.types.is_numeric_dtype(df['confidence']):
#                 print(f"  Info: 'confidence' column in {filename} is not numeric. Attempting conversion...")
#                 try:
#                     df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
#                 except Exception as e:
#                     print(f"  Warning: Could not convert 'confidence' column to numeric in {filename}. Error: {e}. Skipping this file.")
#                     continue
            
#             # Drop rows where 'confidence' could not be converted to numeric (became NaN)
#             df.dropna(subset=['confidence'], inplace=True)

#             # Filter rows where 'confidence' is >= 0.7
#             # The problem statement columns 'ortho_h', 'lat_ph', 'lon_ph' are implicitly kept if they exist,
#             # as we are only filtering rows, not selecting specific columns.
#             filtered_df = df[df['confidence'] >= 0.7].copy() # Use .copy() to avoid SettingWithCopyWarning

#             if filtered_df.empty:
#                 print(f"  No rows met the confidence threshold (>= 0.7) in {filename}. No output file created.")
#             else:
#                 # Construct the output file name and path
#                 output_filename = f"{name_part}_filtered{ext_part}"
#                 output_filepath = os.path.join(directory, output_filename)

#                 # Save the filtered DataFrame to a new CSV file
#                 filtered_df.to_csv(output_filepath, index=False)
#                 print(f"  Successfully filtered {filename}. Output saved to: {output_filename}")
#                 print(f"  Original rows: {len(df)}, Filtered rows: {len(filtered_df)}")

#         except pd.errors.EmptyDataError:
#             print(f"  Warning: {filename} is empty. Skipping this file.")
#         except Exception as e:
#             print(f"  An error occurred while processing {filename}: {e}")
#         finally:
#             print("-" * 30) # Separator for better readability

#     print("\nAll CSV files processed.")

# if __name__ == "__main__":

    
#     folder_to_process = r"B:\Thesis Project\Reference Data\Full_SlideRule_ICESat"

#     if folder_to_process == "your_csv_folder_path_here":
#         print("Please update the 'folder_to_process' variable in the script with the actual path to your CSV folder.")
#     else:
#         filter_csvs_by_confidence(folder_to_process)


#################################################################################################################
#################################################################################################################

""" Flags pSDB that has R^2 values below a set threshold, saves them to csv file """

#input_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results"

#output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results\Flagged Results"


# def process_r2_with_indicator_fallback(input_folder_path, output_csv_path,
#                                        r2_threshold=0.6,
#                                        indicator_col="Indicator",
#                                        r2_col="R2 Value",
#                                        image_name_col="Image Name",
#                                        flag_col_name="R2_Threshold_Outcome_Flag"):
#     """
#     Processes CSV files to evaluate R2 values based on an 'Indicator' column.
#     Prioritizes Indicator=2 rows. If an Indicator=2 row's R2 (rounded) meets
#     the threshold, its R2 is used and flag is 1.
#     If Indicator=2 row's R2 fails, it checks the corresponding Indicator=1 row.
#     If Indicator=1 row's R2 (rounded) meets threshold, its R2 is used and flag is 1.
#     Otherwise, the Indicator=2 row's R2 is used and flag is 0 (if R2 was valid) or NaN.
#     Only outputs data for Image Names that have an Indicator=2 row.

#     Args:
#         input_folder_path (str): Path to the folder containing input CSV files.
#         output_csv_path (str): Full path for the output CSV file.
#         r2_threshold (float, optional): R2 threshold. Defaults to 0.6.
#         indicator_col (str, optional): Name of the Indicator column. Defaults to "Indicator".
#         r2_col (str, optional): Name of the R2 Value column. Defaults to "R2 Value".
#         image_name_col (str, optional): Name of the Image Name column. Defaults to "Image Name".
#         flag_col_name (str, optional): Name for the new flag column.
#                                         Defaults to "R2_Threshold_Outcome_Flag".

#     Returns:
#         bool: True if successful, False otherwise.
#     """
#     results_to_save = []
#     output_column_names = [image_name_col, r2_col, flag_col_name]

#     if not os.path.isdir(input_folder_path):
#         print(f"Error: Input folder not found at '{input_folder_path}'")
#         return False

#     csv_files = glob.glob(os.path.join(input_folder_path, "*.csv"))

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
#         r2_threshold=0.7,   # R2 >= 0.7 is considered "meeting threshold"
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
#                  print(f"\nOutput file '{full_output_csv_path}' is empty (pd.errors.EmptyDataError).")
#             except Exception as e:
#                 print(f"Could not read or display output CSV: {e}")
#     else:
#         print("\nProcess encountered an error.")





#################################################################################################################
#################################################################################################################

# # --- Helper function to extract Rrs wavelength ---


# def extract_rrs_number(file_name):
#     """Extract the number after 'Rrs_' or 'Rrs' in the file name."""
#     match = re.search(r'Rrs_(\d+)', file_name)
#     if match:
#         return int(match.group(1))
#     match_alt = re.search(r'Rrs(\d+)', file_name)
#     if match_alt:
#         return int(match_alt.group(1))
#     return None

# def is_close_to(value, target, tolerance):
#     """Check if a value is close to a target within a tolerance."""
#     if value is None:
#         return False
#     return abs(value - target) <= tolerance

# # --- MODIFICATION: More robust Scene ID extraction ---
# def extract_core_scene_id(filename):
#     """
#     Extracts the core scene identifier from various filename patterns.
#     Target ID Example: S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W
#     """
#     # Try pattern for typical Sentinel-2 L2A product names (often used as a base by ACOLITE)
#     # This captures the part before typical ACOLITE suffixes like _L2W_Rrs_XXX or just _L2W.tif
#     # S2B_MSI_YYYYMMDDTHHMMSS_NXXXX_RXXX_TXXXXX_YYYYMMDDTHHMMSS (standard Sentinel naming)
#     # Or potentially S2B_MSI_YYYY_MM_DD_HH_MM_SS_TILEID_L2W (ACOLITE might keep this part)

#     # Common Sentinel-2 base product name structure (up to processing baseline and tile ID)
#     # e.g. S2B_MSI_20230128T070609_N0509_R110_T39PYP_20230128T085300
#     # Acolite filenames might be like: S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W_Rrs_XXX.tif
#     # Your SDB files are like: S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W__RGB_ODWmasked_SDBgreen.tif

#     # Let's try to capture the core part: S2B_MSI_YYYY_MM_DD_HH_MM_SS_TILEID_L2W
#     # The L2W seems to be the common end part before ACOLITE/your suffixes.
#     # Your SDB identifier also has an extra underscore after L2W.

#     # Regex to capture the part like 'S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W'
#     # This pattern looks for the S2 prefix, MSI, date, time, TileID, and ends with L2W
#     core_id_match = re.match(r'(S2[AB]_MSI_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_T\d{2}[A-Z]{3}_L2W)', filename)
#     if core_id_match:
#         return core_id_match.group(1)

#     # Fallback if the above specific pattern isn't perfect for all cases
#     # Try to strip known suffixes more generally.
#     # This was your red-edge extracted ID: 'S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W'
#     # This was your SDB extracted ID:    'S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W_' (with extra _)

#     # Let's try to match up to L2W and then clean up potential trailing underscores from SDB files
#     base_name_match = re.match(r'(S2[AB]_MSI_.*?_L2W)', filename) # Non-greedy match until L2W
#     if base_name_match:
#         potential_id = base_name_match.group(1)
#         # print(f"DEBUG extract_core_scene_id - Matched up to L2W: {potential_id} from {filename}")
#         return potential_id # This should give 'S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W' for both

#     # If the above fails, revert to a more general stripping (less reliable)
#     print(f"DEBUG extract_core_scene_id - Using fallback for: {filename}")
#     base = os.path.splitext(filename)[0]
#     # List of suffixes to remove, order might matter if some are subsets of others
#     suffixes_to_remove = [
#         "_RGB_ODWmasked_SDBgreen", "_RGB_ODWmasked_SDBred", "_RGB_ODWmasked_SDB_merged",
#         "_SDBgreen", "_SDBred", "_SDB_merged",
#         "_ODWmaskedRE", "_ODWmasked", "_RGB",
#         "_Rrs_704", "_Rrs_666", "_Rrs_560", "_Rrs_492", # Add other Rrs bands if needed
#         "_Rrs704", "_Rrs666", "_Rrs560", "_Rrs492"
#     ]
#     for suffix in suffixes_to_remove:
#         if base.endswith(suffix):
#             base = base[:-len(suffix)]
#             break # Remove one suffix and assume that's enough for this fallback
#     return base.rstrip('_') # Remove any trailing underscores just in case
# # --- END MODIFICATION ---


# # --- Main Function (mask_sdb_with_red_edge) ---
# # ... (The rest of your mask_sdb_with_red_edge function remains the same,
# #      it will now use the refined extract_core_scene_id) ...
# # For brevity, I'll just show where it's called:

# def mask_sdb_with_red_edge(input_sdb_folder, input_multi_band_folder, output_folder_sdb):
#     print(f"Input SDB Folder: {input_sdb_folder}")
#     print(f"Input Multi-Band (Red-Edge source) Folder: {input_multi_band_folder}")
#     print(f"Output SDB (Masked) Folder: {output_folder_sdb}")

#     if not os.path.exists(output_folder_sdb):
#         os.makedirs(output_folder_sdb)
#         print(f"Created output folder: {output_folder_sdb}")

#     sdb_files_paths = glob.glob(os.path.join(input_sdb_folder, "*.tif"))

#     red_edge_candidates = {}
#     all_band_files = glob.glob(os.path.join(input_multi_band_folder, "*.tif"))

#     print(f"DEBUG: Scanning Red-Edge folder: {input_multi_band_folder}")
#     for band_file_path in all_band_files:
#         band_filename = os.path.basename(band_file_path)
#         rrs_wavelength = extract_rrs_number(band_filename)
#         if is_close_to(rrs_wavelength, 704, 5):
#             scene_id = extract_core_scene_id(band_filename) # USE REFINED FUNCTION
#             if scene_id:
#                 red_edge_candidates[scene_id] = band_file_path
#                 print(f"  DEBUG: Found potential Red-Edge: ID='{scene_id}', File='{band_filename}'")
#             else:
#                 print(f"  DEBUG: Could not extract scene_id from Red-Edge candidate: {band_filename}")

#     if not sdb_files_paths: # ... (rest of your checks and main loop) ...
#         print("No SDB TIFF files found in the input SDB folder.")
#         return
#     if not red_edge_candidates:
#         print("No potential Red-Edge (Rrs~704nm) TIFF files identified and mapped in the multi-band folder.")
#         return

#     print(f"Found {len(sdb_files_paths)} SDB files.")
#     print(f"Identified {len(red_edge_candidates)} unique Rrs~704nm Red-Edge files by Scene ID.")

#     for sdb_file_path in sdb_files_paths:
#         sdb_filename = os.path.basename(sdb_file_path)
#         sdb_identifier = extract_core_scene_id(sdb_filename) # USE REFINED FUNCTION

#         print(f"\n--- Processing SDB file: {sdb_filename} (Attempting to match ID: '{sdb_identifier}') ---")
#         corresponding_red_edge_file = red_edge_candidates.get(sdb_identifier)

#         if not corresponding_red_edge_file:
#             print(f"  Corresponding Red-Edge (Rrs~704nm) file not found for SDB identifier: '{sdb_identifier}'. Skipping this SDB file.")
#             continue
#         # ... (rest of the processing logic from your previous script) ...
#         print(f"  Using Red-Edge file: {os.path.basename(corresponding_red_edge_file)}")
#         try:
#             with rasterio.open(sdb_file_path) as sdb_src:
#                 sdb_array = sdb_src.read(1).astype(rasterio.float32)
#                 sdb_profile = sdb_src.profile
#                 sdb_nodata = sdb_src.nodata
#                 if sdb_nodata is not None:
#                     sdb_array[sdb_array == sdb_nodata] = np.nan
#             with rasterio.open(corresponding_red_edge_file) as re_src:
#                 red_edge_array = re_src.read(1).astype(rasterio.float32)
#                 re_nodata = re_src.nodata
#                 if re_nodata is not None:
#                     red_edge_array[red_edge_array == re_nodata] = np.nan
#             if sdb_array.shape != red_edge_array.shape:
#                 print(f"  ERROR: SDB and Red-Edge rasters for {sdb_identifier} do not have the same dimensions. Skipping.")
#                 continue
#             sdb_for_log = sdb_array.copy()
#             sdb_for_log[sdb_for_log <= 0] = np.nan
#             red_edge_for_log = red_edge_array.copy()
#             red_edge_for_log[red_edge_for_log <= 0] = np.nan
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 log_sdb_odw_threshold = -0.63 * np.log(red_edge_for_log) -2.56
#                 ln_sdb_array = np.log(sdb_for_log)
#             odw_condition = ln_sdb_array > log_sdb_odw_threshold
#             num_odw_pixels = np.nansum(odw_condition.astype(int))
#             print(f"  Identified {num_odw_pixels} ODW pixels to mask based on Red-Edge criteria.")
#             sdb_masked_array = sdb_array.copy()
#             sdb_masked_array[odw_condition] = np.nan
#             output_profile = sdb_profile.copy()
#             output_profile.update(dtype=rasterio.float32, nodata=np.nan, count=1)
#             sdb_filename_stem = os.path.splitext(sdb_filename)[0]
#             output_filename = f"{sdb_filename_stem}_m2.tif"
#             output_file_path = os.path.join(output_folder_sdb, output_filename)
#             with rasterio.open(output_file_path, 'w', **output_profile) as dst:
#                 dst.write(sdb_masked_array, 1)
#             print(f"  Saved masked SDB to: {output_file_path}")
#         except Exception as e:
#             print(f"  Error processing file pair for {sdb_filename}: {e}")
#             import traceback
#             traceback.print_exc()

# # --- Configuration ---
# input_red_edge_folder = r"E:\Thesis Stuff\AcoliteWithPython\Corrected_Imagery\odf_test_output\S2"
# input_SDB_folder = r"E:\Thesis Stuff\SDB"
# output_masked_SDB_folder = input_SDB_folder

# # --- Run the process ---
# mask_sdb_with_red_edge(input_SDB_folder, input_red_edge_folder, output_masked_SDB_folder)

# print("\n--- Script finished ---")


#################################################################################################################
#################################################################################################################

"""Takes ICESat files from Slide Rule and prepares them for SDB_Time (Changes to UTM, reorganizes)"""


# def process_and_transform_csvs(input_folder,
#                                  epsg_csv_path,
#                                  lat_col_in='lat_ph', lon_col_in='lon_ph', ortho_col_in='ortho_h',
#                                  new_northing_col='Northing', new_easting_col='Easting', new_ortho_col='Geoid_Corrected_Ortho_Height',
#                                  epsg_aoi_col='Name', epsg_code_col='EPSG code',
#                                  new_file_suffix='_processed'):
#     """
#     Processes CSV files:
#     1. Reads latitude, longitude, and orthometric height.
#     2. Matches filename to an EPSG code from a reference CSV using a multi-step logic.
#     3. Transforms coordinates to the matched UTM projection.
#     4. Renames columns for Northing, Easting, and corrected Ortho Height.
#     5. Reorganizes columns to place these three first.
#     6. Saves the processed data to a new CSV file with a specified suffix.
#     """
#     print("--- Starting CSV Processing and Transformation ---")
#     print(f"Input/Output Folder: {input_folder}")
#     print(f"EPSG CSV Path: {epsg_csv_path}")

#     if not os.path.exists(input_folder):
#         print(f"Error: Input folder not found: {input_folder}")
#         return

#     if not os.path.exists(epsg_csv_path):
#         print(f"Error: EPSG codes CSV file not found: {epsg_csv_path}")
#         return

#     try:
#         epsg_df = pd.read_csv(epsg_csv_path)
#         if epsg_aoi_col not in epsg_df.columns or epsg_code_col not in epsg_df.columns:
#             print(f"Error: EPSG CSV must contain '{epsg_aoi_col}' and '{epsg_code_col}' columns.")
#             return
        
#         epsg_map_full_names = epsg_df.set_index(epsg_aoi_col)[epsg_code_col].to_dict()
        
#         print(f"Loaded {len(epsg_map_full_names)} EPSG codes from {epsg_csv_path}")

#         simplified_epsg_names_for_matching = {}
#         for full_name_key in epsg_map_full_names.keys():
#             # Extract the part before "_ATL03_"
#             simplified_name_base = full_name_key.split('_ATL03_')[0]
#             # Normalize: replace underscores/hyphens with spaces, strip whitespace
#             temp_name = simplified_name_base.replace('_', ' ').replace('-', ' ').strip()
            
#             clean_for_matching_candidate = temp_name # Start with the fully normalized base

#             # Handle specific multi-word or uniquely formatted names as overrides
#             if "North_Fuerteventura" in full_name_key:
#                 clean_for_matching_candidate = "NorthFeut"
#             elif "St_Catherines_Bay" in full_name_key:
#                 clean_for_matching_candidate = "St Catherines"
#             elif "Bum_Bum_Island" in full_name_key:
#                 clean_for_matching_candidate = "Bum Bum"
#             # Add more elif conditions here for other specific names as needed
#             # Example: if "South_Portoscuso" in simplified_name_base: clean_for_matching_candidate = "South Portoscuso"

#             if clean_for_matching_candidate:
#                 simplified_epsg_names_for_matching[clean_for_matching_candidate] = full_name_key 
#             else:
#                 print(f"Warning: Could not reliably simplify EPSG name '{full_name_key}'. Skipping for matching.")
#                 continue
            
#         print(f"First 5 Simplified EPSG Names for Matching (Keys): {list(simplified_epsg_names_for_matching.keys())[:5]}")
#         print(f"First 5 Original Full EPSG Map Keys: {list(epsg_map_full_names.keys())[:5]}")

#     except Exception as e:
#         print(f"Error loading EPSG codes CSV: {e}")
#         return

#     csv_files = glob.glob(os.path.join(input_folder, "*_filtered.csv"))

#     if not csv_files:
#         print(f"No files ending with '_filtered.csv' found in: {input_folder}")
#         return

#     print(f"Found {len(csv_files)} CSV files to process.")

#     crs_wgs84 = CRS("EPSG:4326") 
#     matching_cutoff = 0.5 

#     for file_path in csv_files:
#         file_name = os.path.basename(file_path)
#         print(f"\n--- Processing file: {file_name} ---")

#         aoi_name_from_file = os.path.splitext(file_name)[0]
        
#         if aoi_name_from_file.lower().endswith('_sr_cal'):
#             aoi_name_from_file = aoi_name_from_file[:-len('_sr_cal')]
#         elif aoi_name_from_file.lower().endswith('_sr_acc'):
#             aoi_name_from_file = aoi_name_from_file[:-len('_sr_acc')]
#         elif aoi_name_from_file.lower().endswith('_sr'):
#             aoi_name_from_file = aoi_name_from_file[:-len('_sr')]
        
#         if aoi_name_from_file.lower().endswith('_filtered'):
#             aoi_name_from_file = aoi_name_from_file[:-len('_filtered')]
        
#         aoi_name_for_match_normalized = aoi_name_from_file.replace('_', ' ').strip()
        
#         print(f"  Cleaned input filename AOI for matching: '{aoi_name_from_file}' (Normalized for match: '{aoi_name_for_match_normalized}')")

#         target_epsg_code = None
#         best_match_original_full_key = None # To store the original key from epsg_map_full_names

#         # Attempt 1: Direct match with aoi_name_for_match_normalized (e.g., "Nait 1")
#         if aoi_name_for_match_normalized in simplified_epsg_names_for_matching:
#             best_match_original_full_key = simplified_epsg_names_for_matching[aoi_name_for_match_normalized]
#             target_epsg_code = epsg_map_full_names[best_match_original_full_key]
#             print(f"  Directly matched: Normalized Filename AOI '{aoi_name_for_match_normalized}' to Simplified EPSG Key '{aoi_name_for_match_normalized}'. Original EPSG Name: '{best_match_original_full_key}'. Code: {target_epsg_code}.")
#         else:
#             # Attempt 2: Try matching base name if aoi_name_for_match_normalized is like "Name 1", "Name 2"
#             parts = aoi_name_for_match_normalized.split(' ')
#             base_name_candidate = None
#             if len(parts) > 1 and parts[-1].isdigit():
#                 base_name_candidate = ' '.join(parts[:-1]) 
            
#             if base_name_candidate and base_name_candidate in simplified_epsg_names_for_matching:
#                 best_match_original_full_key = simplified_epsg_names_for_matching[base_name_candidate]
#                 target_epsg_code = epsg_map_full_names[best_match_original_full_key]
#                 print(f"  Directly matched: Base Name '{base_name_candidate}' (from '{aoi_name_for_match_normalized}') to Simplified EPSG Key '{base_name_candidate}'. Original EPSG Name: '{best_match_original_full_key}'. Code: {target_epsg_code}.")
#             else:
#                 # Attempt 3: Fuzzy matching with aoi_name_for_match_normalized
#                 fuzzy_matches = get_close_matches(aoi_name_for_match_normalized,
#                                                   list(simplified_epsg_names_for_matching.keys()),
#                                                   n=1, cutoff=matching_cutoff)
#                 if fuzzy_matches:
#                     simplified_matched_key = fuzzy_matches[0]
#                     best_match_original_full_key = simplified_epsg_names_for_matching[simplified_matched_key]
#                     target_epsg_code = epsg_map_full_names[best_match_original_full_key]
#                     print(f"  Fuzzy matched: Normalized Filename AOI '{aoi_name_for_match_normalized}' to Simplified EPSG Key '{simplified_matched_key}'. Original EPSG Name: '{best_match_original_full_key}'. Code: {target_epsg_code}.")
#                 # Attempt 4: Fuzzy matching with base_name_candidate (if it exists and previous fuzzy failed)
#                 elif base_name_candidate:
#                     fuzzy_matches_base = get_close_matches(base_name_candidate,
#                                                        list(simplified_epsg_names_for_matching.keys()),
#                                                        n=1, cutoff=matching_cutoff)
#                     if fuzzy_matches_base:
#                         simplified_matched_key_base = fuzzy_matches_base[0]
#                         best_match_original_full_key = simplified_epsg_names_for_matching[simplified_matched_key_base]
#                         target_epsg_code = epsg_map_full_names[best_match_original_full_key]
#                         print(f"  Fuzzy matched: Base Name '{base_name_candidate}' (from '{aoi_name_for_match_normalized}') to Simplified EPSG Key '{simplified_matched_key_base}'. Original EPSG Name: '{best_match_original_full_key}'. Code: {target_epsg_code}.")
        
#         if not target_epsg_code:
#             print(f"  Warning: No matching EPSG code found for AOI '{aoi_name_from_file}' (normalized: '{aoi_name_for_match_normalized}', base candidate: '{base_name_candidate if 'base_name_candidate' in locals() else 'N/A'}') in {file_name}. Skipping.")
#             continue
        
#         try:
#             target_epsg_code = int(target_epsg_code) 
#             crs_utm = CRS(f"EPSG:{target_epsg_code}")
#             transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
#         except Exception as e:
#             print(f"  Error: Invalid EPSG code '{target_epsg_code}' for {aoi_name_from_file}. Skipping {file_name}. Error: {e}")
#             continue

#         try:
#             df = pd.read_csv(file_path)
#             required_input_cols = [lon_col_in, lat_col_in, ortho_col_in] 
#             if not all(col in df.columns for col in required_input_cols):
#                 print(f"  Error: Missing one or more required input columns ({required_input_cols}) in {file_name}. Skipping.")
#                 print(f"  Available columns in {file_name}: {df.columns.tolist()}")
#                 continue

#             easting, northing = transformer.transform(df[lon_col_in].values, df[lat_col_in].values)
#             df[lon_col_in] = easting 
#             df[lat_col_in] = northing
#             print(f"  Coordinates in '{lon_col_in}' and '{lat_col_in}' transformed to UTM (EPSG:{target_epsg_code}).")

#             df = df.rename(columns={lon_col_in: new_easting_col, 
#                                     lat_col_in: new_northing_col, 
#                                     ortho_col_in: new_ortho_col})
#             print(f"  Columns renamed: '{lon_col_in}' to '{new_easting_col}', '{lat_col_in}' to '{new_northing_col}', '{ortho_col_in}' to '{new_ortho_col}'.")

#             fixed_order_cols = [new_easting_col, new_northing_col, new_ortho_col]
#             other_cols = [col for col in df.columns if col not in fixed_order_cols]
#             new_column_order = fixed_order_cols + other_cols
#             df = df[new_column_order]
#             print(f"  Columns reordered with {fixed_order_cols} placed first.")

#             base_name_for_output, ext = os.path.splitext(file_name) 
#             if base_name_for_output.lower().endswith('_filtered'):
#                 base_name_for_output = base_name_for_output[:-len('_filtered')]

#             new_file_name = f"{base_name_for_output}{new_file_suffix}{ext}"
#             output_file_path = os.path.join(input_folder, new_file_name)

#             df.to_csv(output_file_path, index=False)
#             print(f"  Processed and saved to new file: {output_file_path}")

#         except Exception as e:
#             print(f"  Error processing {file_name}: {e}")
#             continue

#     print("\n--- All specified CSV files processed. ---")

# # --- Example Usage ---
# input_folder = r"B:\Thesis Project\Reference Data\Full_SlideRule_ICESat\New folder" 
# epsg_csv_path = r"B:\Thesis Project\Reference Data\Refraction Correction\UTM_epsg_codes.csv"

# process_and_transform_csvs(
#     input_folder,
#     epsg_csv_path,
# )


#################################################################################################################
#################################################################################################################


""" More reference data csv filtering"""

# def filter_csv_columns(input_folder, output_folder, columns_to_keep):
#     """
#     Reads CSV files from an input folder, keeps only specified columns,
#     and saves the modified CSVs to an output folder.

#     Args:
#         input_folder (str): Path to the folder containing the input CSV files.
#         output_folder (str): Path to the folder where processed CSV files will be saved.
#         columns_to_keep (list): A list of column names to keep.
#     """
#     if not os.path.exists(input_folder): # Checks input_folder
#         print(f"Error: Input folder not found: {input_folder}")
#         return

#     # If input_folder and output_folder are the same, this check is fine.
#     # The folder will already exist. If it's a new output_folder, it will be created.
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         print(f"Created output folder: {output_folder}")

#     csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

#     if not csv_files:
#         print(f"No CSV files found in: {input_folder}")
#         return

#     print(f"Found {len(csv_files)} CSV files to process.")

#     for file_path in csv_files:
#         file_name = os.path.basename(file_path)
#         print(f"\n--- Processing file: {file_name} ---")

#         # Avoid re-processing already filtered files if script is run multiple times
#         if file_name.endswith("_filtered.csv"):
#             print(f"  Skipping already filtered file: {file_name}")
#             continue

#         try:
#             df = pd.read_csv(file_path)
#             print(f"  Original columns: {df.columns.tolist()}")

#             existing_cols_to_keep = [col for col in columns_to_keep if col in df.columns]

#             if not existing_cols_to_keep:
#                 print(f"  Warning: None of the specified columns to keep {columns_to_keep} were found in {file_name}. Skipping file.")
#                 continue
            
#             # Check if all desired columns are already the only columns
#             if len(existing_cols_to_keep) == len(df.columns) and all(col in df.columns for col in existing_cols_to_keep):
#                 print(f"  File {file_name} already contains only the specified columns. No changes needed, but saving with new name for consistency if output differs from input or to mark as processed.")
#                 # Decide if you still want to save a copy with "_filtered.csv" 
#                 # or skip saving entirely in this case. For now, it will save a copy.

#             df_filtered = df[existing_cols_to_keep]
#             print(f"  Columns kept: {df_filtered.columns.tolist()}")

#             output_file_name = f"{os.path.splitext(file_name)[0]}_filtered.csv"
#             output_file_path = os.path.join(output_folder, output_file_name)

#             # Safety check if input and output filenames become identical
#             # This shouldn't happen with "_filtered.csv" suffix unless original already had it
#             if os.path.abspath(file_path) == os.path.abspath(output_file_path):
#                 print(f"  Warning: Input and output file paths are identical ('{output_file_path}').") 
#                 # Add further logic here if you want to prevent this, e.g., by adding another suffix
#                 # or by having a separate output folder as originally designed.
#                 # For now, it will overwrite if the names somehow became identical.
#                 # However, the "_filtered.csv" suffix should prevent this.

#             df_filtered.to_csv(output_file_path, index=False)
#             print(f"  Saved filtered file to: {output_file_path}")

#         except pd.errors.EmptyDataError:
#             print(f"  Warning: File {file_name} is empty. Skipping.")
#         except Exception as e:
#             print(f"  Error processing {file_name}: {e}")

#     print("\n--- All CSV files processed. ---")

# # --- Configuration ---
# # Specify the folder containing your input CSV files AND where outputs will go
# shared_folder_path = r"B:\Thesis Project\Reference Data\Full_SlideRule_ICESat\New folder" # <--- !!! REPLACE THIS !!!


# # Define the exact column names you want to keep
# columns_to_keep = ["Easting", "Northing", "Geoid_Corrected_Ortho_Height"]


# # --- Run the script ---
# if __name__ == "__main__":
    
#     filter_csv_columns(shared_folder_path, shared_folder_path, columns_to_keep)





#################################################################################################################
#################################################################################################################

""" Simple optically deep finder [(Rrs_green)^2/(Rrs_blue)]"""

# import rasterio
# import numpy as np
# import os
# import glob # Added for finding files
# # import matplotlib.pyplot as plt # Keep if you re-add SHOW_PREVIEW functionality

# def calculate_osi(rgb_tif_path, output_osi_tif_path, green_band_idx=1, blue_band_idx=2):
#     """
#     Calculates the Optically Shallow Index (OSI) from an RGB TIFF image.
#     OSI = (Rrs_green)^2 / (Rrs_blue)

#     The band indices are 0-based (e.g., for a standard RGB TIFF where
#     Band 1=Red, Band 2=Green, Band 3=Blue in GIS software, you would use:
#     green_band_idx = 1 (for the second band)
#     blue_band_idx = 2 (for the third band)
#     which are the defaults for this function).

#     Parameters:
#     - rgb_tif_path (str): Path to the input RGB TIFF file.
#     - output_osi_tif_path (str): Path to save the output single-band OSI TIFF file.
#     - green_band_idx (int): 0-based index of the green band. Default is 1.
#     - blue_band_idx (int): 0-based index of the blue band. Default is 2.
    
#     Returns:
#     - bool: True if successful, False otherwise.
#     """
#     print(f"\n--- Processing file: {os.path.basename(rgb_tif_path)} ---")
#     try:
#         with rasterio.open(rgb_tif_path) as src:
#             # print(f"Reading input: {rgb_tif_path}") # Path is printed above

#             min_required_bands = max(green_band_idx, blue_band_idx) + 1
#             if src.count < min_required_bands:
#                 print( # Changed from raise to print error and return False
#                     f"  Error: Input TIFF '{os.path.basename(rgb_tif_path)}' has {src.count} band(s), but needs "
#                     f"{min_required_bands} for green (0-idx: {green_band_idx}) "
#                     f"and blue (0-idx: {blue_band_idx}). Skipping this file."
#                 )
#                 return False
            
#             profile = src.profile
#             # print(f"  Reading Green band (file band {green_band_idx + 1}) and Blue band (file band {blue_band_idx + 1}).")
#             rrs_green_raw = src.read(green_band_idx + 1)
#             rrs_blue_raw = src.read(blue_band_idx + 1)
#             rrs_green = rrs_green_raw.astype(np.float32)
#             rrs_blue = rrs_blue_raw.astype(np.float32)
#             source_nodata = src.nodata
#             if source_nodata is not None:
#                 # print(f"  Source NoData value: {source_nodata}. Converting to NaN.")
#                 rrs_green[rrs_green_raw == source_nodata] = np.nan
#                 rrs_blue[rrs_blue_raw == source_nodata] = np.nan
            
#             profile.update(
#                 dtype=rasterio.float32,
#                 count=1,
#                 driver='GTiff',
#                 nodata=np.nan,
#                 compress='lzw'
#             )
            
#             # print("  Calculating OSI = (Green^2) / Blue ...")
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 osi = (rrs_green**2) / rrs_blue
#             osi[np.isinf(osi)] = np.nan
            
#             # Ensure output directory for this specific file exists
#             current_output_dir = os.path.dirname(output_osi_tif_path)
#             if not os.path.exists(current_output_dir):
#                 os.makedirs(current_output_dir)
#                 print(f"  Created output directory: {current_output_dir}")

#             # print(f"  Writing OSI raster to: {output_osi_tif_path}")
#             with rasterio.open(output_osi_tif_path, 'w', **profile) as dst:
#                 dst.write(osi.astype(rasterio.float32), 1)
#             print(f"  Successfully created OSI: {os.path.basename(output_osi_tif_path)}")
#             return True

#     except FileNotFoundError: # Should not happen if glob found it
#         print(f"  Error: Input RGB TIFF file not found at '{rgb_tif_path}'. Skipping.")
#     except rasterio.errors.RasterioIOError as e:
#         print(f"  Rasterio I/O Error for '{os.path.basename(rgb_tif_path)}': {e}. Skipping.")
#     except ValueError as ve: # Catch ValueErrors from band check etc.
#         print(f"  ValueError for '{os.path.basename(rgb_tif_path)}': {ve}. Skipping.")
#     except Exception as e:
#         print(f"  An unexpected error occurred with '{os.path.basename(rgb_tif_path)}': {e}. Skipping.")
#         # import traceback # Uncomment for detailed debugging if needed
#         # traceback.print_exc()
#     return False

# if __name__ == '__main__':

#     # Define the INPUT FOLDER containing RGB TIFF files
#     rgb_input_folder_path = r"E:\Thesis Stuff\RGBCompositOutput\New folder" 
    
#     # Define the OUTPUT FOLDER for the single-band OSI TIFFs
#     osi_output_folder_path = r"E:\Thesis Stuff\RGBCompositOutput\New folder"

#     # Define the 0-based indices for the green and blue bands in your RGB TIFFs
#     GREEN_BAND_INDEX = 1  # Corresponds to the 2nd band in the file (e.g., typical RGB)
#     BLUE_BAND_INDEX = 2   # Corresponds to the 3rd band in the file (e.g., typical RGB)


#     # Create the main output folder if it doesn't exist
#     if not os.path.exists(osi_output_folder_path):
#         os.makedirs(osi_output_folder_path)
#         print(f"Created main output folder: {osi_output_folder_path}")

#     # Find all .tif files in the input folder
#     # You can add more extensions if needed, e.g., "*.tiff"
#     search_pattern = os.path.join(rgb_input_folder_path, "*.tif")
#     tiff_files = glob.glob(search_pattern)
    
#     # Also search for .tiff (case-insensitive for extension is harder with glob, often handled by checking both)
#     search_pattern_tiff = os.path.join(rgb_input_folder_path, "*.tiff")
#     tiff_files.extend(glob.glob(search_pattern_tiff))
#     tiff_files = list(set(tiff_files)) # Remove duplicates if both patterns match same files

#     if not tiff_files:
#         print(f"No TIFF files found in '{rgb_input_folder_path}'. Please check the path and file extensions.")
#     else:
#         print(f"Found {len(tiff_files)} TIFF files to process in: {rgb_input_folder_path}")
        
#         successful_count = 0
#         failed_count = 0

#         for current_rgb_tif_path in tiff_files:
#             # Construct the output path for the OSI TIFF
#             base_name = os.path.basename(current_rgb_tif_path)
#             file_name_part, file_ext_part = os.path.splitext(base_name)
            
#             # Create a descriptive output filename
#             output_file_name = f"{file_name_part}_OSI{file_ext_part}"
#             current_output_osi_path = os.path.join(osi_output_folder_path, output_file_name)

#             success = calculate_osi(current_rgb_tif_path, current_output_osi_path,
#                                     green_band_idx=GREEN_BAND_INDEX,
#                                     blue_band_idx=BLUE_BAND_INDEX)
#             if success:
#                 successful_count += 1
#                 # if SHOW_PREVIEW:
#                 #    # Add preview logic here if desired, similar to previous versions
#                 #    # e.g., open current_output_osi_path and plot with matplotlib
#                 #    pass 
#             else:
#                 failed_count += 1
        
#         print(f"\n--- Processing Summary ---")
#         print(f"Successfully processed: {successful_count} files.")
#         print(f"Failed to process: {failed_count} files.")
#         print(f"OSI outputs saved in: {osi_output_folder_path}")























































