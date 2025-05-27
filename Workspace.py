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

# import os
# import pandas as pd
# from pyproj import CRS, Transformer
# from glob import glob
# from difflib import get_close_matches

# def process_and_transform_csvs(input_folder, 
#                                 epsg_csv_path, # This argument needs to be here
#                                 lat_col_in='lat_ph', lon_col_in='lon_ph', ortho_col_in='ortho_h',
#                                 new_northing_col='Northing', new_easting_col='Easting', new_ortho_col='Geoid_Corrected_Ortho_Height',
#                                 epsg_aoi_col='Name', epsg_code_col='EPSG code',
#                                 new_file_suffix='_processed'): # New parameter for suffix
   
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
#             simplified_name_base = full_name_key.split('_ATL03_')[0] 
#             temp_name = simplified_name_base.replace('_', ' ').replace('-', ' ').strip()
#             words = [word for word in temp_name.split(' ') if word]

#             clean_for_matching_candidate = ""
            
#             if words:
#                 clean_for_matching_candidate = words[0]

#             if len(words) > 1 and len(words[1]) > 1:
#                 two_word_candidate = f"{words[0]} {words[1]}"
#                 if 'portoscuso' in two_word_candidate.lower() or 'spigolo' in two_word_candidate.lower() \
#                    or 'fuerteventura' in two_word_candidate.lower() or 'norway' in two_word_candidate.lower():
#                     clean_for_matching_candidate = two_word_candidate
            
#             if "North_Fuerteventura" in full_name_key:
#                 clean_for_matching_candidate = "NorthFeut"
#             elif "St_Catherines_Bay" in full_name_key:
#                 clean_for_matching_candidate = "St Catherines"
#             elif "Bum_Bum_Island" in full_name_key:
#                  clean_for_matching_candidate = "Bum Bum"

#             if clean_for_matching_candidate:
#                  simplified_epsg_names_for_matching[clean_for_matching_candidate] = full_name_key 
#             else:
#                  print(f"Warning: Could not simplify EPSG name '{full_name_key}'. Skipping for matching.")
#                  continue
            
#         print(f"First 5 Simplified EPSG Names for Matching (Keys): {list(simplified_epsg_names_for_matching.keys())[:5]}")
#         print(f"First 5 Original Full EPSG Map Keys: {list(epsg_map_full_names.keys())[:5]}")

#     except Exception as e:
#         print(f"Error loading EPSG codes CSV: {e}")
#         return

#     csv_files = glob(os.path.join(input_folder, "*.csv"))

#     if not csv_files:
#         print(f"No CSV files found in: {input_folder}")
#         return

#     print(f"Found {len(csv_files)} CSV files to process.")

#     crs_wgs84 = CRS("EPSG:4326") 

#     matching_cutoff = 0.5 

#     for file_path in csv_files:
#         file_name = os.path.basename(file_path)
#         print(f"\n--- Processing file: {file_name} ---")

#         aoi_name_from_file = os.path.splitext(file_name)[0]
        
#         if aoi_name_from_file.lower().endswith('_sr_cal'):
#             aoi_name_from_file = aoi_name_from_file[:-7]
#         elif aoi_name_from_file.lower().endswith('_sr_acc'):
#             aoi_name_from_file = aoi_name_from_file[:-7]
#         elif aoi_name_from_file.lower().endswith('_sr'):
#             aoi_name_from_file = aoi_name_from_file[:-3]
        
#         aoi_name_for_match_normalized = aoi_name_from_file.replace('_', ' ').strip()
        
#         print(f"  Cleaned input filename AOI for matching: '{aoi_name_from_file}' (Normalized for match: '{aoi_name_for_match_normalized}')")

#         target_epsg_code = None
        
#         if aoi_name_for_match_normalized in simplified_epsg_names_for_matching:
#             best_match_original_full_key = simplified_epsg_names_for_matching[aoi_name_for_match_normalized]
#             target_epsg_code = epsg_map_full_names[best_match_original_full_key]
#             print(f"  Directly matched normalized file AOI to simplified EPSG name: '{aoi_name_for_match_normalized}' -> '{best_match_original_full_key}'. Code: {target_epsg_code}.")
#         else:
#             fuzzy_matches = get_close_matches(aoi_name_for_match_normalized, 
#                                               list(simplified_epsg_names_for_matching.keys()), 
#                                               n=1, cutoff=matching_cutoff)
            
#             if fuzzy_matches:
#                 simplified_matched_key = fuzzy_matches[0]
#                 best_match_original_full_key = simplified_epsg_names_for_matching[simplified_matched_key]
#                 target_epsg_code = epsg_map_full_names[best_match_original_full_key]
#                 print(f"  Fuzzy matched normalized input AOI '{aoi_name_for_match_normalized}' to simplified EPSG name '{simplified_matched_key}' (Original full: '{best_match_original_full_key}'). Code: {target_epsg_code}.")
#             else:
#                 print(f"  Warning: No close matching EPSG code found for AOI '{aoi_name_from_file}' (normalized: '{aoi_name_for_match_normalized}') in {file_name} with cutoff {matching_cutoff}. Skipping.")
#                 continue 
        
#         try:
#             target_epsg_code = int(target_epsg_code)
#             crs_utm = CRS(f"EPSG:{target_epsg_code}")
#             transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
#         except Exception as e:
#             print(f"  Error: Invalid EPSG code '{target_epsg_code}' for {aoi_name_from_file}. Skipping {file_name}. Error: {e}")
#             continue

#         try:
#             df = pd.read_csv(file_path)

#             # This is the line that was problematic in your error message!
#             # Ensure these are the actual column names from your input CSVs.
#             required_input_cols = [lon_col_in, lat_col_in, ortho_col_in] 
#             if not all(col in df.columns for col in required_input_cols):
#                 print(f"  Error: Missing one or more required input columns ({required_input_cols}) in {file_name}. Skipping.")
#                 print(f"  Available columns in {file_name}: {df.columns.tolist()}") # Add this to debug available columns
#                 continue

#             easting, northing = transformer.transform(df[lon_col_in].values, df[lat_col_in].values)
            
#             df[lon_col_in] = easting
#             df[lat_col_in] = northing

#             print(f"  Coordinates in '{lon_col_in}' and '{lat_col_in}' transformed to UTM (EPSG:{target_epsg_code}).")

#             # --- RENAME COLUMNS ---
#             df = df.rename(columns={lon_col_in: new_easting_col, 
#                                     lat_col_in: new_northing_col, 
#                                     ortho_col_in: new_ortho_col})
#             print(f"  Columns renamed: '{lon_col_in}' to '{new_easting_col}', '{lat_col_in}' to '{new_northing_col}', '{ortho_col_in}' to '{new_ortho_col}'.")

#             # --- REORGANIZE COLUMNS ---
#             fixed_order_cols = [new_easting_col, new_northing_col, new_ortho_col]
#             other_cols = [col for col in df.columns if col not in fixed_order_cols]
#             new_column_order = fixed_order_cols + other_cols
#             df = df[new_column_order]
#             print(f"  Columns reordered with {fixed_order_cols} placed first.")

#             # --- GENERATE NEW FILENAME ---
#             base_name, ext = os.path.splitext(file_name)
#             new_file_name = f"{base_name}{new_file_suffix}{ext}"
#             output_file_path = os.path.join(input_folder, new_file_name)

#             # Save to the new file
#             df.to_csv(output_file_path, index=False)
#             print(f"  Processed and saved to new file: {output_file_path}")

#         except Exception as e:
#             print(f"  Error processing {file_name}: {e}")
#             continue

#     print("\n--- All specified CSV files processed. ---")

# # --- Example Usage ---
# # Define your input folder (which will also be the output folder)
# # and the path to your EPSG codes CSV.
# input_folder = r"B:\Thesis Project\Reference Data\SlideRule_ICESat" # Replace with your actual input folder path
# epsg_csv_path = r"B:\Thesis Project\Reference Data\Refraction Correction\UTM_epsg_codes.csv" # Replace with your actual EPSG codes CSV path

# # Call the function to process your files
# process_and_transform_csvs(
#     input_folder,
#     epsg_csv_path,
#     # You can customize input/output column names and the new file suffix here:
#     # lat_col_in='Original_Latitude', 
#     # lon_col_in='Original_Longitude', 
#     # ortho_col_in='Original_Height',
#     # new_northing_col='UTM_Northing', 
#     # new_easting_col='UTM_Easting', 
#     # new_ortho_col='Adjusted_Height',
#     # new_file_suffix='_UTM' # Example: Anegada_sr_UTM.csv
# )



#################################################################################################################
#################################################################################################################

import os
import pandas as pd
import glob

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
# shared_folder_path = r"B:\Thesis Project\Reference Data\Processed_ICESat" # <--- !!! REPLACE THIS !!!


# # Define the exact column names you want to keep
# columns_to_keep = ["Easting", "Northing", "Geoid_Corrected_Ortho_Height"]


# # --- Run the script ---
# if __name__ == "__main__":
#     if shared_folder_path != r"B:\Thesis Project\Reference Data\Processed_ICESat":
#         print("Please update 'shared_folder_path' variable with your actual path.")
#     else:
#         # Pass the same path for both input and output
#         filter_csv_columns(shared_folder_path, shared_folder_path, columns_to_keep)





#################################################################################################################
#################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def heatscatter(ax, x, y,
                bins, title, cmap,
                xlabel, ylabel, identity_line=False,
                xlim=None, ylim=None, # <<< Assume these will ALWAYS be provided as valid tuples
                **kwargs):
    """
    Create a 2D histogram plot of x and y data.
    Applies provided xlim and ylim directly.

    Parameters
    ----------
    ax : matplotlib axis
    x : numpy ndarray
    y : numpy ndarray
    bins : int or tuple(int,int)
    title : str
    cmap : str
    xlabel: str
    ylabel: str
    identity_line : bool, optional
    xlim : tuple
        Tuple (xmin, xmax) to set the x-axis limits. MUST be provided.
    ylim : tuple
        Tuple (ymin, ymax) to set the y-axis limits. MUST be provided.
    **kwargs:
        Additional keyword arguments passed to matplotlib.hist2d

    Returns
    -------
    hs: tuple or None
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x_finite = x[finite_mask]
    y_finite = y[finite_mask]

    # Set limits FIRST, so even an empty plot has the desired frame
    ax.set_xlim(xlim) # <<< Always applies xlim
    ax.set_ylim(ylim) # <<< Always applies ylim

    if len(x_finite) == 0 or len(y_finite) == 0:
        print("Warning: No finite data points left after filtering NaNs/Infs. Cannot plot hist2d.")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if identity_line:
            # Draw identity line based on the explicitly set limits
            lim_min_plot = max(xlim[0], ylim[0])
            lim_max_plot = min(xlim[1], ylim[1])
            if lim_min_plot < lim_max_plot:
                ax.plot([lim_min_plot, lim_max_plot], [lim_min_plot, lim_max_plot], 'k--', linewidth=1)
            else: # Fallback if limits don't allow for a good diagonal
                ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], 'k--', linewidth=1) # Or use ax.transAxes
        return None

    hs = ax.hist2d(x_finite, y_finite, bins=bins, cmin=1, cmap=cmap, 
                   range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]], # Ensure hist2d also respects limits for binning
                   **kwargs)
    
    if identity_line:
        lim_min_plot = max(xlim[0], ylim[0])
        lim_max_plot = min(xlim[1], ylim[1])
        if lim_min_plot < lim_max_plot:
            ax.plot([lim_min_plot, lim_max_plot], [lim_min_plot, lim_max_plot], 'k--', linewidth=1)
        else:
             ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], 'k--', linewidth=1)


    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # xlim and ylim are already set at the beginning of the function
    return hs

# Define CSV file path and column names
csv_file_path = r"E:\Thesis Stuff\SDB_ExtractedPts\S2A_MSI_2023_03_20_15_41_27_T19TCG_L2W_RGB_m1_SDB_merged_extracted.csv"
x_column_name = "Raster_Value"
y_column_name = "Geoid_Corrected_Ortho_Height"

# Define desired axis limits - these will always be used
manual_xlim = (0, 10)
manual_ylim = (0, 10)

try:
    data_df = pd.read_csv(csv_file_path)
except FileNotFoundError: # Gracefully handle file not found
    print(f"Error: CSV file not found at {csv_file_path}")
    exit()
except Exception as e: # Catch other potential errors during CSV read
    print(f"Error reading CSV file: {e}")
    exit()

if not (x_column_name in data_df.columns and y_column_name in data_df.columns):
    print(f"Error: One or both specified columns ('{x_column_name}', '{y_column_name}') not found in the CSV.")
    print(f"Available columns: {data_df.columns.tolist()}")
    exit()

x_data_from_csv = pd.to_numeric(data_df[x_column_name], errors='coerce').values
y_data_from_csv = pd.to_numeric(data_df[y_column_name], errors='coerce').values


# Create the Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Call your heatscatter function, passing the limits
hist_output = heatscatter(ax, x_data_from_csv, y_data_from_csv,
                          bins=100,
                          title="Density Heatmap",
                          cmap='viridis',
                          xlabel=f"{x_column_name} Values",
                          ylabel=f"{y_column_name} Values",
                          identity_line=True,
                          xlim=manual_xlim, # Pass manual_xlim
                          ylim=manual_ylim  # Pass manual_ylim
                          )

if hist_output is not None:
    if isinstance(hist_output, tuple) and len(hist_output) == 4:
        plt.colorbar(hist_output[3], ax=ax, label='Counts per Bin')
    else:
        try:
            plt.colorbar(hist_output, ax=ax, label='Counts per Bin')
        except TypeError:
            print("Note: Could not automatically create colorbar from hist_output.")

plt.tight_layout()
plt.show()





























