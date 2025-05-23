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
# from pathlib import Path # Not strictly needed if using os.path consistently

# --- Helper function to extract Rrs wavelength ---
def extract_rrs_number(file_name):
    """Extract the number after 'Rrs_' or 'Rrs' in the file name."""
    match = re.search(r'Rrs_(\d+)', file_name)
    if match:
        return int(match.group(1))
    match_alt = re.search(r'Rrs(\d+)', file_name)
    if match_alt:
        return int(match_alt.group(1))
    return None

def is_close_to(value, target, tolerance):
    """Check if a value is close to a target within a tolerance."""
    if value is None:
        return False
    return abs(value - target) <= tolerance

# --- MODIFICATION: More robust Scene ID extraction ---
def extract_core_scene_id(filename):
    """
    Extracts the core scene identifier from various filename patterns.
    Target ID Example: S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W
    """
    # Try pattern for typical Sentinel-2 L2A product names (often used as a base by ACOLITE)
    # This captures the part before typical ACOLITE suffixes like _L2W_Rrs_XXX or just _L2W.tif
    # S2B_MSI_YYYYMMDDTHHMMSS_NXXXX_RXXX_TXXXXX_YYYYMMDDTHHMMSS (standard Sentinel naming)
    # Or potentially S2B_MSI_YYYY_MM_DD_HH_MM_SS_TILEID_L2W (ACOLITE might keep this part)

    # Common Sentinel-2 base product name structure (up to processing baseline and tile ID)
    # e.g. S2B_MSI_20230128T070609_N0509_R110_T39PYP_20230128T085300
    # Acolite filenames might be like: S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W_Rrs_XXX.tif
    # Your SDB files are like: S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W__RGB_ODWmasked_SDBgreen.tif

    # Let's try to capture the core part: S2B_MSI_YYYY_MM_DD_HH_MM_SS_TILEID_L2W
    # The L2W seems to be the common end part before ACOLITE/your suffixes.
    # Your SDB identifier also has an extra underscore after L2W.

    # Regex to capture the part like 'S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W'
    # This pattern looks for the S2 prefix, MSI, date, time, TileID, and ends with L2W
    core_id_match = re.match(r'(S2[AB]_MSI_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_T\d{2}[A-Z]{3}_L2W)', filename)
    if core_id_match:
        return core_id_match.group(1)

    # Fallback if the above specific pattern isn't perfect for all cases
    # Try to strip known suffixes more generally.
    # This was your red-edge extracted ID: 'S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W'
    # This was your SDB extracted ID:    'S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W_' (with extra _)

    # Let's try to match up to L2W and then clean up potential trailing underscores from SDB files
    base_name_match = re.match(r'(S2[AB]_MSI_.*?_L2W)', filename) # Non-greedy match until L2W
    if base_name_match:
        potential_id = base_name_match.group(1)
        # print(f"DEBUG extract_core_scene_id - Matched up to L2W: {potential_id} from {filename}")
        return potential_id # This should give 'S2B_MSI_2023_01_28_07_06_09_T39PYP_L2W' for both

    # If the above fails, revert to a more general stripping (less reliable)
    print(f"DEBUG extract_core_scene_id - Using fallback for: {filename}")
    base = os.path.splitext(filename)[0]
    # List of suffixes to remove, order might matter if some are subsets of others
    suffixes_to_remove = [
        "_RGB_ODWmasked_SDBgreen", "_RGB_ODWmasked_SDBred", "_RGB_ODWmasked_SDB_merged",
        "_SDBgreen", "_SDBred", "_SDB_merged",
        "_ODWmaskedRE", "_ODWmasked", "_RGB",
        "_Rrs_704", "_Rrs_666", "_Rrs_560", "_Rrs_492", # Add other Rrs bands if needed
        "_Rrs704", "_Rrs666", "_Rrs560", "_Rrs492"
    ]
    for suffix in suffixes_to_remove:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break # Remove one suffix and assume that's enough for this fallback
    return base.rstrip('_') # Remove any trailing underscores just in case
# --- END MODIFICATION ---


# --- Main Function (mask_sdb_with_red_edge) ---
# ... (The rest of your mask_sdb_with_red_edge function remains the same,
#      it will now use the refined extract_core_scene_id) ...
# For brevity, I'll just show where it's called:

def mask_sdb_with_red_edge(input_sdb_folder, input_multi_band_folder, output_folder_sdb):
    print(f"Input SDB Folder: {input_sdb_folder}")
    print(f"Input Multi-Band (Red-Edge source) Folder: {input_multi_band_folder}")
    print(f"Output SDB (Masked) Folder: {output_folder_sdb}")

    if not os.path.exists(output_folder_sdb):
        os.makedirs(output_folder_sdb)
        print(f"Created output folder: {output_folder_sdb}")

    sdb_files_paths = glob.glob(os.path.join(input_sdb_folder, "*.tif"))

    red_edge_candidates = {}
    all_band_files = glob.glob(os.path.join(input_multi_band_folder, "*.tif"))

    print(f"DEBUG: Scanning Red-Edge folder: {input_multi_band_folder}")
    for band_file_path in all_band_files:
        band_filename = os.path.basename(band_file_path)
        rrs_wavelength = extract_rrs_number(band_filename)
        if is_close_to(rrs_wavelength, 704, 5):
            scene_id = extract_core_scene_id(band_filename) # USE REFINED FUNCTION
            if scene_id:
                red_edge_candidates[scene_id] = band_file_path
                print(f"  DEBUG: Found potential Red-Edge: ID='{scene_id}', File='{band_filename}'")
            else:
                print(f"  DEBUG: Could not extract scene_id from Red-Edge candidate: {band_filename}")

    if not sdb_files_paths: # ... (rest of your checks and main loop) ...
        print("No SDB TIFF files found in the input SDB folder.")
        return
    if not red_edge_candidates:
        print("No potential Red-Edge (Rrs~704nm) TIFF files identified and mapped in the multi-band folder.")
        return

    print(f"Found {len(sdb_files_paths)} SDB files.")
    print(f"Identified {len(red_edge_candidates)} unique Rrs~704nm Red-Edge files by Scene ID.")

    for sdb_file_path in sdb_files_paths:
        sdb_filename = os.path.basename(sdb_file_path)
        sdb_identifier = extract_core_scene_id(sdb_filename) # USE REFINED FUNCTION

        print(f"\n--- Processing SDB file: {sdb_filename} (Attempting to match ID: '{sdb_identifier}') ---")
        corresponding_red_edge_file = red_edge_candidates.get(sdb_identifier)

        if not corresponding_red_edge_file:
            print(f"  Corresponding Red-Edge (Rrs~704nm) file not found for SDB identifier: '{sdb_identifier}'. Skipping this SDB file.")
            continue
        # ... (rest of the processing logic from your previous script) ...
        print(f"  Using Red-Edge file: {os.path.basename(corresponding_red_edge_file)}")
        try:
            with rasterio.open(sdb_file_path) as sdb_src:
                sdb_array = sdb_src.read(1).astype(rasterio.float32)
                sdb_profile = sdb_src.profile
                sdb_nodata = sdb_src.nodata
                if sdb_nodata is not None:
                    sdb_array[sdb_array == sdb_nodata] = np.nan
            with rasterio.open(corresponding_red_edge_file) as re_src:
                red_edge_array = re_src.read(1).astype(rasterio.float32)
                re_nodata = re_src.nodata
                if re_nodata is not None:
                    red_edge_array[red_edge_array == re_nodata] = np.nan
            if sdb_array.shape != red_edge_array.shape:
                print(f"  ERROR: SDB and Red-Edge rasters for {sdb_identifier} do not have the same dimensions. Skipping.")
                continue
            sdb_for_log = sdb_array.copy()
            sdb_for_log[sdb_for_log <= 0] = np.nan
            red_edge_for_log = red_edge_array.copy()
            red_edge_for_log[red_edge_for_log <= 0] = np.nan
            with np.errstate(divide='ignore', invalid='ignore'):
                log_sdb_odw_threshold = -0.63 * np.log(red_edge_for_log) -2.56
                ln_sdb_array = np.log(sdb_for_log)
            odw_condition = ln_sdb_array > log_sdb_odw_threshold
            num_odw_pixels = np.nansum(odw_condition.astype(int))
            print(f"  Identified {num_odw_pixels} ODW pixels to mask based on Red-Edge criteria.")
            sdb_masked_array = sdb_array.copy()
            sdb_masked_array[odw_condition] = np.nan
            output_profile = sdb_profile.copy()
            output_profile.update(dtype=rasterio.float32, nodata=np.nan, count=1)
            sdb_filename_stem = os.path.splitext(sdb_filename)[0]
            output_filename = f"{sdb_filename_stem}_m2.tif"
            output_file_path = os.path.join(output_folder_sdb, output_filename)
            with rasterio.open(output_file_path, 'w', **output_profile) as dst:
                dst.write(sdb_masked_array, 1)
            print(f"  Saved masked SDB to: {output_file_path}")
        except Exception as e:
            print(f"  Error processing file pair for {sdb_filename}: {e}")
            import traceback
            traceback.print_exc()

# --- Configuration ---
input_red_edge_folder = r"E:\Thesis Stuff\AcoliteWithPython\Corrected_Imagery\odf_test_output\S2"
input_SDB_folder = r"E:\Thesis Stuff\SDB"
output_masked_SDB_folder = input_SDB_folder

# --- Run the process ---
mask_sdb_with_red_edge(input_SDB_folder, input_red_edge_folder, output_masked_SDB_folder)

print("\n--- Script finished ---")

