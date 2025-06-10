# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:58:13 2025

@author: mccullru
"""


import os
import SDB_Time

# =============================================================================
# --- MASTER RASTER PIPELINE CONFIGURATION ---
# =============================================================================
raster_config = {
    # --- Input Files/Folders ---
    'raw_bands_folder': r"E:\Thesis Stuff\AcoliteWithPython\Corrected_Imagery\All_SuperDove\SD_Anegada_output",
    'cal_points_csv': r"B:\Thesis Project\Reference Data\Processed_ICESat\Anegada_cal.csv",
    'val_points_csv': r"B:\Thesis Project\Reference Data\Processed_ICESat\Anegada_acc.csv",

    # --- Define each output folder individually ---
    'output_folders': {
        'rgb_composites': r"E:\Thesis Stuff\RGBCompositOutput",
        'masked_odw':     r"E:\Thesis Stuff\RGBCompositOutput", 
        'psdb_rasters':   r"E:\Thesis Stuff\pSDB",
        'cal_extract':    r"E:\Thesis Stuff\pSDB_ExtractedPts",
        'regr_results':   r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results",
        'sdb_final':      r"E:\Thesis Stuff\SDB", 
        'val_extract':    r"E:\Thesis Stuff\SDB_ExtractedPts",
        'val_analysis':   r"E:\Thesis Stuff\SDB_ExtractedPts_maxR2_results" 
    },

    # --- Processing Parameters ---
    'target_wavelengths': {'red': 666, 'green': 560, 'blue': 492},
    'wavelength_tolerance': 10,
    'odw_threshold': 0.003,
    'delete_intermediate_files': False,
    'r2_threshold': 0.7,
    'merge_lower_limit': 2.0,
    'merge_upper_limit': 3.5,
    'nodata_value': -9999,
}


# =============================================================================
# --- EXECUTION LOGIC  ---
# =============================================================================
if __name__ == "__main__":
    
    print(">>> Test successful: The main script is running! <<<")
    print("--- Initializing Raster Processing Workflow ---")
    SDB_Time.run_full_pipeline(config=raster_config)
    print("\n--- Workflow Complete ---")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    