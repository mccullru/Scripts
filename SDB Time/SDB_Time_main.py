# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:58:13 2025

@author: mccullru
"""

import sys
import SDB_Time


# =============================================================================
# --- A Logger Class to Redirect 'print' output ---
# =============================================================================
class Logger:
    """
    A class to redirect stdout to both the console and a file.
    All 'print' statements will be captured.
    """
    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        self.logfile = open(log_filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # This flush method is needed for compatibility.
        self.terminal.flush()
        self.logfile.flush()
    
    def close(self):
        self.logfile.close()


# =============================================================================
# --- MASTER RASTER PIPELINE CONFIGURATION ---
# =============================================================================

# !!! CHANGE THESE PATHS!!! 
raster_config = {
    # --- Input Files/Folders ---
    
    # ACOLITE Corrected image bands (R, G, and B)
    'raw_bands_folder': r"E:\Thesis Stuff\AcoliteWithPython\Corrected_Imagery\All_SuperDove\SD_Skutvik_output", 
    
    # Calibration data in points format csv. Coords should be UTM,
    # !!! columns should be Easting, Northing, and Depth !!!
    'cal_points_csv': r"B:\Thesis Project\Reference Data\Processed_ICESat\Skutvik_cal.csv",     
                                                                                                
    # Validation data in points format
    'val_points_csv': r"B:\Thesis Project\Reference Data\Processed_ICESat\Skutvik_acc.csv",     


# !!! CHANGE FOLDER PATHS !!!     
# --- Define each output folder individually ---
    'output_folders': {
        'rgb_composites': r"E:\Thesis Stuff\RGBCompositOutput",
        'masked_odw':     r"E:\Thesis Stuff\RGBCompositOutput", 
        'psdb_rasters':   r"E:\Thesis Stuff\pSDB",
        'cal_extract':    r"E:\Thesis Stuff\pSDB_ExtractedPts",
        'regr_results':   r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results",
        'sdb_final':      r"E:\Thesis Stuff\SDB", 
        'val_extract':    r"E:\Thesis Stuff\SDB_ExtractedPts",      # Only needed if there is validation data for testing
        'val_analysis':   r"E:\Thesis Stuff\SDB_ExtractedPts_maxR2_results"     # Only needed if there is validation data for testing
    },


# !!! CHANGE FOLDER PATH !!!
    #--- Log file path ---
    # Contains everything printed to the console
    'log_file': r"E:\Thesis Stuff\SDB_time_log.txt",


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
    
    # --- Redirect all 'print' output to the Logger ---
    original_stdout = sys.stdout 
    logger = Logger(raster_config['log_file'])
    sys.stdout = logger 

    try:
        # All print statements from here on will be logged,
        print(">>> Test successful: The main script is running! <<<")
        print("--- Initializing Raster Processing Workflow ---")
        
        # This function's internal 'print' statements will now be captured
        SDB_Time.run_full_pipeline(config=raster_config)
        
        print("\n--- Workflow Complete ---")

    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        # Also print the full traceback to the log
        import traceback
        traceback.print_exc(file=sys.stdout)
    finally:
        # --- Restore original stdout and close the log file ---
        sys.stdout = original_stdout
        logger.close()
        print("\nLogging complete. Standard output has been restored.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    