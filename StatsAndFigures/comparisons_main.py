# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 12:28:23 2025

@author: mccullru
"""

import os
import sys

import comparisons_kde 
import comparisons_avg_depth


# =============================================================================
# --- MASTER CONFIGURATION FOR AVERAGED KDE PLOT ---
# =============================================================================

# --- 1. Choose Which Plots to Generate ---
# This script is designed to ONLY run the averaged KDE plot
RUN_AVERAGED_KDE_PLOTS = True
RUN_MAX_DEPTH_BAR_COMPARISON = True


# --- 2. Input & Metadata for Averaged KDE Plot ---

# Inputs for the KDE plots
SENTINEL2_INPUT_FOLDER = r"B:\Thesis Project\SDB_Time\Results_main\Sentinel2\Marathon\SDB_ExtractedPts"
SENTINEL2_STATS_FOLDER = r"B:\Thesis Project\SDB_Time\Results_main\Sentinel2\Marathon\SDB_ExtractedPts_maxR2_results"

SUPERDOVE_INPUT_FOLDER = r"B:\Thesis Project\SDB_Time\Results_main\SuperDove\Marathon\SDB_ExtractedPts"
SUPERDOVE_STATS_FOLDER = r"B:\Thesis Project\SDB_Time\Results_main\SuperDove\Marathon\SDB_ExtractedPts_maxR2_results"



# General Info for the plot title and output filename
AOI = 'Marathon'
COMBINED_SENSORS_NAME = "S-2 & SD" 

# Path to the dataset exclusion list 
exclusion_list_path = r"B:\Thesis Project\SDB_Time\Excluded_datasets\Excluded_Datasets.csv"

# List of SDB types to compare (each will get its own chart)
SDB_TYPES_TO_COMPARE = ['SDB_red', 'SDB_green', 'SDB_merged'] 


# --- 3. Output Settings ---
# Master folder for all generated figures from this script
output_parent_folder = r"B:\Thesis Project\SDB_Time\Comparisons\Marathon"


# --- 4. Script-Specific Settings for figures_averaged_kde.py ---
averaged_kde_config = {
    'apply_depth_filter': True,
    'r2_threshold': 0.7,
    'plot_xlim': (-10, 10),
    'plot_ylim': (0, 50), 
    'error_filter_bounds': (-10, 10),
    'save_plots': True,
    'show_plots': True,
    'kde_bw_method': 0.15 
}

# -- Max Depth Bar Comparison Plot Settings --
max_depth_bar_config = {
    'r2_threshold': 0.7,
    'stats_indicator_col': 'Indicator',
    'stats_r2_col': 'R2 Value',
    'save_plots': True,
    'show_plots': True,
}

# =============================================================================
# --- EXECUTION LOGIC ---
# =============================================================================

# Define a custom class to write output to both console and file
class DualOutput:
    def __init__(self, filename, original_stdout):
        self.terminal = original_stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    
    log_filename = "averaged_kde_log.txt"
    log_filepath = os.path.join(output_parent_folder, log_filename)
    
    # Ensure output folder exists for plots and the log
    os.makedirs(output_parent_folder, exist_ok=True)

    # Store original stdout
    original_stdout = sys.stdout

    # Create an instance of DualOutput
    dual_output = DualOutput(log_filepath, original_stdout)

    try:
        # Redirect stdout to the custom DualOutput object
        sys.stdout = dual_output

        print("--- Starting Averaged KDE Plot Generation Run ---")

        if RUN_AVERAGED_KDE_PLOTS: 
            print("\n>>> Running Averaged KDE Plots for S2 vs SuperDove (per SDB type)...")
            
            # The output folder for the averaged KDE plot will be output_parent_folder itself for this script
            output_avg_kde_path = output_parent_folder
            

            for sdb_type in SDB_TYPES_TO_COMPARE: 
                comparisons_kde.generate_averaged_kde_plot(
                    sentinel2_input_folder=SENTINEL2_INPUT_FOLDER,
                    sentinel2_stats_folder=SENTINEL2_STATS_FOLDER,
                    superdove_input_folder=SUPERDOVE_INPUT_FOLDER,
                    superdove_stats_folder=SUPERDOVE_STATS_FOLDER,
                    output_folder=output_avg_kde_path,
                    aoi=AOI,
                    sensor_combined_name=COMBINED_SENSORS_NAME,
                    sdb_type_to_compare=sdb_type, 
                    config=averaged_kde_config,
                    exclusion_path=exclusion_list_path
                )
            print(">>> All Averaged KDE Plots Generation Complete.")


        if RUN_MAX_DEPTH_BAR_COMPARISON:
            print("\n>>> Running Max Depth Bar Chart Comparison (S2 vs SuperDove)...")
            output_max_depth_bar_path = output_parent_folder
            os.makedirs(output_max_depth_bar_path, exist_ok=True)

            comparisons_avg_depth.generate_comparison_max_depth_bar_chart(
                sentinel2_input_folder=SENTINEL2_INPUT_FOLDER, 
                sentinel2_stats_folder=SENTINEL2_STATS_FOLDER, 
                superdove_input_folder=SUPERDOVE_INPUT_FOLDER, 
                superdove_stats_folder=SUPERDOVE_STATS_FOLDER, 
                output_folder=output_max_depth_bar_path,
                aoi=AOI,
                sensor_combined_name=COMBINED_SENSORS_NAME,
                config=max_depth_bar_config,
                exclusion_path=exclusion_list_path
            )
            print(">>> Max Depth Bar Chart Comparison Complete.")

        print("\n--- All selected tasks are finished. ---")

    finally:
        # Close the log file stream within the DualOutput object
        dual_output.log.close()
        # Restore original stdout
        sys.stdout = original_stdout
        print(f"\n--- Log saved to: {log_filepath} ---")

        
        
        
        