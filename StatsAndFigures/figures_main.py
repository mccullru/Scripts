# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:11:07 2025

@author: mccullru
"""

import os
import sys

import figures_kde
import figures_heatmap
import figures_histogram
import figures_avg_depth

# =============================================================================
# --- MASTER CONFIGURATION ---
# =============================================================================

# --- 1. Choose Which Plots to Generate ---
RUN_KDE_PLOT = True
RUN_HEATMAP_PLOT = True
RUN_INDIVIDUAL_HISTOGRAM = True
RUN_DEPTH_RANGE_BAR_PLOT = True

# --- 2. Common Inputs & Metadata ---
# Folder for the primary data CSVs (used by all scripts)
input_csv_folder_path = r"E:\Thesis Stuff\SDB_ExtractedPts"

# Folder for the statistics CSVs (used by KDE plot for filtering)
stats_csv_folder_path = r"E:\Thesis Stuff\SDB_ExtractedPts_maxR2_results"

exclusion_list_path = r"B:\Thesis Project\SDB_Time\Excluded_datasets\Excluded_Datasets.csv"

# General Info
AOI = 'SouthPort'
Sensor = 'S2'

# --- 3. Common Outputs ---
# Master folder for all generated figures
output_parent_folder = r"E:\Thesis Stuff\Figures"


# --- 4. Script-Specific Settings ---

# -- KDE Plot Settings --
kde_config = {
    'sdb_types_to_process': ['SDB', 'SDB_red', 'SDB_green', 'SDB_merged'],
    'apply_depth_filter': True,
    'r2_threshold': 0.7,
    'plot_xlim': (-10, 10),
    'plot_ylim': (0, 2000),
    'error_filter_bounds': (-10, 10),
    'save_plots': True,
    'show_plots': True
}

# -- Heatmap Plot Settings --
heatmap_config = {
    'manual_xlim': (0, 10),
    'manual_ylim': (0, 10),
    'r2_threshold': 0.7,
    'save_plots': True,
    'show_plots': True
}

# -- Individual Histogram Settings --
histogram_config = {
    'fixed_bin_width': 0.1,
    'error_filter_bounds': (-10, 10),
    'hist_xlim': (-10, 10),
    'hist_ylim': (0, 1500),
    'apply_depth_filter': True,
    'r2_threshold': 0.7,
    'save_plots': True,
    'show_plots': True
}

# -- Depth Range Bar Plot Settings --
depth_range_bar_config = {
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
    # Define the log file path
    log_filename = "figure_generation_log.txt"
    log_filepath = os.path.join(output_parent_folder, log_filename)
    os.makedirs(output_parent_folder, exist_ok=True) # Ensure output folder exists for log

    # Store original stdout
    original_stdout = sys.stdout

    # Create an instance of DualOutput
    dual_output = DualOutput(log_filepath, original_stdout)

    try:
        # Redirect stdout to the custom DualOutput object
        sys.stdout = dual_output

        print("--- Starting Plot Generation ---")

        if RUN_KDE_PLOT:
            print("\n>>> Running Combined KDE Plot Generation...")
            output_kde_path = os.path.join(output_parent_folder, "KDE_Plots")
            output_sumstats_folder = os.path.join(output_parent_folder, "Summary_Stats")

            figures_kde.generate_kde_plots(
                input_folder=input_csv_folder_path,
                stats_folder=stats_csv_folder_path,
                output_plot_folder=output_kde_path,
                output_csv_folder=output_sumstats_folder,
                aoi=AOI,
                sensor=Sensor,
                config=kde_config,
                exclusion_path=exclusion_list_path
            )
            print(">>> KDE Plot Generation Complete.")

        if RUN_HEATMAP_PLOT:
            print("\n>>> Running Heatmap Generation...")
            output_heatmap_path = os.path.join(output_parent_folder, "Heatmap_Plots")

            figures_heatmap.generate_heatmaps(
                input_folder=input_csv_folder_path,
                output_folder=output_heatmap_path,
                stats_folder=stats_csv_folder_path,
                aoi=AOI,
                sensor=Sensor,
                config=heatmap_config,
                exclusion_path=exclusion_list_path
            )
            print(">>> Heatmap Generation Complete.")

        if RUN_INDIVIDUAL_HISTOGRAM:
            print("\n>>> Running Individual Histogram Generation...")
            output_hist_path = os.path.join(output_parent_folder, "Individual_Histograms")
            figures_histogram.generate_histograms(
                input_folder=input_csv_folder_path,
                stats_folder=stats_csv_folder_path,
                output_folder=output_hist_path,
                aoi=AOI,
                sensor=Sensor,
                config=histogram_config,
                exclusion_path=exclusion_list_path
            )
            print(">>> Individual Histogram Generation Complete.")

        if RUN_DEPTH_RANGE_BAR_PLOT:
            print("\n>>> Running Depth Range Bar Chart Generation...")
            output_bar_chart_path = os.path.join(output_parent_folder, "Average_Depth_Ranges")
            figures_avg_depth.generate_depth_range_bar_chart(
                input_folder=input_csv_folder_path,
                stats_folder=stats_csv_folder_path,
                output_folder=output_bar_chart_path,
                aoi=AOI,
                sensor=Sensor,
                config=depth_range_bar_config,
                exclusion_path=exclusion_list_path
            )
            print(">>> Depth Range Bar Chart Generation Complete.")

        print("\n--- All selected tasks are finished. ---")

    finally:
        # Close the log file stream within the DualOutput object
        dual_output.log.close()
        
        # Restore original stdout
        sys.stdout = original_stdout
        print(f"\n--- Log saved to: {log_filepath} ---")


    
    
    
    
    