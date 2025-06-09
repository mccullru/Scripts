# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:11:07 2025

@author: mccullru
"""

import os
import plot_kde
import plot_heatmap
import plot_histogram

# =============================================================================
# --- MASTER CONFIGURATION ---
# =============================================================================

# --- 1. Choose Which Plots to Generate ---
RUN_KDE_PLOT = True
RUN_HEATMAP_PLOT = True
RUN_INDIVIDUAL_HISTOGRAM = True

# --- 2. Common Inputs & Metadata ---
# Folder for the primary data CSVs (used by all scripts)
input_csv_folder_path = r"E:\Thesis Stuff\SDB_ExtractedPts"
# Folder for the statistics CSVs (used by KDE plot for filtering)
stats_csv_folder_path = r"E:\Thesis Stuff\SDB_ExtractedPts_maxR2_results"

# General Info
AOI = 'Anegada'
Sensor = 'SD'

# --- 3. Common Outputs ---
# Master folder for all generated figures
output_parent_folder = r"B:\Thesis Project\StatsAndFigures\Combined_Output"


# --- 4. Script-Specific Settings ---

# -- KDE Plot Settings --
kde_config = {
    'sdb_types_to_process': ['SDB', 'SDB_red', 'SDB_green', 'SDB_merged'],
    'apply_depth_filter': True,
    'r2_threshold': 0.7,
    'plot_xlim': (-10, 10),
    'plot_ylim': (0, 700),
    'error_filter_bounds': (-10, 10),
    'save_plots': False,
    'show_plots': True
}

# -- Heatmap Plot Settings --
heatmap_config = {
    'manual_xlim': (0, 10),
    'manual_ylim': (0, 10),
    'save_plots': False,
    'show_plots': True
}

# -- Individual Histogram Settings --
histogram_config = {
    'fixed_bin_width': 0.1,
    'error_filter_bounds': (-10, 10),
    'hist_xlim': (-10, 10),
    'hist_ylim': (0, 700),
    'save_plots': False,
    'show_plots': True
}


# =============================================================================
# --- EXECUTION LOGIC ---
# =============================================================================
if __name__ == "__main__":
    print("--- Starting Plot Generation ---")

    if RUN_KDE_PLOT:
        print("\n>>> Running Combined KDE Plot Generation...")
        output_kde_path = os.path.join(output_parent_folder, "KDE_Plots")
        plot_kde.generate_kde_plots(
            input_folder=input_csv_folder_path,
            stats_folder=stats_csv_folder_path,
            output_folder=output_kde_path,
            aoi=AOI,
            sensor=Sensor,
            config=kde_config
        )
        print(">>> KDE Plot Generation Complete.")

    if RUN_HEATMAP_PLOT:
        print("\n>>> Running Heatmap Generation...")
        output_heatmap_path = os.path.join(output_parent_folder, "Heatmap_Plots")
        plot_heatmap.generate_heatmaps(
            input_folder=input_csv_folder_path,
            output_folder=output_heatmap_path,
            aoi=AOI,
            sensor=Sensor,
            config=heatmap_config
        )
        print(">>> Heatmap Generation Complete.")

    if RUN_INDIVIDUAL_HISTOGRAM:
        print("\n>>> Running Individual Histogram Generation...")
        output_hist_path = os.path.join(output_parent_folder, "Individual_Histograms")
        plot_histogram.generate_histograms(
            input_folder=input_csv_folder_path,
            output_folder=output_hist_path,
            aoi=AOI,
            sensor=Sensor,
            config=histogram_config
        )
        print(">>> Individual Histogram Generation Complete.")

    print("\n--- All selected tasks are finished. ---")