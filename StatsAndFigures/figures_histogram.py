# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:11:06 2025

@author: mccullru
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

from difflib import get_close_matches 


""" Individual Error Histograms """

# Final, complete generate_histograms function

def generate_histograms(input_folder, 
                        output_folder, 
                        stats_folder, 
                        aoi, 
                        sensor, 
                        config, 
                        exclusion_path):
    """
    Creates individual error histograms for each input CSV file, but ONLY for
    files that meet the R^2 threshold from a corresponding statistics file.
    Also applies a depth filter based on the stats file.
    (Complete version with R^2 and depth filtering logic)
    """
    # --- Configuration ---
    input_csv_folder_path = input_folder
    output_plot_folder_path = output_folder
    stats_csv_folder_path = stats_folder
    AOI = aoi
    Sensor = sensor
    
    # Unpack config
    fixed_bin_width = config.get('fixed_bin_width', 0.1)
    error_filter_min, error_filter_max = config.get('error_filter_bounds', (-10, 10))
    hist_xlim = config.get('hist_xlim', (-10, 10))
    hist_ylim = config.get('hist_ylim', (0, 700))
    save_plots = config.get('save_plots', True)
    show_plots = config.get('show_plots', True)
    apply_depth_filter = config.get('apply_depth_filter', True)
    r2_threshold_for_selection = config.get('r2_threshold', 0.7)
    
    # Define column names
    ref_col = "Geoid_Corrected_Ortho_Height"
    SDB_col = "Raster_Value"
    error_col_name = "Error"
    
    # Column names expected in the stats CSVs
    stats_indicator_col = "Indicator"
    stats_r2_col = "R2 Value"
    stats_max_depth_col = "Max Depth Range"
    stats_min_depth_col = "Min Depth Range"
    
    # --- Create Output Folder ---
    if not os.path.exists(output_plot_folder_path):
        os.makedirs(output_plot_folder_path)
        print(f"Created output folder: {output_plot_folder_path}")
    
    
    # --- LOAD THE EXCLUSION LIST FROM CSV ---
    excluded_files = set()
    if os.path.exists(exclusion_path):
        try:
            df_exclude = pd.read_csv(exclusion_path)
            if 'exclusion_list' in df_exclude.columns:
                excluded_files = set(df_exclude['exclusion_list'].dropna().str.lower().tolist())
                print(f"Loaded {len(excluded_files)} files to exclude from '{os.path.basename(exclusion_path)}'.")
            else:
                print(f"Warning: Column 'exclusion_list' not found in {exclusion_path}. No files will be excluded.")
        except Exception as e:
            print(f"Warning: Could not read exclusion CSV file '{exclusion_path}'. Error: {e}")
    else:
        print(f"Warning: No exclusion list found at '{exclusion_path}'.")

    # --- Prepare for Depth Filtering ---
    stats_csv_filenames = []
    if apply_depth_filter and os.path.isdir(stats_csv_folder_path):
        stats_csv_filenames = [f for f in os.listdir(stats_csv_folder_path) if f.lower().endswith('.csv')]
        if not stats_csv_filenames:
            print("Warning: No stats CSVs found. Depth filtering will be disabled.")
            apply_depth_filter = False
    elif apply_depth_filter:
        print("Warning: Stats folder not found. Depth filtering will be disabled.")
        apply_depth_filter = False

    # --- Find and Process Data CSV Files ---
    csv_files = glob.glob(os.path.join(input_csv_folder_path, "*.csv"))
    
    # APPLY THE EXCLUSION LIST FILTER ---
    original_count = len(csv_files)
    csv_files = [
        fpath for fpath in csv_files
        if os.path.splitext(os.path.basename(fpath))[0].lower() not in excluded_files
    ]
    files_removed = original_count - len(csv_files)
    if files_removed > 0:
        print(f"Filtered out {files_removed} files based on the exclusion list.")
   
    
    if not csv_files:
        print(f"No data CSV files found in {input_csv_folder_path}"); return
    
    print(f"Found {len(csv_files)} data CSVs to process.")
    
    for csv_file_path in csv_files:
        print(f"\n--- Processing data file: {os.path.basename(csv_file_path)} ---")
        base_filename = os.path.basename(csv_file_path)
        filename_no_ext = os.path.splitext(base_filename)[0]
        fig = None
        
        selected_min_depth, selected_max_depth = None, None

        # --- R^2 and Depth Filter Logic ---
        if apply_depth_filter:
            expected_stats_csv_name = f"{filename_no_ext}_LR_Stats_iterations.csv"
            matched_stats_csv = next((f for f in stats_csv_filenames if f.lower() == expected_stats_csv_name.lower()), None)

            if not matched_stats_csv:
                print(f"  Warning: No matching stats file found for {base_filename}. Plot will be generated without depth filtering.")
            else:
                try:
                    stats_df = pd.read_csv(os.path.join(stats_csv_folder_path, matched_stats_csv))
                    required_cols = [stats_indicator_col, stats_r2_col, stats_max_depth_col, stats_min_depth_col]
                    if not all(col in stats_df.columns for col in required_cols):
                        print(f"  Warning: Stats CSV {matched_stats_csv} is missing required columns. Skipping depth selection.")
                    else:
                        stats_df = stats_df.dropna(subset=required_cols).copy()
                        row_to_use = None

                        # Check Indicator 2 first
                        rows_ind2 = stats_df[stats_df[stats_indicator_col] == 2]
                        if not rows_ind2.empty:
                            r2_check = round(rows_ind2.iloc[0][stats_r2_col], 2)
                            if r2_check >= r2_threshold_for_selection:
                                row_to_use = rows_ind2.iloc[0]
                        
                        # If Indicator 2 failed, fallback to Indicator 1
                        if row_to_use is None:
                            rows_ind1 = stats_df[stats_df[stats_indicator_col] == 1]
                            if not rows_ind1.empty:
                                r2_check = round(rows_ind1.iloc[0][stats_r2_col], 2)
                                if r2_check >= r2_threshold_for_selection:
                                    row_to_use = rows_ind1.iloc[0]

                        if row_to_use is not None:
                            selected_min_depth = row_to_use[stats_min_depth_col]
                            selected_max_depth = row_to_use[stats_max_depth_col]
                        else:
                            print(f"  R2 criteria not met for {base_filename} (Threshold: {r2_threshold_for_selection}). Skipping histogram generation.")
                            continue
                except Exception as e:
                    print(f"  Error processing stats file {matched_stats_csv}: {e}")

        # --- Data Processing and Plotting ---
        try:
            df = pd.read_csv(csv_file_path)
            if ref_col not in df.columns or SDB_col not in df.columns:
                print(f"Warning: Required columns not found in {base_filename}. Skipping.")
                continue
            
            df[ref_col] = pd.to_numeric(df[ref_col], errors='coerce')
            df[SDB_col] = pd.to_numeric(df[SDB_col], errors='coerce')
            df.dropna(subset=[ref_col, SDB_col], inplace=True)
            if df.empty:
                print(f"Warning: No valid numeric data in {base_filename}. Skipping."); continue

            if selected_min_depth is not None and selected_max_depth is not None:
                original_len = len(df)
                df = df[(df[ref_col] >= selected_min_depth) & (df[ref_col] <= selected_max_depth)]
                print(f"  Applied depth filter ({selected_min_depth:.2f}m - {selected_max_depth:.2f}m). Points: {original_len} -> {len(df)}.")
            
            if df.empty:
                print("  No data points remain after depth filtering. Skipping."); continue
            
            # Error = Measured - Reference
            df[error_col_name] = df[SDB_col] - df[ref_col]
            error_data_for_hist = df[error_col_name].dropna()
            error_data_for_hist = error_data_for_hist[(error_data_for_hist > error_filter_min) & (error_data_for_hist < error_filter_max)]
            
            if error_data_for_hist.empty:
                print("Warning: No error data to plot after final filtering. Skipping."); continue

            stats = error_data_for_hist.describe()
            stats["RMSE"] = np.sqrt((error_data_for_hist**2).mean())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            min_val, max_val = error_data_for_hist.min(), error_data_for_hist.max()
            if np.isclose(min_val, max_val):
                bins_array = np.array([min_val - fixed_bin_width/2, max_val + fixed_bin_width/2])
            else:
                bins_array = np.arange(min_val, max_val + fixed_bin_width, fixed_bin_width)
            
            ax.hist(error_data_for_hist, bins=bins_array, edgecolor='black', alpha=0.7, label=f'{error_col_name} Counts')
            
            ax.set_xlabel(f"{error_col_name} (m)", fontsize=15)
            ax.set_ylabel("Count", fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            
            sdb_type = "Unknown"
            if "green" in base_filename.lower(): sdb_type = "SDBgreen"
            elif "red" in base_filename.lower(): sdb_type = "SDBred"
            elif "merged" in base_filename.lower(): sdb_type = "SDBmerged"
            
            ax.set_title(f"Error Histogram for {Sensor} {sdb_type}: {AOI}", fontsize=25, fontweight='bold', y=1.05)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xlim(hist_xlim); ax.set_ylim(hist_ylim)

            stats_text = (f"Mean = {stats.get('mean', 0):.3f} m\n"
                          f"Std Dev = {stats.get('std', 0):.2f} m\n"
                          f"RMSE = {stats.get('RMSE', 0):.2f} m\n"
                          f"Count = {stats.get('count', 0):.0f}")
            if selected_min_depth is not None:
                stats_text += f"\nDepth Range = {selected_min_depth:.1f}m - {selected_max_depth:.1f}m"
            
            ax.text(0.02, 0.96, stats_text, transform=ax.transAxes, fontsize=15, ha='left', va='top',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            
            ax.legend(fontsize=15); plt.tight_layout()
            
            if save_plots:
                output_plot_filename = f"{filename_no_ext}_error_histogram.png"
                output_plot_full_path = os.path.join(output_plot_folder_path, output_plot_filename)
                plt.savefig(output_plot_full_path, dpi=300); print(f"Saved plot to: {output_plot_full_path}")
            
            if show_plots:
                plt.show()
            
            plt.close(fig)

        except Exception as e:
            print(f"An error occurred while processing {csv_file_path}: {e}")
            if fig: plt.close(fig)
    
    
    
    
    
    
    