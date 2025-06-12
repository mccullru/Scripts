# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:11:06 2025

@author: mccullru
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


"""Scatter heat map"""

def heatscatter(ax, x, y,
                bins, title, cmap,
                xlabel, ylabel, identity_line=False,
                xlim=None, ylim=None,
                **kwargs):
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_finite = x_arr[finite_mask]
    y_finite = y_arr[finite_mask]

    ax.set_xlim(xlim) 
    ax.set_ylim(ylim)

    current_plot_xlim = ax.get_xlim() 
    current_plot_ylim = ax.get_ylim()

    ax.set_title(title, fontweight='bold', fontsize=20, y=1.05)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

    if len(x_finite) == 0 or len(y_finite) == 0:
        print(f"  Warning for plot '{title}': No finite data points. Cannot plot hist2d.")
        if identity_line:
            lim_min_plot = max(current_plot_xlim[0], current_plot_ylim[0])
            lim_max_plot = min(current_plot_xlim[1], current_plot_ylim[1])
            if lim_min_plot < lim_max_plot:
                ax.plot([lim_min_plot, lim_max_plot], [lim_min_plot, lim_max_plot], 'k--', linewidth=1)
        return None

    hs = ax.hist2d(x_finite, y_finite, bins=bins, cmin=1, cmap=cmap, 
                   range=[current_plot_xlim, current_plot_ylim],
                   **kwargs)
    
    if identity_line:
        lim_min_plot = max(current_plot_xlim[0], current_plot_ylim[0])
        lim_max_plot = min(current_plot_xlim[1], current_plot_ylim[1])
        if lim_min_plot < lim_max_plot:
            ax.plot([lim_min_plot, lim_max_plot], [lim_min_plot, lim_max_plot], 'k--', linewidth=1)
    
    return hs


def generate_heatmaps(input_folder, output_folder, stats_folder, aoi, sensor, config, exclusion_path):
    """
    Generates a 2D histogram (heatmap) for each input CSV file, using a
    corrected and robust R^2 filtering logic that correctly checks both
    Indicator 2 and Indicator 1 models.
    """
    # --- Configuration ---
    csv_folder_path = input_folder
    output_folder_path = output_folder
    stats_csv_folder_path = stats_folder
    AOI = aoi
    Sensor = sensor
    
    # Unpack config
    manual_xlim = config.get('manual_xlim')
    manual_ylim = config.get('manual_ylim')
    save_plots = config.get('save_plots', True)
    show_plots = config.get('show_plots', True)
    r2_threshold_for_selection = config.get('r2_threshold', 0.7)
    
    # Column Names
    x_column_name = "Raster_Value"
    y_column_name = "Geoid_Corrected_Ortho_Height"
    stats_indicator_col = "Indicator"
    stats_r2_col = "R2 Value"

    # --- Load Exclusion List from CSV ---
    excluded_files = set()
    if os.path.exists(exclusion_path):
        try:
            df_exclude = pd.read_csv(exclusion_path)
            if 'exclusion_list' in df_exclude.columns:
                excluded_files = set(df_exclude['exclusion_list'].dropna().str.lower().tolist())
                print(f"Loaded {len(excluded_files)} files to exclude from '{os.path.basename(exclusion_path)}'.")
            else:
                print(f"Warning: Column 'exclusion_list' not found in {exclusion_path}.")
        except Exception as e:
            print(f"Warning: Could not read exclusion CSV file '{exclusion_path}'. Error: {e}")
    else:
        print(f"Warning: No exclusion list found at '{exclusion_path}'.")

    # --- Prepare for R² Filtering ---
    stats_csv_filenames = []
    if os.path.isdir(stats_csv_folder_path):
        stats_csv_filenames = [f for f in os.listdir(stats_csv_folder_path) if f.lower().endswith('.csv')]
    
    
    
    
    
    
    # --- Filter and Process Data CSV Files (with DEBUGGING) ---
    all_csv_files = glob.glob(os.path.join(csv_folder_path, "*.csv"))
    csv_files = []
    
    # --- Print the entire exclusion set once to see what's loaded ---
    print("--- DEBUG: Items in exclusion set ---")
    for item in excluded_files:
        print(f"|{item}|") # Use pipe characters to reveal hidden whitespace
    print("-" * 35)
    
    for file_path in all_csv_files:
        # Get the base name we are checking
        base_name_to_check = os.path.splitext(os.path.basename(file_path))[0].lower()
        
        # Perform the check
        if base_name_to_check not in excluded_files:
            csv_files.append(file_path)
        else:
            # This is where we print the success case
            print(f"SUCCESS: Correctly excluding '{base_name_to_check}'")
    
    
    
    
    
    
    if save_plots:
        os.makedirs(output_folder_path, exist_ok=True)
    
    if not csv_files:
        print(f"Error: No CSV files remaining to process in {csv_folder_path}"); return

    print(f"Found {len(csv_files)} CSV files to potentially process.")
    
    for file_path in csv_files:
        base_filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(base_filename)[0]
        print(f"\n--- Evaluating file: {base_filename} ---")
        
        # --- R² Filter Logic (Corrected) ---
        if stats_csv_filenames:
            expected_stats_csv_name = f"{filename_no_ext}_LR_Stats_iterations.csv"
            matched_stats_csv = next((f for f in stats_csv_filenames if f.lower() == expected_stats_csv_name.lower()), None)
            
            if not matched_stats_csv:
                print(f"  Warning: No stats file found for {base_filename}. Skipping heatmap generation.")
                continue
            
            try:
                stats_df = pd.read_csv(os.path.join(stats_csv_folder_path, matched_stats_csv))
                required_cols = [stats_indicator_col, stats_r2_col]
                if stats_df.empty or not all(col in stats_df.columns for col in required_cols):
                    print(f"  Warning: Stats file {matched_stats_csv} is empty or missing required columns. Skipping.")
                    continue
                
                stats_df.dropna(subset=required_cols, inplace=True)
                row_to_use = None
                
                
                # 1. Try to find a passing Indicator 2 model
                rows_ind2 = stats_df[stats_df[stats_indicator_col] == 2]
                if not rows_ind2.empty and round(rows_ind2.iloc[0][stats_r2_col], 2) >= r2_threshold_for_selection:
                    row_to_use = rows_ind2.iloc[0]
                else:
                    # 2. If Indicator 2 failed, FALLBACK to Indicator 1
                    rows_ind1 = stats_df[stats_df[stats_indicator_col] == 1]
                    if not rows_ind1.empty and round(rows_ind1.iloc[0][stats_r2_col], 2) >= r2_threshold_for_selection:
                        row_to_use = rows_ind1.iloc[0]

                # 3. If neither passed, now we skip
                if row_to_use is None:
                    best_r2_row = stats_df.loc[stats_df[stats_r2_col].idxmax()] if not stats_df.empty else None
                    r2_val = f"{best_r2_row[stats_r2_col]:.4f}" if best_r2_row is not None else "N/A"
                    print(f"  Skipping heatmap: Best available R² ({r2_val}) is below threshold ({r2_threshold_for_selection}).")
                    continue
                
                print(f"  R² check passed (R² = {row_to_use[stats_r2_col]:.2f}). Proceeding with heatmap generation.")

            except Exception as e:
                print(f"  Error processing stats file for {base_filename}: {e}. Skipping.")
                continue
        
        # --- Plotting Logic ---
        try:
            data_df = pd.read_csv(file_path)
            if not all(col in data_df.columns for col in [x_column_name, y_column_name]):
                print("  Warning: Skipping file. Required columns not found.")
                continue

            x_data, y_data = pd.to_numeric(data_df[x_column_name], errors='coerce'), pd.to_numeric(data_df[y_column_name], errors='coerce')
            
            fig, ax = plt.subplots(figsize=(8, 6))

            sdb_type = "Unknown"
            if "green" in base_filename.lower(): sdb_type = "SDBgreen"
            elif "red" in base_filename.lower(): sdb_type = "SDBred"
            elif "merged" in base_filename.lower(): sdb_type = "SDBmerged"
            
            plot_title = f"Heatmap of {Sensor} {sdb_type}: {AOI}"

            hist_output = heatscatter(ax, x_data, y_data, bins=100, title=plot_title, cmap='viridis',
                                      xlabel="SDB Value (m)", ylabel="Reference Bathy Values (m)",
                                      identity_line=True, xlim=manual_xlim, ylim=manual_ylim)

            if hist_output:
                plt.colorbar(hist_output[3], ax=ax, label='Counts per Bin')
            
            plt.tight_layout()

            if save_plots:
                output_filename = f"{filename_no_ext}_heatmap.png"
                plt.savefig(os.path.join(output_folder_path, output_filename), dpi=300)
                print(f"  -> Plot saved to {os.path.join(output_folder_path, output_filename)}")

            if show_plots:
                plt.show()
            
            plt.close(fig)
        except Exception as e:
            print(f"  An error occurred while processing file {base_filename}: {e}")
            if 'fig' in locals() and fig: plt.close(fig)
        
        print("-" * 30)

    print("\nFinished processing all files.")
    
    
    
    
    
    