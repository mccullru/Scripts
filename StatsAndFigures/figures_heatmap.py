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


"""Histogram scatter heat map"""

# The heatscatter function remains unchanged.
def heatscatter(ax, x, y,
                bins, title, cmap,
                xlabel, ylabel, identity_line=False,
                xlim=None, ylim=None, # User-provided limits, can contain None
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
        print(f"  Warning for plot '{title}': No finite data points left after filtering NaNs/Infs. Cannot plot hist2d.")
        if identity_line:
            lim_min_plot = max(current_plot_xlim[0], current_plot_ylim[0])
            lim_max_plot = min(current_plot_xlim[1], current_plot_ylim[1])
            if lim_min_plot < lim_max_plot:
                ax.plot([lim_min_plot, lim_max_plot], [lim_min_plot, lim_max_plot], 'k--', linewidth=1)
        return None

    hs = ax.hist2d(x_finite, y_finite, bins=bins, cmin=1, cmap=cmap, 
                    range=[current_plot_xlim, current_plot_ylim], # Pass resolved limits
                    **kwargs)
    
    if identity_line:
        lim_min_plot = max(current_plot_xlim[0], current_plot_ylim[0])
        lim_max_plot = min(current_plot_xlim[1], current_plot_ylim[1])
        if lim_min_plot < lim_max_plot:
            ax.plot([lim_min_plot, lim_max_plot], [lim_min_plot, lim_max_plot], 'k--', linewidth=1)
    
    return hs



def generate_heatmaps(input_folder, output_folder, aoi, sensor, config):

    # --- DATA SOURCE ---
    csv_folder_path = input_folder
    output_folder_path = output_folder
    AOI = aoi
    Sensor = sensor
    
    
    # Unpack config
    manual_xlim = config['manual_xlim']
    manual_ylim = config['manual_ylim']
    save_plots = config['save_plots']
    show_plots = config['show_plots']
    
    
    x_column_name = "Raster_Value"
    y_column_name = "Geoid_Corrected_Ortho_Height"


    # Find all CSV files in the specified folder
    csv_files = glob.glob(os.path.join(csv_folder_path, "*.csv"))
    
    # NEW: Create the output directory if it doesn't exist and saving is enabled
    if save_plots:
        os.makedirs(output_folder_path, exist_ok=True)
        print(f"Plots will be saved to: {output_folder_path}")
    
    if not csv_files:
        print(f"Error: No CSV files found in the folder: {csv_folder_path}")
    else:
        print(f"Found {len(csv_files)} CSV files. Generating a separate heatmap for each.\n")
    
        for file_path in csv_files:
            print(f"Processing file: {os.path.basename(file_path)}")
            base_filename = os.path.basename(file_path)
           
    
            try:
                data_df = pd.read_csv(file_path)
                
                if not (x_column_name in data_df.columns and y_column_name in data_df.columns):
                    print(f"  Warning: Skipping file. Columns ('{x_column_name}', '{y_column_name}') not found.")
                    continue
    
                x_data = pd.to_numeric(data_df[x_column_name], errors='coerce').values
                y_data = pd.to_numeric(data_df[y_column_name], errors='coerce').values
                
                if not np.any(np.isfinite(x_data) & np.isfinite(y_data)):
                    print("Warning: No valid numeric data pairs found. Skipping plot.")
                    continue
    
                fig, ax = plt.subplots(figsize=(8, 6))
    
                # Detect SDB Type from Filename
                sdb_type = "Unknown" # Default value
                base_filename_lower = base_filename.lower()
                if "green" in base_filename_lower:
                    sdb_type = "SDBgreen"
                elif "red" in base_filename_lower:
                    sdb_type = "SDBred"
                elif "merged" in base_filename_lower:
                    sdb_type = "SDBmerged"
    
                plot_title = (f"Heatmap of {Sensor} {sdb_type}: {AOI}")
    
                hist_output = heatscatter(ax, x_data, y_data,
                                          bins=100,
                                          title=plot_title,
                                          cmap='viridis',
                                          xlabel="SDB Value (m)",
                                          ylabel="Reference Bathy Values (m)",
                                          identity_line=True,
                                          xlim=manual_xlim, 
                                          ylim=manual_ylim)
    
                if hist_output:
                    if isinstance(hist_output, tuple) and len(hist_output) == 4:
                        plt.colorbar(hist_output[3], ax=ax, label='Counts per Bin')
                    else:
                        print(f"  Note: Could not create colorbar for {os.path.basename(file_path)}.")
                
                plt.tight_layout()
    
                # NEW: Save the figure if enabled
                if save_plots:
                    base_name = os.path.basename(file_path)
                    file_name_without_ext = os.path.splitext(base_name)[0]
                    output_filename = f"{file_name_without_ext}_heatmap.png"
                    full_output_path = os.path.join(output_folder_path, output_filename)
                    
                    # # Save the figure with a high resolution for better quality
                    # fig.savefig(full_output_path, dpi=300)
                    # print(f"  -> Plot saved to {full_output_path}")
    
                # NEW: Show the plot if enabled
                if show_plots:
                    plt.show()
                
                # NEW: Close the figure to free up memory, especially if not showing
                #plt.close(fig)
    
            except Exception as e:
                print(f"  An error occurred while processing file {os.path.basename(file_path)}: {e}")
            
            print("-" * 30)
    
    if csv_files: 
        print("\nFinished processing all files.")