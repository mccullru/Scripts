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



""" Individual Error Histograms """

def generate_histograms(input_folder, output_folder, aoi, sensor, config):
    
    # Specify the folder containing your input CSV files
    input_csv_folder_path = input_folder
    output_plot_folder_path = output_folder
    AOI = aoi
    Sensor = sensor
    
    # Unpack config
    fixed_bin_width = config['fixed_bin_width']
    error_filter_min, error_filter_max = config['error_filter_bounds']
    hist_xlim = config['hist_xlim']
    hist_ylim = config['hist_ylim']
    save_plots = config['save_plots'] 
    show_plots = config['show_plots'] 
    
    # Define column names
    ref_col = "Geoid_Corrected_Ortho_Height"
    SDB_col = "Raster_Value"
    error_col_name = "Error" 
    
    
    # --- Create Output Folder if it Doesn't Exist ---
    if not os.path.exists(output_plot_folder_path):
        os.makedirs(output_plot_folder_path)
        print(f"Created output folder: {output_plot_folder_path}")
    
    # --- Find and Process CSV Files ---
    csv_files = glob.glob(os.path.join(input_csv_folder_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in the folder: {input_csv_folder_path}")
    else:
        print(f"Found {len(csv_files)} CSV files to process in: {input_csv_folder_path}")
    
    for csv_file_path in csv_files:
        print(f"\n--- Processing file: {csv_file_path} ---")
        base_filename = os.path.basename(csv_file_path)
        filename_no_ext = os.path.splitext(base_filename)[0]
        fig = None # Initialize fig here to ensure it's defined for error handling
    
        try:
            df = pd.read_csv(csv_file_path)
    
            if ref_col not in df.columns or SDB_col not in df.columns:
                print(f"Warning: Required columns ('{ref_col}', '{SDB_col}') not found in {base_filename}. Skipping.")
                continue
    
            df[ref_col] = pd.to_numeric(df[ref_col], errors='coerce')
            df[SDB_col] = pd.to_numeric(df[SDB_col], errors='coerce')
            df = df.dropna(subset=[ref_col, SDB_col])
    
            if df.empty:
                print(f"Warning: No valid numeric data after coercion and NaN removal in {base_filename}. Skipping.")
                continue
    
            df[error_col_name] = df[ref_col] - df[SDB_col]
            
            # Get raw error data points
            error_data_raw = df[error_col_name].dropna()
    
            # Apply the new filter for values between error_filter_min and error_filter_max
            # These values will now be used for the histogram, KDE, and statistics
            error_data_for_hist = error_data_raw[
                (error_data_raw > error_filter_min) & (error_data_raw < error_filter_max)
            ]
    
            if error_data_for_hist.empty:
                print(f"Warning: No error data to plot for {base_filename} after filtering for [{error_filter_min}, {error_filter_max}] range. Skipping plot generation.")
                continue
    
    
    
            stats = error_data_for_hist.describe()
            if error_data_for_hist.empty: # Should be caught by above, but good for safety
                stats["RMSE"] = np.nan
            else:
                stats["RMSE"] = (error_data_for_hist.astype(float) ** 2).mean() ** 0.5
    
            print(f"Statistics for {base_filename} (filtered for [{error_filter_min}, {error_filter_max}]):")
            print(stats)
    
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Define bins based on fixed_bin_width ---
            min_val = error_data_for_hist.min()
            max_val = error_data_for_hist.max()
            # Ensure bins cover the data range appropriately, even if min_val equals max_val
            if np.isclose(min_val, max_val):
                bins_array = np.array([min_val - fixed_bin_width/2, max_val + fixed_bin_width/2])
            else:
                bins_array = np.arange(min_val, max_val + fixed_bin_width, fixed_bin_width)
            if len(bins_array) < 2: # Ensure at least one bin
                bins_array = np.array([min_val - fixed_bin_width/2, max_val + fixed_bin_width/2])
    
            counts, bin_edges, patches = ax.hist(error_data_for_hist, bins=bins_array, edgecolor='black', alpha=0.7, label=f'{error_col_name} Counts')
            
            # # Add KDE Plot
            # if len(error_data_for_hist) > 1: # KDE needs at least 2 points
            #     try:
                    
            #         # Create KDE (now using the filtered error_data_for_hist)
            #         kde = gaussian_kde(error_data_for_hist, 0.15)
                    
            #         # Create x-values for plotting the KDE curve (cover range of histogram)
            #         kde_x = np.linspace(bin_edges[0], bin_edges[-1], 200) # Use histogram bin edges for range
            #         kde_y = kde(kde_x)
    
            #         # Scale KDE to match histogram counts
            #         # (Area under KDE is 1, area under hist is N * bin_width)
            #         bin_width_kde = bin_edges[1] - bin_edges[0] # Use calculated bin_width
            #         N_kde = len(error_data_for_hist)
            #         scaled_kde_y = kde_y * N_kde * bin_width_kde
    
            #         # Plot KDE on the same axis
            #         ax.plot(kde_x, scaled_kde_y, color='red', linestyle='--', linewidth=2, label='KDE')
            #     except Exception as e_kde:
            #         print(f"Warning: Could not compute or plot KDE for {base_filename}. Error: {e_kde}")
            
            # --- ADD THIS TO CHECK BIN WIDTH ---
            if len(bin_edges) > 1: # Check if there are at least two bin edges
                bin_width_current_hist = bin_edges[1] - bin_edges[0]
                print(f"Bin width for {base_filename}: {bin_width_current_hist:.4f}")
            else:
                print(f"Could not determine bin width for {base_filename} (not enough bin_edges).")
    
            
            ax.set_xlabel(f"{error_col_name} (m)", fontsize=15)
            ax.set_ylabel("Count", fontsize=15)
            
            # Customized Tick Marks and Labels
            # Explicitly list the desired ticks
            ax.set_xticks([-10, -5, 0, 5, 10])
    
            # Increase the font size for both X and Y axis tick labels
            ax.tick_params(axis='both', which='major', labelsize=15) # Adjust '12' to your desired size
    
             # Detect SDB Type from Filename
            sdb_type = "Unknown" # Default value
            base_filename_lower = base_filename.lower()
            if "green" in base_filename_lower:
                sdb_type = "SDBgreen"
            elif "red" in base_filename_lower:
                sdb_type = "SDBred"
            elif "merged" in base_filename_lower:
                sdb_type = "SDBmerged"        
    
            plot_title = (f"Error Histogram for {Sensor} {sdb_type}: {AOI}")
            
            ax.set_title(plot_title, fontsize=25, fontweight='bold', y=1.05)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            ax.set_xlim(hist_xlim) # hist_xlim is still (-2, 2) but data is up to (-10, 10)
            if hist_ylim:
                ax.set_ylim(hist_ylim)
    
            stats_text = (f"Mean = {stats.get('mean', np.nan):.3f} m\n"
                          f"Min = {stats.get('min', np.nan):.2f} m\n"
                          f"Max = {stats.get('max', np.nan):.2f} m\n"
                          f"Std Dev = {stats.get('std', np.nan):.2f} m\n"
                          f"RMSE = {stats.get('RMSE', np.nan):.2f} m\n"
                          f"Count = {stats.get('count', 0):.0f}")
            
            current_text_x = hist_xlim[0] + (hist_xlim[1] - hist_xlim[0]) * 0.05
            current_text_y = (ax.get_ylim()[1]) * 0.95
            
            ax.text(current_text_x, current_text_y, stats_text, fontsize=18, color='black', ha='left', va='top',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            
            ax.legend(fontsize=15) # Add legend to show histogram and KDE labels
            plt.tight_layout()
    
            output_plot_filename = f"{filename_no_ext}_error_histogram_kde.png" 
            output_plot_full_path = os.path.join(output_plot_folder_path, output_plot_filename)
            
            # Conditional Save and Show ---
            if save_plots:
                try:
                    plt.savefig(output_plot_full_path, dpi=300)
                    print(f"Saved plot to: {output_plot_full_path}")
                except Exception as e_save:
                    print(f"ERROR: Failed to save plot to {output_plot_full_path}. Error: {e_save}")
            
            if show_plots:
                plt.show()
        
            # Close the figure to free memory
            plt.close(fig)
    
        except FileNotFoundError:
            print(f"Error: File not found at {csv_file_path}. Skipping.")
        except pd.errors.EmptyDataError:
            print(f"Warning: CSV file {base_filename} is empty. Skipping.")
        except KeyError as e:
            print(f"Warning: Column missing in {base_filename} - {e}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {csv_file_path}: {e}")
            if fig and plt.fignum_exists(fig.number): # Ensure fig exists before trying to close
                plt.close(fig)
    
    print("\n--- All CSV files processed. ---")
