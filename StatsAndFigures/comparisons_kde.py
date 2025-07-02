# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 12:23:38 2025

@author: mccullru
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import mplcursors

from scipy.stats import gaussian_kde, skew, kurtosis
from difflib import get_close_matches

def _process_sensor_data(input_folder, stats_folder, config, exclusion_path, sensor_name, sdb_type_filter):
    """
    Helper function to process data for a single sensor (e.g., Sentinel-2 or SuperDove),
    filtering for a specific SDB type (e.g., 'red', 'green', 'merged').
    Pools all valid error data and returns it for KDE calculation.
    Returns: pooled_errors_array, pooled_stats, processed_files_count
    """
    print(f"\n--- Processing data for {sensor_name} (SDB Type: {sdb_type_filter}) ---")

    # Unpack config
    apply_depth_filter = config.get('apply_depth_filter', True)
    r2_threshold_for_selection = config.get('r2_threshold', 0.7)
    kde_error_filter_min, kde_error_filter_max = config.get('error_filter_bounds')

    # Column names for SDB_ExtractedPts files (main data)
    ref_col = "Geoid_Corrected_Ortho_Height"
    SDB_col = "Raster_Value"
    error_col_name = "Error"

    # Column names expected in the stats CSVs
    stats_indicator_col = "Indicator"
    stats_r2_col = "R2 Value"
    stats_max_depth_col = "Max Depth Range"

    all_error_data_pooled = []
    processed_files_count = 0 # This will count files that successfully contribute points
    skipped_files_count = 0
    total_original_lines_sensor = 0
    total_lines_removed_by_filter_sensor = 0

    # Load exclusion list
    excluded_files = set()
    if os.path.exists(exclusion_path):
        try:
            df_exclude = pd.read_csv(exclusion_path)
            if 'exclusion_list' in df_exclude.columns:
                excluded_files = set(df_exclude['exclusion_list'].dropna().str.lower().tolist())
        except Exception as e:
            print(f"Warning: Could not read exclusion CSV for {sensor_name} at '{exclusion_path}'. Error: {e}")

    # Prepare for depth filtering (find stats CSVs)
    stats_csv_filenames = []
    local_apply_depth_filter = apply_depth_filter # Use a local copy
    if local_apply_depth_filter:
        if os.path.isdir(stats_folder):
            stats_csv_filenames = [f for f in os.listdir(stats_folder) if f.lower().endswith('.csv')]
            if not stats_csv_filenames:
                print(f"  Warning: No CSV files found in {sensor_name} stats folder: {stats_folder}. Depth filtering disabled.")
                local_apply_depth_filter = False
        else:
            print(f"  Warning: {sensor_name} stats folder not found: {stats_folder}. Depth filtering disabled.")
            local_apply_depth_filter = False

    # Find and process data CSV files for the current sensor and SDB type
    # Filter by sdb_type_filter (e.g., 'red', 'green', 'merged')
    sdb_type_keyword = sdb_type_filter.lower().replace("sdb_", "") # e.g., 'red' from 'SDB_red'

    all_input_csv_files_in_folder = glob.glob(os.path.join(input_folder, "*.csv"))

    # Apply exclusion list filter
    original_count_input_files = len(all_input_csv_files_in_folder)
    
    # Filter by exclusion list, then by SDB type keyword
    input_csv_files_filtered = [
        fpath for fpath in all_input_csv_files_in_folder
        if os.path.splitext(os.path.basename(fpath))[0].lower() not in excluded_files
    ]
    
    # Filter by the specific SDB type keyword
    input_csv_files_to_process = [
        fpath for fpath in input_csv_files_filtered
        if sdb_type_keyword in os.path.basename(fpath).lower()
    ]

    files_removed_by_exclusion = original_count_input_files - len(input_csv_files_filtered)
    if files_removed_by_exclusion > 0:
        print(f"  {sensor_name}: Filtered out {files_removed_by_exclusion} files based on the exclusion list.")

    if not input_csv_files_to_process:
        print(f"  {sensor_name}: No input data CSV files found matching '{sdb_type_keyword}' after exclusion filter. Skipping data collection.")
        return np.array([]), {}, 0 # Return 0 for processed_files_count

    print(f"  {sensor_name}: Found {len(input_csv_files_to_process)} data CSVs to process for SDB type '{sdb_type_keyword}'.")

    for i, data_csv_path in enumerate(input_csv_files_to_process):
        base_data_filename = os.path.basename(data_csv_path)
        data_filename_no_ext = os.path.splitext(base_data_filename)[0]

        selected_max_depth = None # Reset for each file

        if local_apply_depth_filter:
            expected_stats_csv_name = f"{data_filename_no_ext}_LR_Stats_iterations.csv"
            matched_stats_csv_name = None

            for c_name in stats_csv_filenames:
                if c_name.lower() == expected_stats_csv_name.lower():
                    matched_stats_csv_name = c_name
                    break
            if not matched_stats_csv_name:
                close_matches = get_close_matches(expected_stats_csv_name, stats_csv_filenames, n=1, cutoff=0.85)
                if close_matches:
                    matched_stats_csv_name = close_matches[0]

            if matched_stats_csv_name:
                stats_csv_full_path = os.path.join(stats_folder, matched_stats_csv_name)
                try:
                    stats_df = pd.read_csv(stats_csv_full_path)
                    if stats_df.empty:
                        print(f"    {base_data_filename}: Warning: Matched stats CSV '{matched_stats_csv_name}' is empty. Skipping depth selection.")
                    else:
                        cols_to_check_stats = [stats_indicator_col, stats_r2_col, stats_max_depth_col]
                        if not all(col in stats_df.columns for col in cols_to_check_stats):
                            print(f"    {base_data_filename}: Warning: Stats CSV {matched_stats_csv_name} is missing columns: {cols_to_check_stats}. Skipping depth selection.")
                        else:
                            stats_df[stats_r2_col] = pd.to_numeric(stats_df[stats_r2_col], errors='coerce')
                            stats_df[stats_indicator_col] = pd.to_numeric(stats_df[stats_indicator_col], errors='coerce')
                            stats_df[stats_max_depth_col] = pd.to_numeric(stats_df[stats_max_depth_col], errors='coerce')
                            cleaned_stats_df = stats_df.dropna(subset=[stats_indicator_col, stats_r2_col, stats_max_depth_col]).copy()

                            if cleaned_stats_df.empty:
                                print(f"    {base_data_filename}: Warning: No valid rows in {matched_stats_csv_name} after cleaning NaNs. Skipping depth selection.")
                            else:
                                row_to_use_for_depth = None
                                rows_ind2 = cleaned_stats_df[cleaned_stats_df[stats_indicator_col] == 2]
                                if not rows_ind2.empty:
                                    temp_row_ind2 = rows_ind2.iloc[0]
                                    r2_val_ind2 = temp_row_ind2[stats_r2_col]
                                    r2_check_ind2 = round(r2_val_ind2, 2)
                                    if r2_check_ind2 >= r2_threshold_for_selection:
                                        row_to_use_for_depth = temp_row_ind2
                                    else:
                                        rows_ind1 = cleaned_stats_df[cleaned_stats_df[stats_indicator_col] == 1]
                                        if not rows_ind1.empty:
                                            temp_row_ind1 = rows_ind1.iloc[0]
                                            r2_val_ind1 = temp_row_ind1[stats_r2_col]
                                            r2_check_ind1 = round(r2_val_ind1, 2)
                                            if r2_check_ind1 >= r2_threshold_for_selection:
                                                row_to_use_for_depth = temp_row_ind1

                                if row_to_use_for_depth is not None:
                                    selected_max_depth = row_to_use_for_depth[stats_max_depth_col]
                                else:
                                    print(f"    {base_data_filename}: Could not determine 'Max Depth Range' from {matched_stats_csv_name} (R2 criteria not met). Skipping depth filter for this file.")
                except FileNotFoundError:
                    print(f"    {base_data_filename}: Warning: Matched stats CSV '{matched_stats_csv_name}' not found. Skipping depth filter for this file.")
                except Exception as e_coeff:
                    print(f"    {base_data_filename}: Error processing stats CSV {matched_stats_csv_name}: {e_coeff}. Skipping depth filter for this file.")

        try:
            df = pd.read_csv(data_csv_path)
            original_lines_in_file = len(df)
            total_original_lines_sensor += original_lines_in_file

            if not (ref_col in df.columns and SDB_col in df.columns):
                print(f"    {base_data_filename}: Main data CSV missing '{ref_col}' or '{SDB_col}'. Skipping.")
                skipped_files_count += 1
                total_lines_removed_by_filter_sensor += original_lines_in_file
                continue

            df[ref_col] = pd.to_numeric(df[ref_col], errors='coerce')
            df[SDB_col] = pd.to_numeric(df[SDB_col], errors='coerce')
            df.dropna(subset=[ref_col, SDB_col], inplace=True)

            lines_removed_nan = original_lines_in_file - len(df)
            total_lines_removed_by_filter_sensor += lines_removed_nan

            if df.empty:
                print(f"    {base_data_filename}: No valid numeric data after NaN removal. Skipping.")
                skipped_files_count += 1
                continue

            if local_apply_depth_filter and pd.notna(selected_max_depth):
                current_point_count_before_depth_filter = len(df)
                min_depth_cutoff = 0.0
                df = df[(df[ref_col] >= min_depth_cutoff) & (df[ref_col] <= selected_max_depth)]

                lines_removed_depth_filter = current_point_count_before_depth_filter - len(df)
                total_lines_removed_by_filter_sensor += lines_removed_depth_filter

                if df.empty:
                    print(f"    {base_data_filename}: No data points remain after depth filtering. Skipping.")
                    skipped_files_count += 1
                    continue
            elif local_apply_depth_filter and selected_max_depth is None:
                print(f"    {base_data_filename}: Depth filtering was ON, but no Max Depth was selected. Skipping.")
                skipped_files_count += 1
                continue

            df[error_col_name] = df[SDB_col] - df[ref_col]
            error_data_raw = df[error_col_name].dropna()

            error_data_filtered = error_data_raw[
                (error_data_raw > kde_error_filter_min) & (error_data_raw < kde_error_filter_max)
            ]

            lines_removed_kde_filter = len(error_data_raw) - len(error_data_filtered)
            total_lines_removed_by_filter_sensor += lines_removed_kde_filter

            if error_data_filtered.empty or len(error_data_filtered) < 2:
                print(f"    {base_data_filename}: Not enough error data points ({len(error_data_filtered)}) for KDE. Skipping.")
                skipped_files_count += 1
                continue

            all_error_data_pooled.extend(error_data_filtered.values)
            processed_files_count += 1 # Increment count only if file successfully contributes data

        except pd.errors.EmptyDataError:
            print(f"    {base_data_filename}: Error: CSV is empty. Skipping.")
            skipped_files_count += 1
            total_lines_removed_by_filter_sensor += original_lines_in_file # Count all lines if file is empty
            continue
        except Exception as e:
            print(f"    {base_data_filename}: An error occurred: {e}. Skipping this file.")
            skipped_files_count += 1
            total_lines_removed_by_filter_sensor += original_lines_in_file # Count all lines if error prevents processing
            continue

    pooled_errors_array = np.array(all_error_data_pooled)

    print(f"\n--- {sensor_name} (SDB Type: {sdb_type_filter}) Processing Summary ---")
    print(f"  Processed {processed_files_count} files, skipped {skipped_files_count} files.")
    print(f"  Total original lines: {total_original_lines_sensor}")
    print(f"  Total lines removed by all filters: {total_lines_removed_by_filter_sensor}")
    print(f"  Total points pooled for {sensor_name} KDE: {len(pooled_errors_array)}")

    if len(pooled_errors_array) < 2:
        print(f"  Not enough data points ({len(pooled_errors_array)}) to compute KDE for {sensor_name}.")
        return np.array([]), {}, 0 # Return 0 for processed_files_count

    # Calculate statistics for the pooled data
    pooled_stats = {}
    pooled_stats['mean'] = np.mean(pooled_errors_array)
    pooled_stats['min'] = np.min(pooled_errors_array)
    pooled_stats['max'] = np.max(pooled_errors_array)
    pooled_stats['mae'] = np.mean(np.abs(pooled_errors_array))
    pooled_stats['std'] = np.std(pooled_errors_array)
    pooled_stats['skewness'] = skew(pooled_errors_array)
    pooled_stats['kurtosis'] = kurtosis(pooled_errors_array, fisher=True)
    pooled_stats['count'] = len(pooled_errors_array)

    return pooled_errors_array, pooled_stats, processed_files_count # Return processed_files_count


def generate_averaged_kde_plot(sentinel2_input_folder,
                               sentinel2_stats_folder,
                               superdove_input_folder,
                               superdove_stats_folder,
                               output_folder,
                               aoi,
                               sensor_combined_name, 
                               sdb_type_to_compare, 
                               config,
                               exclusion_path):

    plot_title_sdb_type = sdb_type_to_compare.replace("_", " ") # For cleaner title

    print(f"\n--- Generating Averaged KDE Plot for {sensor_combined_name} ({plot_title_sdb_type}) for AOI: {aoi} ---")
    print(f"  Sentinel-2 data from: {sentinel2_input_folder}")
    print(f"  SuperDove data from: {superdove_input_folder}")

    # --- Configuration ---
    output_plot_folder_path = output_folder
    plot_xlim = config['plot_xlim']
    plot_ylim = config['plot_ylim']
    fixed_bin_width = 0.1 # Used for scaling KDE, from original script

    # Plot Settings for the average lines
    s2_kde_color = 'blue'
    s2_kde_linestyle = '-'
    sd_kde_color = 'red'
    sd_kde_linestyle = '--'
    average_linewidth = 2.5
    average_alpha = 0.8
    average_zorder = 10 # Ensure average lines are on top

    if not os.path.exists(output_plot_folder_path):
        os.makedirs(output_plot_folder_path)
        print(f"Created output folder: {output_plot_folder_path}")

    # --- Process data for each sensor, now with SDB type filter ---
    s2_pooled_errors, s2_stats, s2_files_count = _process_sensor_data( # <-- S2: Get file count
        sentinel2_input_folder, sentinel2_stats_folder, config, exclusion_path, "Sentinel-2", sdb_type_to_compare
    )

    sd_pooled_errors, sd_stats, sd_files_count = _process_sensor_data( # <-- SD: Get file count
        superdove_input_folder, superdove_stats_folder, config, exclusion_path, "SuperDove", sdb_type_to_compare
    )

    # --- Calculate KDE for pooled data ---
    plot_data_for_kde = []
    max_scaled_kde_y = 0
    common_kde_x = None

    # Determine common_kde_x range from all pooled errors
    all_combined_errors = []
    if s2_pooled_errors.size > 0:
        all_combined_errors.extend(s2_pooled_errors)
    if sd_pooled_errors.size > 0:
        all_combined_errors.extend(sd_pooled_errors)

    if not all_combined_errors:
        print(f"Error: No data available from either Sentinel-2 or SuperDove for '{sdb_type_to_compare}' to generate KDE plot.")
        return


    pooled_min = np.min(all_combined_errors)
    pooled_max = np.max(all_combined_errors)
    x_padding = (pooled_max - pooled_min) * 0.1 if (pooled_max - pooled_min) > 0 else 1.0
    x_min_kde_calc = pooled_min - x_padding
    x_max_kde_calc = pooled_max + x_padding
    common_kde_x = np.linspace(x_min_kde_calc, x_max_kde_calc, 400)


    # Generate KDE for Sentinel-2
    if s2_pooled_errors.size >= 2:
        try:
            kde_s2 = gaussian_kde(s2_pooled_errors, bw_method=config.get('kde_bw_method', 0.15))
            
            # Calculate scaling factor based on average points per file
            s2_scaling_factor = (s2_stats['count'] / s2_files_count) if s2_files_count > 0 else 0
            scaled_kde_y_s2 = kde_s2(common_kde_x) * s2_scaling_factor * fixed_bin_width # <-- Adjusted scaling
            
            plot_data_for_kde.append({
                'label': 'Sentinel-2', # Label still shows total N
                'x': common_kde_x,
                'y': scaled_kde_y_s2,
                'color': s2_kde_color,
                'linestyle': s2_kde_linestyle,
                'stats': s2_stats,
                'files_count': s2_files_count # Store file count for textbox
            })
            if scaled_kde_y_s2.size > 0:
                current_max_y = np.max(scaled_kde_y_s2)
                if current_max_y > max_scaled_kde_y: max_scaled_kde_y = current_max_y
        except np.linalg.LinAlgError as lae:
            print(f"Warning: LinAlgError computing KDE for Sentinel-2 ({sdb_type_to_compare}): {lae}. Skipping S2 KDE line.")
        except ValueError as ve:
            print(f"Warning: ValueError computing KDE for Sentinel-2 ({sdb_type_to_compare}): {ve}. Skipping S2 KDE line.")
    else:
        print(f"Warning: Not enough data points ({s2_pooled_errors.size}) for Sentinel-2 ({sdb_type_to_compare}) KDE.")

    # Generate KDE for SuperDove
    if sd_pooled_errors.size >= 2:
        try:
            kde_sd = gaussian_kde(sd_pooled_errors, bw_method=config.get('kde_bw_method', 0.15))
            # Calculate scaling factor based on average points per file
            sd_scaling_factor = (sd_stats['count'] / sd_files_count) if sd_files_count > 0 else 0
            scaled_kde_y_sd = kde_sd(common_kde_x) * sd_scaling_factor * fixed_bin_width # <-- Adjusted scaling
            plot_data_for_kde.append({
                'label': 'SuperDove', # Label still shows total N
                'x': common_kde_x,
                'y': scaled_kde_y_sd,
                'color': sd_kde_color,
                'linestyle': sd_kde_linestyle,
                'stats': sd_stats,
                'files_count': sd_files_count # Store file count for textbox
            })
            
            if scaled_kde_y_sd.size > 0:
                current_max_y = np.max(scaled_kde_y_sd)
                if current_max_y > max_scaled_kde_y: max_scaled_kde_y = current_max_y
        except np.linalg.LinAlgError as lae:
            print(f"Warning: LinAlgError computing KDE for SuperDove ({sdb_type_to_compare}): {lae}. Skipping SuperDove KDE line.")
        except ValueError as ve:
            print(f"Warning: ValueError computing KDE for SuperDove ({sdb_type_to_compare}): {ve}. Skipping SuperDove KDE line.")
    else:
        print(f"Warning: Not enough data points ({sd_pooled_errors.size}) for SuperDove ({sdb_type_to_compare}) KDE.")


    if not plot_data_for_kde:
        print(f"No valid KDE lines to plot for {sdb_type_to_compare} after processing both sensors. Exiting.")
        return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))

    plotted_lines = []
    legend_handles = []

    # Plot each averaged KDE line
    for data_item in plot_data_for_kde:
        line, = ax.plot(data_item['x'], data_item['y'],
                        color=data_item['color'],
                        linestyle=data_item['linestyle'],
                        linewidth=average_linewidth,
                        alpha=average_alpha,
                        label=data_item['label'])
        plotted_lines.append(line)
        legend_handles.append(line)

    # Common plot settings
    ax.set_xlabel("Error (m)", fontsize=15)
    ax.set_ylabel("Count", fontsize=15) # Y-axis represents scaled density, so "Count" is appropriate
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title(f"Distributions of {sensor_combined_name} {plot_title_sdb_type} Error: {aoi}", fontsize=25, fontweight='bold', y=1.05)
    ax.set_xlim(plot_xlim)
    
    # Adjust ylim if max_scaled_kde_y exceeds config's ylim
    if max_scaled_kde_y > plot_ylim[1]:
        ax.set_ylim(0, max_scaled_kde_y * 1.1) # Add 10% buffer
    else:
        ax.set_ylim(plot_ylim)
    ax.grid(True, linestyle='--', alpha=0.7)


    # --- Statistics Textboxes (one for each sensor's average KDE) ---
    s2_stats_text = "N/A"
    if s2_stats:
        s2_stats_text = (f"Sentinel-2 Stats ({plot_title_sdb_type}):\n"
                         f"Mean = {s2_stats.get('mean', 0):.3f} m\n"
                         f"Min = {s2_stats.get('min', 0):.2f} m\n"
                         f"Max = {s2_stats.get('max', 0):.2f} m\n"
                         f"MAE = {s2_stats.get('mae', 0):.2f} m\n"
                         f"Std Dev = {s2_stats.get('std', 0):.2f} m\n"
                         f"Skewness = {s2_stats.get('skewness', 0):.2f}\n"
                         f"Kurtosis = {s2_stats.get('kurtosis', 0):.2f}\n"
                         f"Total Pts = {s2_stats.get('count', 0):.0f}\n" 
                         f"Files Used = {s2_files_count:.0f}") 
    ax.text(0.02, 0.98, s2_stats_text, transform=ax.transAxes, fontsize=12, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.75, edgecolor=s2_kde_color))


    sd_stats_text = "N/A"
    if sd_stats:
        sd_stats_text = (f"SuperDove Stats ({plot_title_sdb_type}):\n"
                         f"Mean = {sd_stats.get('mean', 0):.3f} m\n"
                         f"Min = {sd_stats.get('min', 0):.2f} m\n"
                         f"Max = {sd_stats.get('max', 0):.2f} m\n"
                         f"MAE = {sd_stats.get('mae', 0):.2f} m\n"
                         f"Std Dev = {sd_stats.get('std', 0):.2f} m\n"
                         f"Skewness = {sd_stats.get('skewness', 0):.2f}\n"
                         f"Kurtosis = {sd_stats.get('kurtosis', 0):.2f}\n"
                         f"Total Pts = {sd_stats.get('count', 0):.0f}\n" 
                         f"Files Used = {sd_files_count:.0f}") 
    ax.text(0.98, 0.98, sd_stats_text, transform=ax.transAxes, fontsize=12, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.75, edgecolor=sd_kde_color))

    # --- Interactive Plot Functions ---
    if plotted_lines:
        cursor = mplcursors.cursor(plotted_lines, hover=True)
        @cursor.connect("add")
        def on_add_cursor(sel):
            text = f"{sel.artist.get_label()}\nError: {sel.target[0]:.2f}\nDensity: {sel.target[1]:.2f}"
            sel.annotation.set_text(text)
            sel.annotation.get_bbox_patch().set(alpha=0.85, facecolor='lightyellow')

    if legend_handles: ax.legend(handles=legend_handles, fontsize=15, loc='lower left')
    plt.tight_layout()

    # Define output filename based on the new structure
    combined_plot_filename = f"{aoi}_{sdb_type_to_compare}_Averaged_Error_KDE.png"
    output_plot_full_path = os.path.join(output_plot_folder_path, combined_plot_filename)

    # --- Conditional Save and Show ---
    if config.get('save_plots', False):
        try:
            plt.savefig(output_plot_full_path, dpi=300)
            print(f"\nSUCCESS: Averaged KDE plot saved to: {output_plot_full_path}")
        except Exception as e_save:
            print(f"\nERROR: Failed to save averaged KDE plot to {output_plot_full_path}. Error: {e_save}")

    if config.get('show_plots', True):
        plt.show()

    plt.close(fig) # Close the figure to free memory
    print(f"\n--- Averaged KDE Plot Generation Complete for {sensor_combined_name} ({plot_title_sdb_type}) ---")








