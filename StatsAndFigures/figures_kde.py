# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:11:05 2025

@author: mccullru
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import mplcursors
import matplotlib.lines as mlines

from scipy.stats import gaussian_kde, skew, kurtosis
from difflib import get_close_matches



""" A single plot with multiple Kernel Density Estimates (KDE) which represent the histogram values without
cluttering the chart. Also adds an average line from all the plotted lines """


""" 
!! NOTE !!: If you want to be able to use the interactive plot settings to look at individual line info, have
            to go to Preferences > IPython console > Graphics tab > Backend: "Inline" for plot placement on right, 
            "Automatic" for a separate window to pop-up and allow interactions

!! OTHER WEIRD NOTE !!: When in interactive mode, in order to look at the name for each individual line, don't
                        just click on it, you have to hold the mouse button down and hover over the line.
                        It's weird, I know

"""




def generate_kde_plots(input_folder, stats_folder, output_folder, aoi, sensor, config):
    
    # --- Configuration ---
    input_csv_folder_path = input_folder
    output_plot_folder_path = output_folder
    stats_csv_folder_path = stats_folder
    AOI = aoi
    Sensor = sensor
    
    # Unpack the config dictionary
    SDB_TYPES_TO_PROCESS = config['sdb_types_to_process']
    apply_depth_filter = config['apply_depth_filter']
    r2_threshold_for_selection = config['r2_threshold']
    plot_xlim = config['plot_xlim']
    plot_ylim = config['plot_ylim']
    kde_error_filter_min, kde_error_filter_max = config['error_filter_bounds']
    save_plots = config['save_plots']
    show_plots = config['show_plots']
    

    # --- Column Names for SDB_ExtractedPts files (main data) ---
    ref_col = "Geoid_Corrected_Ortho_Height"
    SDB_col = "Raster_Value"
    error_col_name = "Error"
    
    # Plot Settings
    individual_kde_color = 'gray'
    individual_kde_linestyle = '-'
    individual_kde_linewidth = 1.5
    fixed_bin_width = 0.1
    
    
    # Column names expected in the stats CSVs
    stats_indicator_col = "Indicator"
    stats_r2_col = "R2 Value"
    stats_max_depth_col = "Max Depth Range"
    
    
    if not os.path.exists(output_plot_folder_path):
        os.makedirs(output_plot_folder_path)
        print(f"Created output folder: {output_plot_folder_path}")
    
    # --- Outer loop to process each SDB_type category ---
    for current_sdb_type in SDB_TYPES_TO_PROCESS:
        print(f"\n======== Processing SDB_type: {current_sdb_type} ========")
    
        # Reset data collections for each SDB_type iteration
        all_error_data_for_overall_range = []
        datasets_for_kde = []
        total_original_lines = 0
        total_lines_removed_by_filter = 0
        removed_file_names = []
    
        # Re-evaluate stats_csv_filenames and apply_depth_filter for each run
        # (This section remains mostly as it was, but is now inside the loop)
        local_apply_depth_filter = apply_depth_filter # Use a local copy for potential modification
        stats_csv_filenames = []
        if local_apply_depth_filter:
            if os.path.isdir(stats_csv_folder_path):
                stats_csv_filenames = [f for f in os.listdir(stats_csv_folder_path) if f.lower().endswith('.csv')]
                if not stats_csv_filenames:
                    print(f"Warning: No CSV files found in stats folder: {stats_csv_folder_path}. Depth filtering will be disabled for this run.")
                    local_apply_depth_filter = False
            else:
                print(f"Warning: Stats folder not found: {stats_csv_folder_path}. Depth filtering will be disabled for this run.")
                local_apply_depth_filter = False
    
        all_input_csv_files_in_folder = glob.glob(os.path.join(input_csv_folder_path, "*.csv"))
    
        # Determine which files to process based on the current SDB_type
        sdb_type_keyword_to_match = current_sdb_type.lower().replace("sdb_", "")
        input_csv_files_to_process = []
    
        if current_sdb_type == 'SDB': # 'SDB' means all files, no specific keyword filter
            input_csv_files_to_process = all_input_csv_files_in_folder
            print(f"SDB_type='{current_sdb_type}'. Will attempt to process all {len(input_csv_files_to_process)} CSVs found in input folder.")
        elif sdb_type_keyword_to_match:
            input_csv_files_to_process = [
                f for f in all_input_csv_files_in_folder if sdb_type_keyword_to_match in os.path.basename(f).lower()
            ]
            print(f"Based on SDB_type='{current_sdb_type}', filtered to {len(input_csv_files_to_process)} input data files matching '{sdb_type_keyword_to_match}'.")
        else: # Fallback if SDB_type is something unexpected or empty after replace
            input_csv_files_to_process = []
            print(f"Warning: Invalid or unhandled SDB_type '{current_sdb_type}'. No files will be processed for this type.")
    
    
        if not input_csv_files_to_process:
            print(f"No input data CSV files to process for SDB_type '{current_sdb_type}' (after SDB_type filter if applied). Skipping plot generation for this type.")
            continue # Skip to the next SDB_type if no files for current one
    
        # --- Data Processing Loop (original content, now inside the outer loop) ---
        for i, data_csv_path in enumerate(input_csv_files_to_process):
            print(f"\n--- Processing data file: {os.path.basename(data_csv_path)} ---")
            base_data_filename = os.path.basename(data_csv_path)
            data_filename_no_ext = os.path.splitext(base_data_filename)[0]
    
            selected_max_depth = None
    
            if local_apply_depth_filter and stats_csv_filenames:
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
                        print(f"  Note: Used get_close_matches. Found stats CSV: {matched_stats_csv_name} for expected pattern {expected_stats_csv_name}")
    
                if matched_stats_csv_name:
                    stats_csv_path = os.path.join(stats_csv_folder_path, matched_stats_csv_name)
                    print(f"  Using stats CSV: {matched_stats_csv_name}")
                    try:
                        stats_df = pd.read_csv(stats_csv_path)
                        if stats_df.empty:
                            print(f"  Warning: Matched stats CSV '{matched_stats_csv_name}' is empty. Skipping depth selection.")
                        else:
                            cols_to_check_stats = [stats_indicator_col, stats_r2_col, stats_max_depth_col]
                            if not all(col in stats_df.columns for col in cols_to_check_stats):
                                print(f"  Warning: Stats CSV {matched_stats_csv_name} is missing one or more required columns: {cols_to_check_stats}. Skipping depth selection.")
                            else:
                                stats_df[stats_r2_col] = pd.to_numeric(stats_df[stats_r2_col], errors='coerce')
                                stats_df[stats_indicator_col] = pd.to_numeric(stats_df[stats_indicator_col], errors='coerce')
                                stats_df[stats_max_depth_col] = pd.to_numeric(stats_df[stats_max_depth_col], errors='coerce')
                                cleaned_stats_df = stats_df.dropna(subset=[stats_indicator_col, stats_r2_col, stats_max_depth_col]).copy()
    
                                if cleaned_stats_df.empty:
                                    print(f"  Warning: No valid rows in {matched_stats_csv_name} after cleaning NaNs from essential columns. Skipping depth selection.")
                                else:
                                    row_to_use_for_depth = None
                                    rows_ind2 = cleaned_stats_df[cleaned_stats_df[stats_indicator_col] == 2]
                                    if not rows_ind2.empty:
                                        temp_row_ind2 = rows_ind2.iloc[0]
                                        r2_val_ind2 = temp_row_ind2[stats_r2_col]
                                        r2_check_ind2 = round(r2_val_ind2, 2)
                                        if r2_check_ind2 >= r2_threshold_for_selection:
                                            row_to_use_for_depth = temp_row_ind2
                                            print(f"  Selected 'Max Depth Range' using Indicator 2 (R2={r2_check_ind2:.2f}).")
                                        else:
                                            print(f"  Indicator 2 R2 ({r2_check_ind2:.2f}) < {r2_threshold_for_selection}. Checking Indicator 1.")
                                            rows_ind1 = cleaned_stats_df[cleaned_stats_df[stats_indicator_col] == 1]
                                            if not rows_ind1.empty:
                                                temp_row_ind1 = rows_ind1.iloc[0]
                                                r2_val_ind1 = temp_row_ind1[stats_r2_col]
                                                r2_check_ind1 = round(r2_val_ind1, 2)
                                                if r2_check_ind1 >= r2_threshold_for_selection:
                                                    row_to_use_for_depth = temp_row_ind1
                                                    print(f"  Fell back to 'Max Depth Range' from Indicator 1 (R2={r2_check_ind1:.2f}).")
                                                else:
                                                    print(f"  Indicator 1 R2 ({r2_check_ind1:.2f}) also < {r2_threshold_for_selection}.")
                                            else:
                                                print(f"  No Indicator 1 row found for fallback in {matched_stats_csv_name}.")
                                    else:
                                        print(f"  No valid Indicator 2 row found in {matched_stats_csv_name} to evaluate R2.")
    
                                    if row_to_use_for_depth is not None:
                                        selected_max_depth = row_to_use_for_depth[stats_max_depth_col]
                                        print(f"  Determined 'Max Depth Range' for filtering: {selected_max_depth:.2f} m")
                                    else:
                                        print(f"  Could not determine 'Max Depth Range' from {matched_stats_csv_name} for {base_data_filename} (R2 criteria not met). Skipping this file for KDE.")
                                        removed_file_names.append(base_data_filename)
                                        continue
                    except FileNotFoundError:
                        print(f"  Warning: Matched stats CSV '{matched_stats_csv_name}' was not found at path: {stats_csv_path}. Skipping depth filter for this file.")
                        selected_max_depth = None
                    except Exception as e_coeff:
                        print(f"  Error processing stats CSV {matched_stats_csv_name} for {base_data_filename}: {e_coeff}")
                        selected_max_depth = None
                else:
                    print(f"  Warning: No matching stats CSV found for {data_filename_no_ext} (expected ~{expected_stats_csv_name}). Skipping this file for KDE plotting.")
                    removed_file_names.append(base_data_filename)
                    continue
    
            try:
                df = pd.read_csv(data_csv_path)
                original_lines_in_file = len(df)
                total_original_lines += original_lines_in_file
    
                if not (ref_col in df.columns and SDB_col in df.columns):
                    print(f"  Main data CSV {base_data_filename} missing '{ref_col}' or '{SDB_col}'. Skipping.")
                    total_lines_removed_by_filter += original_lines_in_file
                    removed_file_names.append(base_data_filename)
                    continue
    
                df[ref_col] = pd.to_numeric(df[ref_col], errors='coerce')
                df[SDB_col] = pd.to_numeric(df[SDB_col], errors='coerce')
                df.dropna(subset=[ref_col, SDB_col], inplace=True)
    
                lines_removed_nan = original_lines_in_file - len(df)
                total_lines_removed_by_filter += lines_removed_nan
    
                if df.empty:
                    print(f"  No valid numeric data in {base_data_filename} after initial NaN removal. Skipping.")
                    if original_lines_in_file > 0 and lines_removed_nan == original_lines_in_file:
                        removed_file_names.append(base_data_filename)
                    continue
    
                if local_apply_depth_filter and pd.notna(selected_max_depth):
                    current_point_count_before_depth_filter = len(df)
                    min_depth_cutoff = 0.0
                    df = df[(df[ref_col] >= min_depth_cutoff) & (df[ref_col] <= selected_max_depth)]
    
                    lines_removed_depth_filter = current_point_count_before_depth_filter - len(df)
                    total_lines_removed_by_filter += lines_removed_depth_filter
    
                    print(f"  Filtered data for {base_data_filename} using Y-axis ({ref_col}) range: {min_depth_cutoff:.2f}m - {selected_max_depth:.2f} m. Points: {current_point_count_before_depth_filter} -> {len(df)}.")
                    if df.empty:
                        print(f"  No data points remain in {base_data_filename} after depth filtering. Skipping for KDE.")
                        removed_file_names.append(base_data_filename)
                        continue
                elif local_apply_depth_filter and selected_max_depth is None:
                    print(f"  Depth filtering was ON, but no Max Depth was selected for {base_data_filename}. This file will be SKIPPED for KDE.")
                    continue
    
                df[error_col_name] = df[SDB_col] - df[ref_col]
                error_data_raw = df[error_col_name].dropna()
    
                error_data_filtered = error_data_raw[
                    (error_data_raw > kde_error_filter_min) & (error_data_raw < kde_error_filter_max)
                ]
    
                lines_removed_kde_filter = len(error_data_raw) - len(error_data_filtered)
                if lines_removed_kde_filter > 0:
                    print(f"  Removed {lines_removed_kde_filter} points outside [{kde_error_filter_min}, {kde_error_filter_max}] for KDE in {base_data_filename}.")
                total_lines_removed_by_filter += lines_removed_kde_filter
    
                if error_data_filtered.empty or len(error_data_filtered) < 2:
                    print(f"  Not enough error data points ({len(error_data_filtered)}) for KDE in {base_data_filename} after all filtering. Skipping.")
                    if base_data_filename not in removed_file_names:
                        removed_file_names.append(base_data_filename)
                    continue
    
                all_error_data_for_overall_range.extend(error_data_filtered.values)
                datasets_for_kde.append({'label': data_filename_no_ext, 'data': error_data_filtered, 'N': len(error_data_filtered)})
                print(f"  Added {len(error_data_filtered)} error values from '{base_data_filename}' to KDE dataset.")
    
            except Exception as e:
                print(f"  An error occurred while processing main data CSV {data_csv_path}: {e}")
                if base_data_filename not in removed_file_names:
                    removed_file_names.append(base_data_filename)
    
        # --- Plotting section (original content, now inside the outer loop) ---
        if not datasets_for_kde:
            print(f"\nNo datasets available for KDE plotting for SDB_type '{current_sdb_type}'. Skipping plot generation.")
            # Print final stats for this SDB_type even if no plot is generated
            print(f"\n--- Final Stats for SDB_type: {current_sdb_type} ---")
            print(f"Total original lines across input files: {total_original_lines}")
            print(f"Total lines removed by filters: {total_lines_removed_by_filter}")
            if removed_file_names:
                unique_removed_files = sorted(list(set(removed_file_names)))
                print("Files from which all or most lines were removed (or were skipped entirely):")
                for fname in unique_removed_files:
                    print(f"- {fname}")
            else:
                print("No files had all or most of their lines removed or were skipped based on the specified criteria.")
            continue # Skip to the next SDB_type
    
        fig, ax = plt.subplots(figsize=(12, 7))
        common_bin_width_for_scaling = fixed_bin_width
        print(f"\nGenerating combined KDE plot for SDB_type '{current_sdb_type}'. Using fixed common bin_width for scaling: {common_bin_width_for_scaling:.4f}")
    
        # Customized Tick Marks and Labels
        ax.set_xticks([-10, -5, 0, 5, 10])
        y_min_plot, y_max_plot = plot_ylim
        #y_ticks = np.arange(int(y_min_plot / 5) * 5, y_max_plot + 5, 5)
        #ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', which='major', labelsize=15)
    
        max_scaled_kde_y = 0
        all_scaled_kde_y_arrays = []
        common_kde_x = None
        plotted_lines_info = []
        interactive_kde_lines = []
        HIGHLIGHT_COLOR = 'blue'
        HIGHLIGHT_LINEWIDTH = 3.0
        HIGHLIGHT_ALPHA = 1.0
        HIGHLIGHT_ZORDER = 10
        linestyle_map = {'green': '--', 'red': '-.', 'merged': ':'}
        special_line_color = 'black'
        legend_handles = []
    
        # --- Counting files for legend ---
        red_files_count = 0
        green_files_count = 0
        merged_files_count = 0
        other_files_count = 0 # This might be less relevant now since we filter by current_sdb_type earlier
    
        for dataset in datasets_for_kde:
            label = dataset['label'].lower()
            if 'red' in label:
                red_files_count += 1
            elif 'green' in label:
                green_files_count += 1
            elif 'merged' in label:
                merged_files_count += 1
            else:
                other_files_count += 1
    
        # Only add to legend if files exist for the current SDB_type
        if current_sdb_type == 'SDB': # All files (red, green, merged)
            for keyword, style in linestyle_map.items():
                count = 0
                if keyword == 'red': count = red_files_count
                elif keyword == 'green': count = green_files_count
                elif keyword == 'merged': count = merged_files_count
                if count > 0:
                    label_text = f"{keyword.capitalize()} Files (N={count})"
                    legend_handles.append(
                        mlines.Line2D([0], [0], color=special_line_color, lw=individual_kde_linewidth, linestyle=style, label=label_text)
                    )
            if other_files_count > 0:
                legend_handles.append(mlines.Line2D([0], [0], color=individual_kde_color, lw=individual_kde_linewidth, linestyle=individual_kde_linestyle, label=f'Other Files (N={other_files_count})'))
    
        elif current_sdb_type == 'SDB_red' and red_files_count > 0:
            legend_handles.append(mlines.Line2D([0], [0], color=special_line_color, lw=individual_kde_linewidth, linestyle=linestyle_map['red'], label=f'Red Files (N={red_files_count})'))
        elif current_sdb_type == 'SDB_green' and green_files_count > 0:
            legend_handles.append(mlines.Line2D([0], [0], color=special_line_color, lw=individual_kde_linewidth, linestyle=linestyle_map['green'], label=f'Green Files (N={green_files_count})'))
        elif current_sdb_type == 'SDB_merged' and merged_files_count > 0:
            legend_handles.append(mlines.Line2D([0], [0], color=special_line_color, lw=individual_kde_linewidth, linestyle=linestyle_map['merged'], label=f'Merged Files (N={merged_files_count})'))
    
    
        for i, dataset_info in enumerate(datasets_for_kde):
            label = dataset_info['label']
            error_values = dataset_info['data']
            N = dataset_info['N']
            try:
                if len(error_values) < 2 : continue
                bw_factor = 0.15
                kde = gaussian_kde(error_values, bw_method=bw_factor)
                if common_kde_x is None:
                    if all_error_data_for_overall_range:
                        pooled_min = np.min(all_error_data_for_overall_range); pooled_max = np.max(all_error_data_for_overall_range)
                        x_padding = (pooled_max - pooled_min) * 0.1 if (pooled_max - pooled_min) > 0 else 1.0
                        x_min_kde_calc = pooled_min - x_padding; x_max_kde_calc = pooled_max + x_padding
                        x_min_kde = plot_xlim[0] if plot_xlim and plot_xlim[0] <= x_min_kde_calc else x_min_kde_calc
                        x_max_kde = plot_xlim[1] if plot_xlim and plot_xlim[1] >= x_max_kde_calc else x_max_kde_calc
                    else:
                        x_min_kde = plot_xlim[0] if plot_xlim else kde_error_filter_min
                        x_max_kde = plot_xlim[1] if plot_xlim else kde_error_filter_max
                    common_kde_x = np.linspace(x_min_kde, x_max_kde, 400)
    
                kde_y_density = kde(common_kde_x)
                scaled_kde_y = kde_y_density * N * common_bin_width_for_scaling
                all_scaled_kde_y_arrays.append(scaled_kde_y)
                if scaled_kde_y.size > 0:
                    current_max_y = np.max(scaled_kde_y)
                    if current_max_y > max_scaled_kde_y: max_scaled_kde_y = current_max_y
    
                line_style_to_use = individual_kde_linestyle; line_color_to_use = individual_kde_color
                for keyword, style in linestyle_map.items():
                    if keyword in label.lower():
                        line_style_to_use = style; line_color_to_use = special_line_color; break
    
                line, = ax.plot(common_kde_x, scaled_kde_y, color=line_color_to_use, linestyle=line_style_to_use,
                                linewidth=individual_kde_linewidth, alpha=0.7, picker=True, pickradius=5)
                plotted_lines_info.append({'line': line, 'label': label.replace("_", " ")})
                interactive_kde_lines.append({'line': line, 'original_color': line_color_to_use,
                                                'original_linestyle': line_style_to_use,
                                                'original_linewidth': individual_kde_linewidth,
                                                'original_alpha': 0.7, 'original_zorder': line.get_zorder()})
            except np.linalg.LinAlgError as lae:
                print(f"  LinAlgError for {label} (likely singular matrix in KDE, too few unique points or all points identical): {lae}. Skipping KDE.")
            except ValueError as ve:
                print(f"  ValueError for {label} during KDE (e.g. empty error_values): {ve}. Skipping KDE.")
            except Exception as e_kde:
                print(f"  Warning: Could not compute or plot KDE for {label}. Error: {e_kde}")
    
        average_kde_stats_text = "Summary Stats:\nN/A (No KDEs plotted or common x-axis issue)"
        average_line_handle = None
    
        if all_scaled_kde_y_arrays and common_kde_x is not None and len(all_scaled_kde_y_arrays) > 0:
            valid_arrays = [arr for arr in all_scaled_kde_y_arrays if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) == len(common_kde_x)]
            if valid_arrays and len(valid_arrays) > 0:
                try:
                    stacked_kdes = np.array(valid_arrays)
                    if stacked_kdes.ndim == 2 and stacked_kdes.shape[0] > 0:
                        average_kde_y = np.mean(stacked_kdes, axis=0)
                        if average_kde_y.size > 0:
                            current_max_avg_y = np.max(average_kde_y)
                            if current_max_avg_y > max_scaled_kde_y: max_scaled_kde_y = current_max_avg_y
                            avg_line_plots = ax.plot(common_kde_x, average_kde_y, color='red', linestyle='-', linewidth=2.5,
                                                     label=f'Average KDE (of {len(valid_arrays)} datasets)',
                                                     zorder=len(interactive_kde_lines) + 10)
                            if avg_line_plots:
                                average_line_handle = avg_line_plots[0]
                                if average_line_handle not in legend_handles : legend_handles.append(average_line_handle)
                                plotted_lines_info.append({'line': average_line_handle, 'label': average_line_handle.get_label()})
                            print("Plotted Average KDE line.")
                        else: print("Could not compute average KDE: Stacked array is not 2D or is empty.")
                except ValueError as ve:
                    print(f"Could not stack KDEs for averaging: {ve}. Check array shapes.")
            else: print("No valid KDE arrays to average.")
    
        if all_error_data_for_overall_range:
            pooled_errors = np.array(all_error_data_for_overall_range)
            avg_kde_mean, avg_kde_min, avg_kde_max, avg_kde_std = np.mean(pooled_errors), np.min(pooled_errors), np.max(pooled_errors), np.std(pooled_errors)
            avg_kde_kurtosis, avg_kde_skewness = kurtosis(pooled_errors, fisher=True), skew(pooled_errors)
            average_kde_stats_text = (f"Summary Stats:\n\nMean = {avg_kde_mean:.3f} m\nMin = {avg_kde_min:.2f} m\nMax = {avg_kde_max:.2f} m\n"
                                      f"Std Dev = {avg_kde_std:.2f} m\nSkewness = {avg_kde_skewness:.2f}\nKurtosis = {avg_kde_kurtosis:.2f}\nTotal Pts = {len(pooled_errors)}")
        else:
            average_kde_stats_text = "Summary Stats:\n(No data pooled)"
    
        ax.set_xlabel(error_col_name + " (m)", fontsize=15)
        ax.set_ylabel("Count", fontsize=15)
        title_suffix = " (Depth Filtered by R² Criteria)" if local_apply_depth_filter and datasets_for_kde else ""
        ax.set_title(f"Distributions of {Sensor} {current_sdb_type} Error: {AOI}", fontsize=25, fontweight='bold', y=1.05)
    
        ax.set_xlim(plot_xlim)
        if plot_ylim: ax.set_ylim(plot_ylim)
    
        if datasets_for_kde:
            current_plot_xlim_text = ax.get_xlim(); text_x_pos = current_plot_xlim_text[0] + (current_plot_xlim_text[1] - current_plot_xlim_text[0]) * 0.02
            current_ax_ylim_text = ax.get_ylim(); text_y_pos = current_ax_ylim_text[1] * 0.97
            ax.text(text_x_pos, text_y_pos, average_kde_stats_text, fontsize=18, color='black', ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.75))
    
        # --- Interactive Plot Functions (Re-integrated) ---
        def on_pick_kde_plot(event):
            picked_line_artist = event.artist
            is_interactive_target = any(item['line'] == picked_line_artist for item in interactive_kde_lines)
            if not is_interactive_target: return
            for item in interactive_kde_lines:
                line_obj = item['line']
                if line_obj == picked_line_artist:
                    line_obj.set_color(HIGHLIGHT_COLOR); line_obj.set_linestyle('-'); line_obj.set_linewidth(HIGHLIGHT_LINEWIDTH); line_obj.set_alpha(HIGHLIGHT_ALPHA); line_obj.set_zorder(HIGHLIGHT_ZORDER)
                else:
                    line_obj.set_color(item['original_color']); line_obj.set_linestyle(item['original_linestyle']); line_obj.set_linewidth(item['original_linewidth']); line_obj.set_alpha(item['original_alpha']); line_obj.set_zorder(item['original_zorder'])
            fig.canvas.draw_idle()
        fig.canvas.mpl_connect('pick_event', on_pick_kde_plot)
    
        def on_scroll_kde_plot(event):
            if event.inaxes != ax: return
            base_scale=1.1; cur_xlim,cur_ylim=ax.get_xlim(),ax.get_ylim(); xdata,ydata=event.xdata,event.ydata
            if xdata is None or ydata is None: xdata,ydata=(cur_xlim[0]+cur_xlim[1])/2,(cur_ylim[0]+cur_ylim[1])/2
            scale_factor=1/base_scale if event.button=='up' else base_scale if event.button=='down' else 1
            if scale_factor==1: return
            # The following lines related to new_width/height and relx/rely for dynamic zooming are commented out
            # as they interact with manual_xlim/ylim and might need careful adjustment based on desired behavior.
            # Keeping ax.set_xlim(plot_xlim) and ax.set_ylim(plot_ylim) for now.
            # new_width=(cur_xlim[1]-cur_xlim[0])*scale_factor;
            # relx=((xdata-cur_xlim[0])/(cur_xlim[1]-cur_xlim[0])) if (cur_xlim[1]-cur_xlim[0])!=0 else 0.5;
            ax.set_xlim(plot_xlim) # Re-applies fixed xlim
            # new_height=(cur_ylim[1]-cur_ylim[0])*scale_factor;
            # rely=((ydata-cur_ylim[0])/(cur_ylim[1]-cur_ylim[0])) if (cur_ylim[1]-cur_ylim[0])!=0 else 0.5;
            ax.set_ylim(plot_ylim) # Re-applies fixed ylim
            ax.figure.canvas.draw_idle()
        fig.canvas.mpl_connect('scroll_event', on_scroll_kde_plot)
    
        line_to_label_map_kde = {item['line']: item['label'] for item in plotted_lines_info if item['line'] is not None}
        lines_for_cursor_kde = [item['line'] for item in plotted_lines_info if item['line'] is not None]
        if lines_for_cursor_kde:
            cursor = mplcursors.cursor(lines_for_cursor_kde, hover=False)
            @cursor.connect("add")
            def on_add_cursor_kde(sel):
                label_for_line = line_to_label_map_kde.get(sel.artist, "Unknown")
                text = (f"{label_for_line}\nErr:{sel.target[0]:.2f}\nAvg.Dens:{sel.target[1]:.2f}" if average_line_handle and sel.artist == average_line_handle else f"File:{label_for_line}\nErr:{sel.target[0]:.2f}\nDens:{sel.target[1]:.2f}")
                sel.annotation.set_text(text); sel.annotation.get_bbox_patch().set(alpha=0.85, facecolor='lightyellow')
        # --- End Interactive Plot Functions ---
    
        if legend_handles: ax.legend(handles=legend_handles, fontsize=15, loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
    
        # Define output filename based on the current SDB_type
        combined_plot_filename = f"{Sensor}_Combined_Error_KDE_{AOI}_{current_sdb_type}_DepthFiltered.png"
        output_plot_full_path = os.path.join(output_plot_folder_path, combined_plot_filename)
    
        # --- NEW: Conditional Save and Show ---
        if save_plots:
            try:
                plt.savefig(output_plot_full_path, dpi=300)
                print(f"\nSUCCESS: Combined KDE plot saved to: {output_plot_full_path}")
            except Exception as e_save:
                print(f"\nERROR: Failed to save combined KDE plot to {output_plot_full_path}. Error: {e_save}")
    
        if show_plots:
            plt.show()
    
        # Close the figure to free memory, crucial for loops
        plt.close(fig) 
    
        # Print final stats for the current SDB_type category
        print(f"\n--- Final Stats for SDB_type: {current_sdb_type} ---")
        print(f"Total original lines across input files: {total_original_lines}")
        print(f"Total lines removed by filters: {total_lines_removed_by_filter}")
        if removed_file_names:
            unique_removed_files = sorted(list(set(removed_file_names)))
            print("Files from which all or most lines were removed (or were skipped entirely):")
            for fname in unique_removed_files:
                print(f"- {fname}")
        else:
            print("No files had all or most of their lines removed or were skipped based on the specified criteria.")
    
    print("\n--- All SDB_type categories processed. ---")