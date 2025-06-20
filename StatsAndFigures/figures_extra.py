# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:47:25 2025

@author: mccullru
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import mplcursors
import matplotlib.lines as mlines

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde, skew, kurtosis
from difflib import get_close_matches


##############################################################################################################
##############################################################################################################

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

# --- Configuration ---
input_csv_folder_path = r"E:\Thesis Stuff\SDB_ExtractedPts"

AOI = 'Anegada'
Sensor = 'SD'

# Define the SDB_types you want to loop through and generate plots for
SDB_TYPES_TO_PROCESS = ['SDB', 'SDB_red', 'SDB_green', 'SDB_merged']

output_plot_folder_path = r"B:\Thesis Project\StatsAndFigures\Trial Run Figures\Homer\KDE plots"

# --- CONFIGURATION FOR DEPTH FILTERING FROM STATS CSVS ---
stats_csv_folder_path = r"E:\Thesis Stuff\SDB_ExtractedPts_maxR2_results"
apply_depth_filter = True
r2_threshold_for_selection = 0.7

# --- Column Names for SDB_ExtractedPts files (main data) ---
ref_col = "Geoid_Corrected_Ortho_Height"
SDB_col = "Raster_Value"
error_col_name = "Error"

# Plot Settings
plot_xlim = (-10, 10)
plot_ylim = (0, 700)
individual_kde_color = 'gray'
individual_kde_linestyle = '-'
individual_kde_linewidth = 1.5
fixed_bin_width = 0.1

# Define bounds for error filtering for KDE plot
kde_error_filter_min = -10
kde_error_filter_max = 10

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

    try:
        plt.savefig(output_plot_full_path, dpi=300)
        print(f"\nSUCCESS: Combined KDE plot saved to: {output_plot_full_path}")
    except Exception as e_save:
        print(f"\nERROR: Failed to save combined KDE plot to {output_plot_full_path}. Error: {e_save}")

    plt.show()
    plt.close(fig) # Close the figure to free memory after showing/saving

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


##############################################################################################################
##############################################################################################################

"""Histogram scatter heat map"""

# --- DATA SOURCE ---
csv_folder_path = r"E:\Thesis Stuff\SDB_ExtractedPts" 
x_column_name = "Raster_Value"
y_column_name = "Geoid_Corrected_Ortho_Height"

# --- PLOT APPEARANCE ---
manual_xlim = (0, 10)
manual_ylim = (0, 10)

# --- SAVE OPTIONS  ---
# Set the folder where you want to save the plots.
output_folder_path = r"B:\Thesis Project\StatsAndFigures\Trial Run Figures\Marathon\SD_Heatplots"
save_plots = False  # Set to True to save the plots, False to disable saving.
show_plots = True  # Set to True to display plots on screen, False to hide them.



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
        filename_no_ext = os.path.splitext(base_filename)[0]

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

##############################################################################################################
##############################################################################################################

""" Individual Error Histograms """



# Specify the folder containing your input CSV files
input_csv_folder_path = r"E:\Thesis Stuff\SDB_ExtractedPts"

# Specify the folder where you want to save the output plots
output_plot_folder_path = r"B:\Thesis Project\SDB_Time\Results_main\Marathon\SuperDove\Final5\Extracted Pts\New folder\test_output" 

# Define column names
ref_col = "Geoid_Corrected_Ortho_Height"
SDB_col = "Raster_Value"
error_col_name = "Error" 

#  Define a FIXED BIN WIDTH ---
fixed_bin_width = 0.1  # Adjust this value based on your data's typical error range and desired resolution

# --- NEW: Define bounds for error filtering for KDE and Histogram ---
error_filter_min = -10
error_filter_max = 10


# Histogram plot settings
hist_xlim = (-10, 10)
hist_ylim = (0, 700) 

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
        else:
            # Dynamic y-limit if hist_ylim is not set or is (0,0) which can happen
            max_y_val = 0
            if len(counts) > 0:
                max_y_val = max(counts)
            if 'scaled_kde_y' in locals() and len(scaled_kde_y) > 0 : # Check if KDE was plotted
                  max_y_val = max(max_y_val, np.max(scaled_kde_y))
            ax.set_ylim(0, max_y_val * 1.1 if max_y_val > 0 else 10)


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
        
        #plt.savefig(output_plot_full_path) # Uncommented this
        #print(f"Saved plot to: {output_plot_full_path}")

        plt.show() 
        #plt.close(fig)

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


##############################################################################################################
##############################################################################################################

"""Linear Regressions"""

# def process_single_csv(input_file, output_folder):
#     """
#     Processes a single CSV file, performs linear regression, 
#     and saves the results in the output folder.
#     """
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # Check if the input file exists
#     if not os.path.exists(input_file):
#         print(f"Error: {input_file} does not exist.")
#         return

#     print(f"Processing {input_file}...")

#     # Read the CSV file
#     data = pd.read_csv(input_file)

#     # Drop rows where 'Raster_Value' is blank
#     data = data.dropna(subset=['Raster_Value'])

#     # Perform linear regression
#     x = data[['Raster_Value']].values
#     y = data['Geoid_Corrected_Ortho_Height'].values
#     model = LinearRegression()
#     model.fit(x, y)

#     # Calculate regression metrics
#     y_pred = model.predict(x)
#     r2 = r2_score(y, y_pred)  # Scikit-learn R² calculation
#     rmse = np.sqrt(mean_squared_error(y, y_pred))

#     # Create the line of best fit equation
#     coef = model.coef_[0]
#     intercept = model.intercept_
#     equation = f"y = {coef:.4f}x + {intercept:.4f}"

#     # Calculate perpendicular distances
#     distances = np.abs(coef * x.flatten() - y + intercept) / np.sqrt(coef**2 + 1)

#     # Compute statistics for distances
#     min_dist = np.min(distances)
#     max_dist = np.max(distances)
#     mean_dist = np.mean(distances)
#     std_dist = np.std(distances)

#     # Append the results to the list
#     results = {
#         "Image Name": os.path.basename(input_file),
#         "R^2": r2,
#         "RMSE": rmse,
#         "Line of Best Fit": equation,
#         "m1": coef,
#         "m0": intercept,
#         "min perp dist": min_dist,
#         "max perp dist": max_dist,
#         "mean perp dist": mean_dist,
#         "std dev perp dist": std_dist
#     }

#     # Regression Plot
#     plt.figure(figsize=(8, 6))
#     plt.scatter(x, y, color='blue', alpha=0.7)
#     plt.plot(x, y_pred, color='red', linewidth=2)
    
#     plt.title('Sentinel-2 Linear Regression', fontsize=30)
    
#     plt.xlabel("pSDB values", fontsize=24)
#     plt.ylabel("Reference Depth (m)", fontsize=24)
#     plt.legend()
#     plt.grid(True)
    
#     #plt.xlim(.9, None)
#     #plt.ylim(None, 0.9)

#     # Add R^2 and RMSE as text on the plot
#     min_x = np.min(x)
#     max_y = np.max(y)
#     min_y = np.min(y)
#     mean_y = np.mean(y)
#     plt.text(min_x, 10, f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}", fontsize=18, color='black', ha='left',
#               bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

#     # Invert both axes so 0 is bottom left, and up and right are negative
#     #plt.gca().invert_xaxis()
#     #plt.gca().invert_yaxis()

#     # Save the regression plot in the output folder
#     plot_filename = f"{os.path.splitext(os.path.basename(input_file))[0]}_LR_plot_better.png"
#     plot_path = os.path.join(output_folder, plot_filename)
#     plt.savefig(plot_path)
#     plt.close()



# input_file = r"E:\Thesis Stuff\pSDB_ExtractedPts\SD_PlanetScope_2274_2022_06_03_21_10_58_L2W__RGB_pSDBred_extracted.csv"
# output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts"
# process_single_csv(input_file, output_folder)



##############################################################################################################
##############################################################################################################


""" Single Histogram chart for viewing multiple SuperDove and Sentinel-2 error values for a single site """


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd # For reading CSV files
# import os # For path manipulation
# import glob # For finding files

# # --- 1. Configuration ---
# # Specify the folder containing your CSV files
# csv_folder_path = r"B:\Thesis Project\SDB_Time\Results_main\Marathon\SuperDove\Final5\Extracted Pts\New folder"  # <--- !!! REPLACE THIS !!!

# # Specify the column name in your CSV files that contains the error values
# error_column_name = "Error"  # <--- !!! REPLACE THIS if your column name is different !!!

# # Optional: Define colors and linestyles if you have a fixed number of expected files
# # or want to cycle through them.
# colors = ['blue', 'darkorange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (5, 1)), (0, (1, 1))]

# # --- 2. Load Data from CSV Files ---
# all_error_data = [] # To store data from all files for common bin calculation
# data_to_plot = []   # To store tuples of (label, error_values_array)

# csv_files = glob.glob(os.path.join(csv_folder_path, "*.csv"))

# if not csv_files:
#     print(f"No CSV files found in the folder: {csv_folder_path}")
#     # You might want to exit or raise an error here if no files are found
#     # For now, it will just produce an empty plot.
# else:
#     print(f"Found {len(csv_files)} CSV files to process in: {csv_folder_path}")

# for i, file_path in enumerate(csv_files):
#     try:
#         df = pd.read_csv(file_path)
#         if error_column_name in df.columns:
#             # Remove NaNs and convert to numpy array
#             error_values = df[error_column_name].dropna().values.astype(float)
#             if error_values.size > 0:
#                 all_error_data.append(error_values)
#                 # Create a label from the filename (without extension)
#                 file_name = os.path.basename(file_path)
#                 label = os.path.splitext(file_name)[0]
#                 data_to_plot.append((label, error_values))
#                 print(f"Loaded {len(error_values)} error values from '{file_name}' (column: '{error_column_name}').")
#             else:
#                 print(f"Warning: No valid error values found in column '{error_column_name}' in file '{file_path}'.")
#         else:
#             print(f"Warning: Column '{error_column_name}' not found in file '{file_path}'. Skipping this file.")
#     except Exception as e:
#         print(f"Error processing file '{file_path}': {e}")

# # --- 3. Plot Histograms as Outlines (Steps) ---
# if data_to_plot: # Only proceed if there's data to plot
#     plt.figure(figsize=(12, 7)) # Adjust figure size if needed

#     # Determine common bins based on all loaded data
#     if all_error_data:
#         concatenated_errors = np.concatenate(all_error_data)
#         min_bin = 0.0 # RMSE/errors are typically non-negative
#         # Consider a robust max_bin, e.g., percentile, or just max if outliers are okay
#         max_bin = np.percentile(concatenated_errors, 99.5) if concatenated_errors.size > 0 else 1.0
#         if max_bin <= min_bin: # handle cases with very little data or constant values
#             max_bin = min_bin + 1.0
#         num_bins = 30 # Adjust number of bins as desired
#         bins = np.linspace(min_bin, max_bin, num_bins + 1)
#     else: # Fallback if no data was loaded at all (shouldn't happen if data_to_plot is not empty)
#         bins = np.linspace(0, 1, num_bins + 1)


#     # Plot histograms using histtype='step'
#     for i, (label, error_values) in enumerate(data_to_plot):
#         plt.hist(error_values, bins=bins,
#                   #density=True,          # Use density for fair comparison if sample sizes vary
#                   histtype='step',       # Draw outline
#                   linewidth=1.5,         # Line thickness
#                   linestyle=linestyles[i % len(linestyles)], # Cycle through linestyles
#                   color=colors[i % len(colors)],             # Cycle through colors
#                   label=f'{label} (N={len(error_values)})')

#     # Add plot labels and title
#     plt.xlabel(f"{error_column_name} Values", fontsize=12) # Use the actual column name
#     plt.ylabel("Point Count", fontsize=12)
#     plt.title(f"Distribution of {error_column_name} Values (Outlines)", fontsize=14)
#     plt.legend(fontsize=10)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout() # Adjust layout
#     plt.show()
# else:
#     print("No data loaded to plot.")



