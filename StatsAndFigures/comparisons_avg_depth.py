# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:04:25 2025

@author: mccullru
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from difflib import get_close_matches


def _collect_max_depth_data(input_folder: str, stats_folder: str, config: dict, sensor_name: str, sdb_type: str):
    """
    Helper function to collect max_depth_range values and identify contributing source files.

    MODIFIED: Now also returns a list of source data files that passed the R2 check.
    """
    print(f"\n--- Collecting max depth data for {sensor_name} {sdb_type.capitalize()} ---")

    r2_threshold_for_selection = config.get('r2_threshold', 0.7)
    stats_indicator_col = config.get('stats_indicator_col', 'Indicator')
    stats_r2_col = config.get('stats_r2_col', 'R2 Value')
    max_depth_range_col = 'Max Depth Range'

    collected_depths = []
    
    # Initialize the list at the beginning of the function.
    contributing_files = []
    processed_files_count = 0
    skipped_files_count = 0

    if not os.path.isdir(stats_folder):
        print(f"  Warning: Stats folder for {sensor_name} not found: {stats_folder}. Skipping data collection for this sensor/type.")
        # Ensure we always return 3 values, even on an early exit.
        return [], 0, []

    for filename in os.listdir(stats_folder):
        file_lower = filename.lower()
        if not (filename.endswith(".csv") and "_extracted_lr_stats_iterations" in file_lower and \
                sdb_type in file_lower and "merged" not in file_lower):
            continue

        filepath = os.path.join(stats_folder, filename)

        try:
            stats_df = pd.read_csv(filepath)
            if stats_df.empty:
                skipped_files_count += 1
                continue

            row_to_use = None
            required_cols = [stats_indicator_col, stats_r2_col, max_depth_range_col]
            if not all(col in stats_df.columns for col in required_cols):
                skipped_files_count += 1
                continue

            # (Selection logic for row_to_use remains the same...)
            rows_ind2 = stats_df[stats_df[stats_indicator_col] == 2]
            if not rows_ind2.empty and round(rows_ind2.iloc[0][stats_r2_col], 2) >= r2_threshold_for_selection:
                row_to_use = rows_ind2.iloc[0]

            if row_to_use is None:
                rows_ind1 = stats_df[stats_df[stats_indicator_col] == 1]
                if not rows_ind1.empty and round(rows_ind1.iloc[0][stats_r2_col], 2) >= r2_threshold_for_selection:
                    row_to_use = rows_ind1.iloc[0]

            if row_to_use is None:
                skipped_files_count += 1
                continue

            max_depth = row_to_use[max_depth_range_col]
            if pd.isna(max_depth):
                skipped_files_count += 1
                continue

            # If we get here, the file is valid.
            collected_depths.append(max_depth)
            processed_files_count += 1

            # This block constructs the source data file path and adds it to the list.
            base_data_filename = filename.replace('_LR_Stats_iterations.csv', '.csv')
            data_filepath = os.path.join(input_folder, base_data_filename)
            if os.path.exists(data_filepath):
                contributing_files.append(data_filepath)
            else:
                print(f"  Warning: Could not find corresponding source data file at {data_filepath}")

        except Exception as e:
            print(f"  An error occurred processing {filename}: {e}.")
            skipped_files_count += 1
            continue

    print(f"  {sensor_name} {sdb_type.capitalize()}: Processed {processed_files_count} files, found {len(contributing_files)} source files.")
    # Return all 3 values as expected by the main function.
    return collected_depths, processed_files_count, contributing_files


def generate_comparison_max_depth_bar_chart(
    sentinel2_input_folder: str,
    sentinel2_stats_folder: str,
    superdove_input_folder: str,
    superdove_stats_folder: str,
    output_folder: str,
    aoi: str,
    sensor_combined_name: str,
    config: dict,
    exclusion_path: str = None
):
    
    """
    Generates a bar chart comparing average max depth range for SDBred and SDBgreen
    between Sentinel-2 and SuperDove.

    MODIFIED: Now includes a horizontal line indicating the maximum reference depth
    found across all contributing source data files.
    """
    
    print(f"\n--- Generating Comparison Max Depth Bar Chart for {sensor_combined_name}: {aoi} ---")
    os.makedirs(output_folder, exist_ok=True)
    ref_col = "Geoid_Corrected_Ortho_Height"

    # --- 1. Collect Data and Contributing File Paths ---
    # This part calls the helper function and gets the lists of files that passed the checks.
    sd_red_depths, _, sd_red_files = _collect_max_depth_data(superdove_input_folder, superdove_stats_folder, config, "SuperDove", "red")
    s2_red_depths, _, s2_red_files = _collect_max_depth_data(sentinel2_input_folder, sentinel2_stats_folder, config, "Sentinel-2", "red")
    sd_green_depths, _, sd_green_files = _collect_max_depth_data(superdove_input_folder, superdove_stats_folder, config, "SuperDove", "green")
    s2_green_depths, _, s2_green_files = _collect_max_depth_data(sentinel2_input_folder, sentinel2_stats_folder, config, "Sentinel-2", "green")


    # --- 2. Calculate Overall Max Reference Depth  ---
    # This is the new section that defines the variable before it is used.
    all_contributing_files = set(sd_red_files + s2_red_files + sd_green_files + s2_green_files)
    overall_max_ref_depth = -np.inf # Initialize the variable

    if all_contributing_files:
        print(f"\n--- Calculating Max Reference Depth from {len(all_contributing_files)} unique source files ---")
        for data_filepath in all_contributing_files:
            try:
                df = pd.read_csv(data_filepath)
                if ref_col in df.columns:
                    local_max = pd.to_numeric(df[ref_col], errors='coerce').dropna().max()
                    if pd.notna(local_max) and local_max > overall_max_ref_depth:
                        overall_max_ref_depth = local_max # The variable is updated here
            except Exception as e:
                print(f"  Warning: Could not process source file {os.path.basename(data_filepath)}: {e}")

    # Final check to ensure the variable is valid before plotting
    if np.isinf(overall_max_ref_depth):
        overall_max_ref_depth = None # It will be None if no files were found
        print("\nCould not determine an overall maximum reference depth.")
    else:
        print(f"Overall Maximum Reference Depth found: {overall_max_ref_depth:.2f}m")


    # --- 3. Prepare Data for Plotting ---
    # This section prepares the data for the bars themselves.
    plot_data = []
    def add_bar_data_or_placeholder(label, display_label, depths, count, color):
        if depths:
            plot_data.append({'label': label, 'display_label': display_label, 'average': np.mean(depths), 'std': np.std(depths), 'count': count, 'color': color, 'is_pseudo': False})
        else:
            plot_data.append({'label': label, 'display_label': display_label, 'average': 0.0, 'std': 0.0, 'count': 0, 'color': 'white', 'edgecolor': 'lightgray', 'is_pseudo': True})

    add_bar_data_or_placeholder('SuperDove Red', 'SD', sd_red_depths, len(sd_red_depths), 'red')
    add_bar_data_or_placeholder('Sentinel-2 Red', 'S2', s2_red_depths, len(s2_red_depths), 'blue')
    add_bar_data_or_placeholder('SuperDove Green', 'SD', sd_green_depths, len(sd_green_depths), 'red')
    add_bar_data_or_placeholder('Sentinel-2 Green', 'S2', s2_green_depths, len(s2_green_depths), 'blue')

    ordered_plot_data = []
    label_order_keys = ['SuperDove Red', 'Sentinel-2 Red', 'SuperDove Green', 'Sentinel-2 Green']
    for l_key in label_order_keys:
        for item in plot_data:
            if item['label'] == l_key:
                ordered_plot_data.append(item)
                break

    # --- 4. Generate Plot ---
    # This section USES the `overall_max_ref_depth` variable defined in Step 2.
    fig, ax = plt.subplots(figsize=(10, 8))
    actual_x_positions_for_plotting = [0.5, 0.9, 1.85, 2.25]
    bar_width = 0.35
    bars_patches = []

    for i, item in enumerate(ordered_plot_data):
        bar = ax.bar(actual_x_positions_for_plotting[i], item['average'], yerr=item['std'], capsize=5,
                     color=item['color'], edgecolor=item.get('edgecolor', 'black'),
                     width=bar_width, linewidth=1.5 if item['is_pseudo'] else 1)
        bars_patches.extend(bar.patches)

    for i, bar_patch in enumerate(bars_patches):
        item = ordered_plot_data[i]
        if item['is_pseudo']:
            ax.text(bar_patch.get_x() + bar_patch.get_width()/2, 0.1, "N/A",
                    ha='center', va='bottom', fontsize=12, color='gray', fontstyle='italic')
        else:
            yval = bar_patch.get_height()
            error_offset = item['std']
            ax.text(bar_patch.get_x() + bar_patch.get_width()/2, yval + error_offset + 0.05,
                    f"n={item['count']}", ha='center', va='bottom', fontsize=15, fontweight='bold')

    max_height = 0
    for item in ordered_plot_data:
        if not item['is_pseudo']:
            current_height = item['average'] + item['std']
            if current_height > max_height:
                max_height = current_height

    # Add the horizontal line using the now-defined variable
    if overall_max_ref_depth is not None:
        ax.axhline(y=overall_max_ref_depth, color='gray', linestyle='--', linewidth=2.5,
                   label=f'Max Ref. Depth ({overall_max_ref_depth:.2f}m)')
        if overall_max_ref_depth > max_height:
            max_height = overall_max_ref_depth

    ax.set_ylabel('Average Max Depth Range (m)', fontsize=15)
    ax.set_title(f'Average {sensor_combined_name} Max Depth Range: {aoi}', fontsize=25, fontweight='bold', y=1.05)
    ax.set_xticks(actual_x_positions_for_plotting)
    ax.set_xticklabels([item['display_label'] for item in ordered_plot_data], fontsize=12)
    ax.set_ylim(0, max_height * 1.2 if max_height > 0 else 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if overall_max_ref_depth is not None:
        ax.legend(fontsize=12, loc='upper left')

    plt.subplots_adjust(bottom=0.2)

    red_group_center_x = 0.72
    green_group_center_x = 2.05
    red_group_center_x_display, _ = ax.transData.transform((red_group_center_x, 0))
    red_group_center_x_fig, _ = fig.transFigure.inverted().transform((red_group_center_x_display, 0))
    green_group_center_x_display, _ = ax.transData.transform((green_group_center_x, 0))
    green_group_center_x_fig, _ = fig.transFigure.inverted().transform((green_group_center_x_display, 0))
    y_text_pos_fig = 0.08

    fig.text(red_group_center_x_fig, y_text_pos_fig, "^------------SDB Red------------^",
             ha='center', va='top', fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    fig.text(green_group_center_x_fig, y_text_pos_fig, "^-----------SDB Green-----------^",
             ha='center', va='top', fontsize=14, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    # --- 5. Save and Show Plot ---
    plot_filename = f"{aoi}_Max_Depth_Comparison.png"
    plot_filepath = os.path.join(output_folder, plot_filename)
    if config.get('save_plots', False):
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"\nSUCCESS: Comparison bar chart saved to: {plot_filepath}")
    if config.get('show_plots', True):
        plt.show()
    else:
        plt.close(fig)

    print("--- Comparison Max Depth Bar Chart Generation Complete. ---")
    
    
    
    
    
    
