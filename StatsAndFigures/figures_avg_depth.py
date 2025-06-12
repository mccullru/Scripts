# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:09:12 2025

@author: mccullru
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_depth_range_bar_chart(
    input_folder: str,
    stats_folder: str,
    output_folder: str,
    aoi: str,
    sensor: str,
    config: dict,
    exclusion_path: str = None 
):
    """
    Reads SDB_extracted_pts_maxR2_results files, categorizes them by 'green' or 'red'
    in the filename, extracts the 'Max Depth Range' value from the specific row
    determined by a selection logic (Indicator 2, then Indicator 1, based on R2 threshold),
    calculates average max_depth_range for SDBgreen and SDBred, and generates a bar chart with error bars.
    Includes extensive print statements for debugging.

    Args:
        input_folder (str): Path to the folder containing primary data CSVs.
        stats_folder (str): Path to the folder containing statistics CSVs
                            (SDB_extracted_pts_maxR2_results).
        output_folder (str): Path to the folder where the bar chart will be saved.
        aoi (str): Area of Interest (for plot title/filename).
        sensor (str): Sensor type (for plot title/filename).
        config (dict): Configuration dictionary for this script.
                       Expected keys: 'save_plots' (bool), 'show_plots' (bool),
                       'r2_threshold' (float), 'stats_indicator_col' (str),
                       'stats_r2_col' (str).
        exclusion_path (str): Path to a CSV file containing datasets to exclude.
    """

    print(f"--- Generating Depth Range Bar Chart for AOI: {aoi}, Sensor: {sensor} ---")
    print(f"Stats folder being scanned: {stats_folder}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get configuration values for row selection
    r2_threshold_for_selection = config.get('r2_threshold', 0.7)
    
    # IMPORTANT: These column names MUST match your CSV headers exactly.
    stats_indicator_col = config.get('stats_indicator_col', 'Indicator')
    stats_r2_col = config.get('stats_r2_col', 'R2 Value')
    max_depth_range_col = 'Max Depth Range'

    sdb_green_depths = []
    sdb_red_depths = []
    processed_files_count = 0

    # Iterate through files in the stats folder
    for filename in os.listdir(stats_folder):
        file_lower = filename.lower()
        
        # Initial check for CSV, and presence of '_extracted_lr_stats_iterations' and 'green' or 'red', 
        # and absence of 'merged'.
        if filename.endswith(".csv") \
           and "_extracted_lr_stats_iterations" in file_lower \
           and ("green" in file_lower or "red" in file_lower) \
           and "merged" not in file_lower:

            filepath = os.path.join(stats_folder, filename)
            print(f"\n--- Attempting to process file: {filename} ---")
            processed_files_count += 1

            try:
                stats_df = pd.read_csv(filepath)
                print(f"  Successfully loaded {filename}. Shape: {stats_df.shape}")
                if stats_df.empty:
                    print(f"  Warning: {filename} is empty. Skipping.")
                    continue

                # Initialize row_to_use to None for each file
                row_to_use = None

                # Check if required columns exist in the current DataFrame
                required_cols = [stats_indicator_col, stats_r2_col, max_depth_range_col]
                missing_cols = [col for col in required_cols if col not in stats_df.columns]
                if missing_cols:
                    print(f"  Warning: Missing required columns in {filename}: {missing_cols}. Skipping.")
                    print(f"  Available columns in {filename}: {stats_df.columns.tolist()}")
                    continue


                # --- 1. Try to find a passing Indicator 2 model ---
                rows_ind2 = stats_df[stats_df[stats_indicator_col] == 2]
                if not rows_ind2.empty:
                    r2_ind2 = round(rows_ind2.iloc[0][stats_r2_col], 2)
                    print(f"  Found Indicator 2 row. R2: {r2_ind2}, Threshold: {r2_threshold_for_selection}")
                    if r2_ind2 >= r2_threshold_for_selection:
                        row_to_use = rows_ind2.iloc[0]
                        print(f"  Selected Indicator 2 row from {filename} (R2={r2_ind2}).")
                    else:
                        print(f"  Indicator 2 R2 ({r2_ind2}) is below threshold ({r2_threshold_for_selection}).")
                else:
                    print(f"  No Indicator 2 row found in {filename}.")


                # --- 2. If Indicator 2 failed or R2 was too low, FALLBACK to Indicator 1 ---
                if row_to_use is None:
                    rows_ind1 = stats_df[stats_df[stats_indicator_col] == 1]
                    if not rows_ind1.empty:
                        r2_ind1 = round(rows_ind1.iloc[0][stats_r2_col], 2)
                        print(f"  Found Indicator 1 row. R2: {r2_ind1}, Threshold: {r2_threshold_for_selection}")
                        if r2_ind1 >= r2_threshold_for_selection:
                            row_to_use = rows_ind1.iloc[0]
                            print(f"  Selected Indicator 1 row from {filename} (R2={r2_ind1}).")
                        else:
                            print(f"  Indicator 1 R2 ({r2_ind1}) is below threshold ({r2_threshold_for_selection}).")
                    else:
                        print(f"  No Indicator 1 row found in {filename}.")

                # --- 3. If neither passed, now we skip this file ---
                if row_to_use is None:
                    best_r2_row = stats_df.loc[stats_df[stats_r2_col].idxmax()] if not stats_df.empty else None
                    r2_val = f"{best_r2_row[stats_r2_col]:.4f}" if best_r2_row is not None else "N/A"
                    print(f"  Skipping {filename}: No valid Indicator 1 or 2 row found above R2 threshold ({r2_threshold_for_selection}). Best R2: {r2_val}.")
                    continue # Skip to the next file

                # If we reached here, row_to_use contains the selected row, extract Max Depth Range
                max_depth = row_to_use[max_depth_range_col]
                if pd.isna(max_depth):
                    print(f"  Warning: '{max_depth_range_col}' in selected row of {filename} is NaN. Skipping.")
                    continue

                # Determine if the extracted depth belongs to SDBgreen or SDBred based on the current file's name
                if "green" in file_lower:
                    sdb_green_depths.append(max_depth)
                    print(f"  Successfully added green depth ({max_depth:.2f}m) from {filename}.")
                elif "red" in file_lower:
                    sdb_red_depths.append(max_depth)
                    print(f"  Successfully added red depth ({max_depth:.2f}m) from {filename}.")
                else:
                    # This should ideally not be reached if the outer 'if' condition is correct
                    print(f"  Error: File {filename} passed initial filter but did not contain 'green' or 'red' for categorization. Skipping.")


            except pd.errors.EmptyDataError:
                print(f"  Error: {filename} is an empty CSV file. Skipping.")
                continue
            except Exception as e:
                print(f"  An unexpected error occurred while processing {filename}: {e}. Skipping this file.")
                continue
        else:
            # Debugging: Log files that are skipped by initial filter
            if "_extracted_lr_stats_iterations" not in file_lower:
                print(f"Skipping {filename}: Does not contain '_extracted_lr_stats_iterations' or is not a CSV.")
            elif not (("green" in file_lower or "red" in file_lower) and "merged" not in file_lower):
                print(f"Skipping {filename}: Does not match 'green'/'red' and non-'merged' naming convention after initial filter.")


    print(f"\n--- Finished scanning {processed_files_count} relevant files. ---")
    
    print(f"Total SDBgreen depths collected: {len(sdb_green_depths)} values")
    print(f"Total SDBred depths collected: {len(sdb_red_depths)} values")
    print(f"SDBgreen depths: {sdb_green_depths}")
    print(f"SDBred depths: {sdb_red_depths}")

    # Consolidated print of collected max depths
    print(f"\nAll collected SDBgreen max depths: {sdb_green_depths}\n")
    print(f"All collected SDBred max depths: {sdb_red_depths}\n")

    # Calculate averages and standard deviations
    avg_green_depth = np.mean(sdb_green_depths) if sdb_green_depths else 0
    std_green_depth = np.std(sdb_green_depths) if sdb_green_depths else 0

    avg_red_depth = np.mean(sdb_red_depths) if sdb_red_depths else 0
    std_red_depth = np.std(sdb_red_depths) if sdb_red_depths else 0

    # Generate and Save Summary CSV ---

    if sdb_red_depths or sdb_green_depths:
        # Create a DataFrame with the individual depth values first.
        # This correctly handles unequal lengths and sets the required number of rows.
        output_df = pd.DataFrame({
            'SDBred_max_depths': pd.Series(sdb_red_depths),
            'SDBgreen_max_depths': pd.Series(sdb_green_depths)
        })
        
        # Create empty columns for the averages, filled with NaN (Not a Number)
        output_df['SDBred_avg_max_depth'] = np.nan
        output_df['SDBgreen_avg_max_depth'] = np.nan

        # Now, place the calculated average ONLY in the first row (index 0) of the appropriate column.
        if sdb_red_depths:
            output_df.loc[0, 'SDBred_avg_max_depth'] = avg_red_depth
        if sdb_green_depths:
            output_df.loc[0, 'SDBgreen_avg_max_depth'] = avg_green_depth

        # Reorder columns to the desired format
        final_columns = [
            'SDBred_max_depths', 'SDBred_avg_max_depth',
            'SDBgreen_max_depths', 'SDBgreen_avg_max_depth'
        ]
        output_df = output_df[final_columns]

        # Define file path and save the CSV
        csv_filename = "max_depth_summary.csv"
        csv_filepath = os.path.join(output_folder, csv_filename)
        output_df.to_csv(csv_filepath, index=False, float_format='%.4f')
        print(f"\nDepth summary data saved to: {csv_filepath}")
    else:
        print("\nSkipping CSV generation because no valid depth data was collected.")

    # Prepare data for plotting
    labels = []
    averages = []
    stds = []
    colors = []
    counts = []

    # Only add bars if data exists for them, ensuring SDBred comes first if both are present
    if sdb_red_depths:
        labels.append('SDBred')
        averages.append(avg_red_depth)
        stds.append(std_red_depth)
        colors.append('red')
        counts.append(len(sdb_green_depths))

    if sdb_green_depths:
        labels.append('SDBgreen')
        averages.append(avg_green_depth)
        stds.append(std_green_depth)
        colors.append('green')
        counts.append(len(sdb_red_depths))

    if not labels:
        print("\nNo valid SDBred or SDBgreen data found to generate the bar chart after applying all selection criteria and file filtering. Please verify your data files, their names, and column headers.")
        return # Exit if no data to plot

    # Create the bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, averages, yerr=stds, capsize=5, color=colors, width=0.2)

    plt.ylabel('Average Max Depth Range (m)')
    plt.title(f'Average Max Depth Range for {aoi} ({sensor})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Changes the x distance between bars (it's a little weird, but just mess around with it until something works)
    plt.xlim(-0.9, len(labels) + -0.1)

    # Display count (n=X) on top of bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        # Use the standard deviation directly from the 'stds' list for text positioning.
        # This avoids the AttributeError if _yerr is not present on the Rectangle object.
        error_bar_offset = stds[i] if len(stds) > i else 0
        plt.text(bar.get_x() + bar.get_width()/2, yval + error_bar_offset + 0.05,
                 f"n={counts[i]}", ha='center', va='bottom')

    # Set y-axis limits to start from 0 and provide some padding above the highest bar
    max_height = 0
    for i in range(len(averages)):
        current_height = averages[i] + stds[i]
        if current_height > max_height:
            max_height = current_height

    plt.ylim(0, max_height * 1.2 if max_height > 0 else 1)

    # Save and show plot based on config
    plot_filename = f"{aoi}_{sensor}_Max_Depth_Range_Bar_Chart.png"
    plot_filepath = os.path.join(output_folder, plot_filename)

    if config.get('save_plots', False):
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to: {plot_filepath}")

    if config.get('show_plots', True):
        plt.show()
    else:
        plt.close()

    print("--- Depth Range Bar Chart Generation Complete. ---")




