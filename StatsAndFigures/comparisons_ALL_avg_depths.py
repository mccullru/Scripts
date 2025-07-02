# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 12:52:53 2025

@author: mccullru
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score

def aggregate_depth_summaries(base_directory, output_directory):
    """
    Aggregates max depth summary data from a nested folder structure and
    returns a dictionary with paths to the created summary files.
    """
    print("--- Step 1: Aggregating Data Summaries ---")
    os.makedirs(output_directory, exist_ok=True)

    base_path = Path(base_directory)
    search_pattern = "**/Figures/Average_Depth_Ranges/max_depth_summary.csv"
    summary_files = list(base_path.glob(search_pattern))

    if not summary_files:
        print("Error: No 'max_depth_summary.csv' files found.")
        return {'red': None, 'green': None}

    print(f"Found {len(summary_files)} summary files to process.\n")

    all_red_data, all_green_data = [], []

    for file_path in summary_files:
        try:
            path_parts = file_path.parts
            aoi = path_parts[-4]
            platform = path_parts[-5]
            df = pd.read_csv(file_path)

            if 'SDBred_max_depths' in df.columns:
                red_depths = df['SDBred_max_depths'].dropna()
                if not red_depths.empty:
                    all_red_data.append(pd.DataFrame({
                        'SDBred_max_depths': red_depths, 'AOI': aoi, 'Platform': platform
                    }))

            if 'SDBgreen_max_depths' in df.columns:
                green_depths = df['SDBgreen_max_depths'].dropna()
                if not green_depths.empty:
                    all_green_data.append(pd.DataFrame({
                        'SDBgreen_max_depths': green_depths, 'AOI': aoi, 'Platform': platform
                    }))
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    output_paths = {'red': None, 'green': None}

    if all_red_data:
        final_red_summary_df = pd.concat(all_red_data, ignore_index=True)
        red_output_path = os.path.join(output_directory, 'SDBred_summary.csv')
        final_red_summary_df.to_csv(red_output_path, index=False)
        print(f"Successfully created SDB Red summary at: {red_output_path}")
        output_paths['red'] = red_output_path

    if all_green_data:
        final_green_summary_df = pd.concat(all_green_data, ignore_index=True)
        green_output_path = os.path.join(output_directory, 'SDBgreen_summary.csv')
        final_green_summary_df.to_csv(green_output_path, index=False)
        print(f"Successfully created SDB Green summary at: {green_output_path}")
        output_paths['green'] = green_output_path

    return output_paths


def analyze_and_plot_depth_differences(input_csv_path, output_directory):
    """
    Performs a full statistical analysis, creates an individual comparison scatter plot,
    and returns the processed DataFrame for combined plotting.
    """
    if not input_csv_path or not os.path.exists(input_csv_path):
        print(f"Warning: Input file path is invalid: {input_csv_path}. Skipping analysis.")
        return None

    filename = os.path.basename(input_csv_path)
    # FIX: Define plot colors based on the sdb_type
    if 'red' in filename.lower():
        sdb_type = 'SDBred'
        depth_col = 'SDBred_max_depths'
        marker_color = 'red'
        error_color = 'salmon'
        marker_shape='o'
    elif 'green' in filename.lower():
        sdb_type = 'SDBgreen'
        depth_col = 'SDBgreen_max_depths'
        marker_color = 'darkgreen'
        error_color = 'green'
        marker_shape='s'
    else:
        # Fallback case
        sdb_type = 'Unknown'
        depth_col = ''
        marker_color = 'royalblue'
        error_color = 'lightgray'

    print(f"\n--- Running Individual Analysis for {sdb_type} ---")
    os.makedirs(output_directory, exist_ok=True)

    df = pd.read_csv(input_csv_path)

    summary_stats = df.groupby(['Platform', 'AOI'])[depth_col].agg(['mean', 'std']).reset_index()
    pivot_df = summary_stats.pivot(index='AOI', columns='Platform', values=['mean', 'std'])
    pivot_df.columns = [f'{val}_{platform}' for val, platform in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    if 'mean_Sentinel2' not in pivot_df.columns or 'mean_SuperDove' not in pivot_df.columns:
        print(f"Warning: Missing S2 or SD data for {sdb_type}. Cannot proceed with analysis.")
        return None

    pivot_df['difference'] = pivot_df['mean_SuperDove'] - pivot_df['mean_Sentinel2']
    pivot_df['propagated_std'] = np.sqrt(pivot_df['std_SuperDove']**2 + pivot_df['std_Sentinel2']**2)
    
    analysis_output_path = os.path.join(output_directory, f'analysis_summary_{sdb_type}.csv')
    pivot_df.to_csv(analysis_output_path, index=False, float_format='%.4f')
    print(f"Saved detailed analysis to: {analysis_output_path}")

    analysis_data = pivot_df.dropna(subset=['mean_SuperDove', 'mean_Sentinel2'])
    if len(analysis_data) > 1:
        t_statistic, p_value = ttest_rel(analysis_data['mean_SuperDove'], analysis_data['mean_Sentinel2'])
        print("\n--- Paired t-test Results ---")
        print(f"T-statistic: {t_statistic:.4f}, P-value: {p_value:.4f}")
        print("Conclusion: The difference is statistically significant (p < 0.05)." if p_value < 0.05 else 
              "Conclusion: The difference is NOT statistically significant (p >= 0.05).")
    
    print("Generating individual comparison scatter plot...")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.errorbar(x=pivot_df['mean_Sentinel2'], y=pivot_df['mean_SuperDove'],
                xerr=pivot_df['std_Sentinel2'], yerr=pivot_df['std_SuperDove'],
                fmt=marker_shape, capsize=5, color=marker_color, ecolor=error_color, markersize='8', label='AOI Comparison')
    
 
    # 1. Validate data for regression
    x_data_raw = pivot_df['mean_Sentinel2']
    y_data_raw = pivot_df['mean_SuperDove']
    finite_mask = np.isfinite(x_data_raw) & np.isfinite(y_data_raw)
    x_data = x_data_raw[finite_mask]
    y_data = y_data_raw[finite_mask]

    # 2. Calculate and plot regression line if there are enough points
    if len(x_data) >= 2:
        m, b = np.polyfit(x_data, y_data, 1) # Calculate slope and intercept
        x_line = np.array([0, 25])
        y_line = m * x_line + b

        # Calculate R²
        y_pred = m * x_data + b
        r2_val = r2_score(y_data, y_pred)

        # Plot the line and create a label with the R² value
        regression_label = f'Regression Line (R² = {r2_val:.2f})'
        ax.plot(x_line, y_line, color='black', linestyle='-', linewidth=2.5, label=regression_label, zorder=3)
    else:
        print(f"Warning: Not enough valid data points for {sdb_type} regression. Skipping line.")

    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, 25], [0, 25], 'k--', alpha=0.8)

    ax.set_xlabel('Avg S-2 Max Depth (m)', fontsize=20)
    ax.set_ylabel('Avg SD Max Depth (m)', fontsize=20)
    ax.set_title(f'Max Optical Depth Comparison: {sdb_type}', fontsize=22, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(fontsize=20, edgecolor='black')
    plt.tight_layout()
    ax.set_xlim(0,25)
    ax.set_ylim(0,25)

    plot_output_path = os.path.join(output_directory, f'scatter_comparison_{sdb_type}.png')
    plt.savefig(plot_output_path, dpi=300)
    print(f"Plot saved to: {plot_output_path}")
    plt.show()

    # Return the processed data for the combined plot
    return pivot_df


def plot_combined_chart(red_data, green_data, output_directory):
    """
    Creates a single scatter plot comparing SDBred and SDBgreen results.
    """
    if red_data is None and green_data is None:
        print("\nNo data available to generate a combined plot.")
        return

    print("\n--- Generating Combined SDBred vs. SDBgreen Plot ---")
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot SDBred data if it exists
    if red_data is not None:
        
        # --- CHANGE START: Validate SDBred data before polyfit ---
        x_red_raw = red_data['mean_Sentinel2']
        y_red_raw = red_data['mean_SuperDove']
       
        # Create a mask for finite (not NaN, not Inf) values in both arrays
        finite_mask_red = np.isfinite(x_red_raw) & np.isfinite(y_red_raw)
        x_red = x_red_raw[finite_mask_red]
        y_red = y_red_raw[finite_mask_red]
       
        # Only proceed if there are enough points to calculate a line
        if len(x_red) >= 2:
            m_red, b_red = np.polyfit(x_red, y_red, 1)
            x_line_red = np.array([0, 25])
            y_line_red = m_red * x_line_red + b_red
            ax.plot(x_line_red, y_line_red, color='red', linestyle='-', linewidth=2.5, zorder=3)
            ax.set_xlim(0,25)
            ax.set_ylim(0,25)
            
            # Calculate R² for the legend
            y_pred_red = m_red * x_red + b_red
            r2_red = r2_score(y_red, y_pred_red)
            red_label = f'SDBred (R² = {r2_red:.2f})'
        else:
            print("Warning: Not enough valid data points for SDBred regression. Skipping line.")
            red_label = 'SDBred'
        
        ax.errorbar(x=red_data['mean_Sentinel2'], y=red_data['mean_SuperDove'],
                    xerr=red_data['std_Sentinel2'], yerr=red_data['std_SuperDove'],
                    fmt='o', capsize=5, color='red', ecolor='salmon',
                    label='SDBred', markersize='8')

    # Plot SDBgreen data if it exists
    if green_data is not None:
        
        # Validate SDBgreen data before polyfit ---
        x_green_raw = green_data['mean_Sentinel2']
        y_green_raw = green_data['mean_SuperDove']

        # Create a mask for finite (not NaN, not Inf) values in both arrays
        finite_mask_green = np.isfinite(x_green_raw) & np.isfinite(y_green_raw)
        x_green = x_green_raw[finite_mask_green]
        y_green = y_green_raw[finite_mask_green]

        # Only proceed if there are enough points to calculate a line
        if len(x_green) >= 2:
            m_green, b_green = np.polyfit(x_green, y_green, 1)
            x_line_green = np.array([0, 25])
            y_line_green = m_green * x_line_green + b_green
            ax.plot(x_line_green, y_line_green, color='darkgreen', linestyle='-', linewidth=2.5, zorder=3)
            ax.set_xlim(0,25)
            ax.set_ylim(0,25)
            
            # Calculate R² for the legend
            y_pred_green = m_green * x_green + b_green
            r2_green = r2_score(y_green, y_pred_green)
            green_label = f'SDBgreen (R² = {r2_green:.2f})'
        else:
            print("Warning: Not enough valid data points for SDBgreen regression. Skipping line.")
            green_label = 'SDBgreen'


        ax.errorbar(x=green_data['mean_Sentinel2'], y=green_data['mean_SuperDove'],
                    xerr=green_data['std_Sentinel2'], yerr=green_data['std_SuperDove'],
                    fmt='s', capsize=5, color='darkgreen', ecolor='green', 
                    label='SDBgreen', markersize='8')

    # Determine axis limits to include all data
    max_x = max(red_data['mean_Sentinel2'].max(), green_data['mean_Sentinel2'].max()) if red_data is not None and green_data is not None else 0
    max_y = max(red_data['mean_SuperDove'].max(), green_data['mean_SuperDove'].max()) if red_data is not None and green_data is not None else 0
    max_val = max(max_x, max_y, 1) # Use 1 as a minimum to avoid empty plots

    # Plot the 1:1 line of agreement
    ax.plot([0, 25], [0, 25], 'k--')

    # Add labels and titles
    ax.set_xlabel('Avg S-2 Max Depth (m)', fontsize=20)
    ax.set_ylabel('Avg SD Max Depth (m)', fontsize=20)
    ax.set_title('Max Optical Depth Comparison: SDBgreen and SDBred', fontsize=22, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(fontsize=20, edgecolor='black')
    plt.tight_layout()
    ax.set_xlim(0,25)
    ax.set_ylim(0,25)


    # Save the plot
    plot_output_path = os.path.join(output_directory, 'scatter_comparison_COMBINED.png')
    plt.savefig(plot_output_path, dpi=300)
    print(f"Combined plot saved to: {plot_output_path}")
    plt.show()









if __name__ == '__main__':
    # --- User Configuration ---
    BASE_DATA_FOLDER = r"B:\Thesis Project\SDB_Time\Results_main"
    MAIN_OUTPUT_FOLDER = r"B:\Thesis Project\SDB_Time\Max Depth Analysis"

    aggregated_data_dir = os.path.join(MAIN_OUTPUT_FOLDER, "aggregated_csvs")
    analysis_dir = os.path.join(MAIN_OUTPUT_FOLDER, "statistical_analysis")

    # 1. Aggregate data from all AOI folders
    summary_paths = aggregate_depth_summaries(BASE_DATA_FOLDER, aggregated_data_dir)

    # 2. Run individual analysis and get the processed data back
    red_analysis_data = None
    if summary_paths and summary_paths['red']:
        red_analysis_data = analyze_and_plot_depth_differences(summary_paths['red'], analysis_dir)

    green_analysis_data = None
    if summary_paths and summary_paths['green']:
        green_analysis_data = analyze_and_plot_depth_differences(summary_paths['green'], analysis_dir)

    # 3. Create the final combined plot using the returned data
    plot_combined_chart(red_analysis_data, green_analysis_data, analysis_dir)

    print("\n--- Full Pipeline Finished ---")













