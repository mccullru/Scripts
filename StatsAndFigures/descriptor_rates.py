# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 17:47:50 2025

@author: mccullru
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================

# 1. Define the full path to your input data file
DATA_FILE_PATH = r"B:\Thesis Project\StatsAndFigures\Descriptor rates\Totals_v2.csv" # <-- IMPORTANT: UPDATE THIS PATH

# 2. Define the folder where you want to save the charts
OUTPUT_FOLDER_PATH = r"B:\Thesis Project\StatsAndFigures\Descriptor rates" # <-- IMPORTANT: UPDATE THIS PATH

# =============================================================================
# --- SCRIPT EXECUTION ---
# =============================================================================

def create_bar_charts(data_path, output_folder):
    """
    Reads summary data, creates an average bar chart, and then a
    grouped bar chart for each individual AOI.
    """
    os.makedirs(output_folder, exist_ok=True)

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_path}'")
        return

    aoi_list = df['AOI'].unique()
    category_columns = [col for col in df.columns if col not in ['AOI', 'Sensor']]

    # =======================================================================
    # --- NEW: Calculate and Plot Average Chart Across All AOIs ---
    # =======================================================================
    print("--- Generating Average Chart for All AOIs ---")
    
    # Group by 'Sensor' and calculate the mean for all category columns
    average_df = df.groupby('Sensor')[category_columns].mean().reset_index()

    # Separate the averaged data for SD and S2
    sd_avg_data = average_df[average_df['Sensor'] == 'SD']
    s2_avg_data = average_df[average_df['Sensor'] == 'S2']

    if not sd_avg_data.empty and not s2_avg_data.empty:
        sd_avg_values = sd_avg_data[category_columns].iloc[0].values
        s2_avg_values = s2_avg_data[category_columns].iloc[0].values

        fig_avg, ax_avg = plt.subplots(figsize=(15, 8))
        x = np.arange(len(category_columns))
        bar_width = 0.35

        ax_avg.bar(x - bar_width/2, sd_avg_values, bar_width, label='SuperDove (SD) Average', color='red')
        ax_avg.bar(x + bar_width/2, s2_avg_values, bar_width, label='Sentinel-2 (S2) Average', color='blue')

        ax_avg.set_ylabel('Tot descriptor / Tot Q1&Q2 images', fontsize=12)
        ax_avg.set_title('Average Q1 & Q2 Descriptor Rates', fontsize=16, fontweight='bold')
        ax_avg.set_xticks(x)
        ax_avg.set_xticklabels(category_columns, rotation=45, ha="right")
        ax_avg.legend()
        ax_avg.grid(axis='y', linestyle='--', alpha=0.7)
        ax_avg.margins(x=0.02)
        fig_avg.tight_layout()

        output_path_avg = os.path.join(output_folder, "Average_All_AOIs_comparison.png")
        try:
            plt.savefig(output_path_avg, dpi=300)
            print(f"Average chart saved to {output_path_avg}")
        except Exception as e:
            print(f"Could not save average chart. Error: {e}")
        
        plt.close(fig_avg)
    else:
        print("Could not generate average chart: Missing data for SD or S2.")

    # =======================================================================
    # --- Existing Loop for Individual AOI Charts ---
    # =======================================================================
    print("\n--- Generating Individual Charts for Each AOI ---")
    for aoi in aoi_list:
        print(f"Generating chart for {aoi}...")

        aoi_df = df[df['AOI'] == aoi]
        sd_data = aoi_df[aoi_df['Sensor'] == 'SD']
        s2_data = aoi_df[aoi_df['Sensor'] == 'S2']

        if sd_data.empty or s2_data.empty:
            print(f"  Skipping {aoi}: Missing data for either SD or S2.")
            continue

        sd_values = sd_data[category_columns].iloc[0].values
        s2_values = s2_data[category_columns].iloc[0].values

        fig, ax = plt.subplots(figsize=(15, 8))
        x = np.arange(len(category_columns))
        bar_width = 0.35

        ax.bar(x - bar_width/2, sd_values, bar_width, label='SuperDove (SD)', color='red')
        ax.bar(x + bar_width/2, s2_values, bar_width, label='Sentinel-2 (S2)', color='blue')

        ax.set_ylabel('Proportion / Tot AOI Q1&Q2 images', fontsize=12)
        ax.set_title(f'Category Comparison for {aoi}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(category_columns, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.margins(x=0.02)
        fig.tight_layout()

        output_path = os.path.join(output_folder, f"{aoi}_category_comparison.png")
        try:
            plt.savefig(output_path, dpi=300)
            print(f"  Chart saved to {output_path}")
        except Exception as e:
            print(f"  Could not save chart for {aoi}. Error: {e}")
        
        plt.close(fig)

# --- Run the main function ---
if __name__ == "__main__":
    create_bar_charts(DATA_FILE_PATH, OUTPUT_FOLDER_PATH)
    print("\nAll charts generated.")