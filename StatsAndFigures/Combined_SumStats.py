# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:40:13 2025

@author: mccullru
"""


import os
import pandas as pd


from scipy.stats import ttest_ind

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================

# 1. Define Base Path for the main results folder
BASE_PATH = r"B:\Thesis Project\SDB_Time\Results_main"

# 2. Define the lists to iterate through
SENSORS = ['SuperDove', 'Sentinel2']
AOIS = [
    'Anegada', 'Bombah', 'BumBum', 'Gyali', 'Homer', 'Hyannis', 'Marathon', 'Nait', 'NorthFeut',
    'Punta', 'Rago', 'Risoysundet', 'Skutvik', 'SouthPort'
]
SDB_TYPES = ['SDB_red', 'SDB_green', 'SDB_merged']

# 3. Define the path for the final summary output file
OUTPUT_FILE = r"B:\Thesis Project\StatsAndFigures\Summary Stats\Overall_Sensor_SDB_Type_Summary.csv"

# =============================================================================
# --- SCRIPT EXECUTION ---
# =============================================================================

def main():
    """Finds, combines, and averages summary statistics for all sensors, AOIs, and SDB types."""
    final_summary_list = []

    # --- CHANGE 1: Add a dictionary to store the MSE lists for the t-test ---
    mse_data_for_ttest = {
        'SuperDove': {'SDB_red': [], 'SDB_green': [], 'SDB_merged': []},
        'Sentinel2': {'SDB_red': [], 'SDB_green': [], 'SDB_merged': []}
    }

    # Loop through each sensor type
    for sensor in SENSORS:
        # Loop through each SDB type
        for sdb_type in SDB_TYPES:
            
            print(f"--- Processing: {sensor} - {sdb_type} ---")
            
            # List to hold the data from each AOI's summary file
            dfs_to_combine = []

            # Loop through all AOIs to find the matching summary file
            for aoi in AOIS:
                # Handle filename abbreviations (SuperDove -> SD, Sentinel-2 -> S2)
                sensor_in_filename = 'SD' if sensor == 'SuperDove' else 'S2'
                
                # Construct the expected filename and full path
                filename = f"summary_stats_{aoi}_{sensor_in_filename}_{sdb_type}.csv"
                filepath = os.path.join(BASE_PATH, sensor, aoi, "Figures", "Summary_Stats", filename)

                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        dfs_to_combine.append(df)
                    except Exception as e:
                        print(f"  - Could not read {filename}: {e}")
                else:
                    # This is not an error, just means no data for this combo
                    print(f"  - File not found, skipping: {filename}")
            
            # After checking all AOIs, process the collected data for this group
            if not dfs_to_combine:
                print(f"  -> No data found for {sensor} - {sdb_type}. Skipping summary.")
                continue

            # Combine all found dataframes into one
            combined_df = pd.concat(dfs_to_combine, ignore_index=True)

            # --- CHANGE 2: Store the list of MSEs for the t-test ---
            # We store the MSE values from each AOI before they are averaged
            if 'MSE' in combined_df.columns:
                mse_data_for_ttest[sensor][sdb_type] = combined_df['MSE'].dropna().values

            # --- Calculate the final averaged and summed statistics ---
            # Define which columns get averaged and which get summed
            cols_to_average = ['Mean', 'Std Dev', 'Min', 'Max', 'MAE', 'MSE', 'RMSE', 'Kurtosis', 'Skewness']
            cols_to_sum = ['Total Pts', 'Files Used']

            # Calculate the average of the stat metrics
            avg_stats = combined_df[cols_to_average].mean()
            # Calculate the sum of the counts
            sum_stats = combined_df[cols_to_sum].sum()

            # Create a dictionary for the final results row
            final_stat_row = {
                'Sensor': sensor,
                'SDB_Type': sdb_type,
                **avg_stats.to_dict(),  # Unpack the averaged stats
                **sum_stats.to_dict()   # Unpack the summed stats
            }
            final_summary_list.append(final_stat_row)
            print(f"  -> Finished aggregation for {sensor} - {sdb_type}. Used data from {len(dfs_to_combine)} AOIs.")

    # --- Save the final results to a single CSV file ---
    if not final_summary_list:
        print("\nNo data was processed. Final summary not created.")
        return

    final_df = pd.DataFrame(final_summary_list)
    
    # --- Perform t-test and add p-value column ---
    final_df['p-value (MSE vs other sensor)'] = pd.NA # Initialize new column
 
    for sdb_type in SDB_TYPES:
        # Get the MSE data for both sensors for the current SDB type
        sd_mses = mse_data_for_ttest['SuperDove'][sdb_type]
        s2_mses = mse_data_for_ttest['Sentinel2'][sdb_type]
        
        # Perform the t-test only if both have enough data
        if len(sd_mses) >= 2 and len(s2_mses) >= 2:
            # ttest_ind performs a two-sided t-test for two independent samples
            t_stat, p_value = ttest_ind(sd_mses, s2_mses, equal_var=False) # Welch's t-test
            
            # Add the calculated p-value to the correct rows in the final DataFrame
            final_df.loc[final_df['SDB_Type'] == sdb_type, 'p-value (MSE vs other sensor)'] = p_value
            print(f"\nCalculated t-test for {sdb_type}: p-value = {p_value:.4f}")
        else:
            print(f"\nNot enough data to perform t-test for {sdb_type}.")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    final_df.to_csv(OUTPUT_FILE, index=False, float_format='%.4f')
    print(f"\nSUCCESS: Final summary of all sensors and SDB types saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
    print("\n--- Script finished. ---")
    
    
    
    
    
    
    
    
    
    
    
    
    
    