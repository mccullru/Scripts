# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:47:25 2025

@author: mccullru
"""


##############################################################################################################
##############################################################################################################

# Linear Regressions

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


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

# # Error Histograms

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



# # Load CSV file
# csv_file = r"E:\Thesis Stuff\SDB_ExtractedPts\PlanetScope_24c5_2023_02_20_15_07_07_L2W__RGB_SDBgreen_extracted.csv"
# df = pd.read_csv(csv_file)

# # Extract relevant columns
# ref_col = "Geoid_Corrected_Ortho_Height"
# SDB_col = "Raster_Value"

# # Compute error
# df["Error"] = df[ref_col] - df[SDB_col]

# # Compute statistics
# stats = df["Error"].describe()
# stats["RMSE"] = (df["Error"] ** 2).mean() ** 0.5  # Root Mean Square Error

# # Print results
# print(stats)


# # Plot histogram of errors
# plt.figure(figsize=(8, 5))
# plt.hist(df["Error"], bins=30, edgecolor='black', alpha=0.7)
# plt.xlabel("Error (m)", fontsize=14)
# plt.ylabel("Count", fontsize=14)
# plt.title("PSSgreen SDB Error Distribution", fontsize=18)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xlim(-3,3)
# plt.ylim(0,60)


# plt.text(-1.85, 35, f"Mean = {stats['mean']:.3f}\nMin = {stats['min']:.2f}\nMax = {stats['max']:.2f}\nStd Dev = {stats['std']:.2f}\nRMSE = {stats['RMSE']:.2f}\nCount = {stats['count']:.0f}", fontsize=16, color='black', ha='left',
#           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))






##############################################################################################################
##############################################################################################################

"""
Creates box and whisker plots out of all the RMSE values (1 for each image) for each AOI, and overlays all R^2
values from each regression line over each box plot. The purpose of this is to visualize not only the R^2 values 
(which can't tell you everything anyway), but the RMSE as well to get a better picture. 


"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# --- 1. Sample Data Creation (Replace this with loading your actual data) ---
np.random.seed(42)
sites = [f'Site_{i}' for i in range(1, 4)]
sensors = ['Sentinel-2', 'SuperDove']
algorithms = ['pSDBred', 'pSDBgreen']
data_list = []
for site in sites:
    for sensor in sensors:
        n_images = 5 if sensor == 'Sentinel-2' else 20
        for algo in algorithms:
            for i in range(n_images):
                base_rmse = 0.8 if sensor == 'Sentinel-2' else 1.1
                base_r2 = 0.85 if sensor == 'Sentinel-2' else 0.75
                rmse = max(0.1, base_rmse + np.random.randn() * 0.3 + (0.1 if algo == 'pSDBred' else -0.1))
                r2 = min(0.99, max(0.1, base_r2 + np.random.randn() * 0.1 + (0.05 if algo == 'pSDBred' else -0.05)))
                data_list.append({'Site': site, 'Sensor': sensor, 'Algorithm': algo, 'RMSE': rmse, 'R2': r2})
results_df = pd.DataFrame(data_list)
results_df['Group'] = results_df['Sensor'] + ' - ' + results_df['Algorithm'] # Example grouping
print("Sample DataFrame head:")
print(results_df.head())
# --- End Sample Data Creation ---


# --- 2. Prepare Data for Matplotlib Boxplot ---
grouping_variable = 'Group' # Column to group by on x-axis
y_metric = 'RMSE'           # Column for boxplot y-axis
color_metric = 'R2'         # Column to map to color

unique_groups = sorted(results_df[grouping_variable].unique())
# Data for boxplot: list of arrays, one array of RMSE values per group
data_to_plot = [results_df.loc[results_df[grouping_variable] == grp, y_metric].dropna().values
                for grp in unique_groups]
# X positions for the boxes
box_positions = np.arange(len(unique_groups)) + 1


# --- 3. Create the Plot ---
fig, ax = plt.subplots(figsize=(10, 7)) # Adjust size as needed

# Create the box plots
# patch_artist=True allows filling boxes with color (optional)
# showfliers=False prevents boxplot from drawing outliers, scatter will show them
bp = ax.boxplot(data_to_plot, positions=box_positions, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.6), # Example box styling
                medianprops=dict(color='red', linewidth=1.5),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'))

# --- 4. Prepare for Scatter Overlay ---
# Create a mapping from group name to x position
group_to_xpos = {group: pos for group, pos in zip(unique_groups, box_positions)}
# Get x position for each row in the original dataframe
x_scatter = results_df[grouping_variable].map(group_to_xpos)

# Add jitter (small random horizontal shift) to x positions to reduce overlap
jitter_amount = 0.08
#x_scatter_jitter = x_scatter + np.random.normal(0, jitter_amount, size=len(results_df))

# Get the y values and color values for the scatter plot
y_scatter = results_df[y_metric]
color_values = results_df[color_metric]

# --- 5. Normalize R2 values and choose colormap ---
norm = mcolors.Normalize(vmin=color_values.min(), vmax=color_values.max())
cmap = cm.RdYlGn # Choose a colormap (e.g., viridis, plasma, coolwarm)

# --- 6. Plot Scatter Overlay ---
scatter_plot = ax.scatter(x_scatter, y_scatter,
                          c=color_values, # Use R2 values for color
                          cmap=cmap,      # Apply the colormap
                          norm=norm,      # Apply the normalization
                          alpha=0.9,      # Point transparency 
                          #edgecolor='k',
                          s=50,           # Adjust marker size if needed
                          zorder=3)       # Draw scatter on top of boxes

# --- 7. Add Color Bar ---
cbar = fig.colorbar(scatter_plot, ax=ax)
cbar.set_label(f'{color_metric} Value', rotation=270, labelpad=15)

# --- 8. Customize and Show Plot ---
ax.set_xticks(box_positions) # Set ticks at the box positions
ax.set_xticklabels(unique_groups, rotation=45, ha='right') # Set group names as labels
ax.set_title(f'Distribution of {y_metric} with points colored by {color_metric}', fontsize=16)
ax.set_xlabel("Group", fontsize=12)
ax.set_ylabel(y_metric, fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout() # Adjust layout
plt.show()



















