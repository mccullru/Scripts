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


def process_single_csv(input_file, output_folder):
    """
    Processes a single CSV file, performs linear regression, 
    and saves the results in the output folder.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return

    print(f"Processing {input_file}...")

    # Read the CSV file
    data = pd.read_csv(input_file)

    # Drop rows where 'Raster_Value' is blank
    data = data.dropna(subset=['Raster_Value'])

    # Perform linear regression
    x = data[['Raster_Value']].values
    y = data['Orthometric Height(m)'].values
    model = LinearRegression()
    model.fit(x, y)

    # Calculate regression metrics
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)  # Scikit-learn R² calculation
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Create the line of best fit equation
    coef = model.coef_[0]
    intercept = model.intercept_
    equation = f"y = {coef:.4f}x + {intercept:.4f}"

    # Calculate perpendicular distances
    distances = np.abs(coef * x.flatten() - y + intercept) / np.sqrt(coef**2 + 1)

    # Compute statistics for distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    # Append the results to the list
    results = {
        "Image Name": os.path.basename(input_file),
        "R^2": r2,
        "RMSE": rmse,
        "Line of Best Fit": equation,
        "m1": coef,
        "m0": intercept,
        "min perp dist": min_dist,
        "max perp dist": max_dist,
        "mean perp dist": mean_dist,
        "std dev perp dist": std_dist
    }

    # Regression Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.7)
    plt.plot(x, y_pred, color='red', linewidth=2)
    
    plt.title('Sentinel-2 Linear Regression', fontsize=30)
    
    plt.xlabel("pSDB values", fontsize=24)
    plt.ylabel("Reference Depth (m)", fontsize=24)
    plt.legend()
    plt.grid(True)
    
    plt.xlim(.9, None)
    plt.ylim(None, 0.9)

    # Add R^2 and RMSE as text on the plot
    max_x = np.max(x)
    max_y = np.max(y)
    min_y = np.min(y)
    
    plt.text(.96, -6.5, f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}", fontsize=18, color='black', ha='left',
              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Invert both axes so 0 is bottom left, and up and right are negative
    #plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    # Save the regression plot in the output folder
    plot_filename = f"{os.path.splitext(os.path.basename(input_file))[0]}_LR_plot_better.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path)
    plt.close()



input_file = r"E:\Thesis Stuff\Results\Marathon\Condition1_dsSD\Extracted Pts\pSDB\Marathon_S2A_MSI_2023_02_14_16_06_29_T17RMH_L2W__RGB_pSDBred_extracted.csv"
output_folder = r"E:\Thesis Stuff\Results\Marathon\Condition1_dsSD\Extracted Pts\pSDB\test"
process_single_csv(input_file, output_folder)


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
# csv_file = r"E:\Thesis Stuff\Results\Marathon\Condition2\Extracted Pts\SDB\Marathon_S2A_MSI_2023_02_14_16_06_29_T17RMH_L2W__RGB_SDBred_extracted.csv"
# df = pd.read_csv(csv_file)

# # Extract relevant columns
# ref_col = "Orthometric Height(m)"
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
# plt.title("Sentinel-2 SDB Error Distribution", fontsize=18)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xlim(-2,2)
# plt.ylim(0,150)

# plt.text(-1.85, 85, f"Mean = {stats['mean']:.2f}\nMin = {stats['min']:.2f}\nMax = {stats['max']:.2f}\nStd Dev = {stats['std']:.2f}\nRMSE = {stats['RMSE']:.2f}\nCount = {stats['count']:.0f}", fontsize=16, color='black', ha='left',
#          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))






