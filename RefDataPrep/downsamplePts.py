# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:04:52 2024

@author: mccullru
"""

### Script for downsampling large point cloud data to more managable sizes ###

# import laspy
# #import json
# import pandas as pd
# import numpy as np
# #from scipy.spatial import cKDTree
# import os
# import glob
# from concurrent.futures import ProcessPoolExecutor



# # Function to import las file and downsample using a grid
# def downsample_lidar(input_file, grid_size, output_folder):
        
#     # Load the .las file using laspy
#     lidar_data = laspy.read(input_file)
#     print(f"Processing file: {input_file}")
    
    
#     # Extract the x, y, z coordinates from the lidar data
#     x = lidar_data.x
#     y = lidar_data.y
#     z = lidar_data.z
    
#     # Create the 2D grid by defining the bounds and grid size
#     x_min, x_max = np.min(x), np.max(x)
#     y_min, y_max = np.min(y), np.max(y)
    
    
#     # Create bins (grid cells) based on the defined grid size
#     x_bins = np.arange(x_min, x_max, grid_size)
#     y_bins = np.arange(y_min, y_max, grid_size)
    
    
#     downsampled_points = []

#     # Calculate the indices for grid cells
#     x_indices = np.digitize(x, x_bins) - 1
#     y_indices = np.digitize(y, y_bins) - 1
    
#     # Loop over unique grid cells (by unique x, y index pairs)
#     for i in np.unique(x_indices):
#          for j in np.unique(y_indices):
#              cell_mask = (x_indices == i) & (y_indices == j)
#              cell_points = np.where(cell_mask)[0]
    
#              if len(cell_points) > 0:
#                  # Calculate centroid
#                  centroid_x = np.mean(x[cell_points])
#                  centroid_y = np.mean(y[cell_points])
                 
#                  # Find point closest to centroid
#                  distances = (x[cell_points] - centroid_x)**2 + (y[cell_points] - centroid_y)**2
#                  closest_point_idx = cell_points[np.argmin(distances)]
                 
#                  # Ensure closest_point_idx is a scalar, not an array
#                  print(f"closest_point_idx type: {type(closest_point_idx)}")  # Debugging
#                  print(f"closest_point_idx value: {closest_point_idx}")  # Debugging
                 
                 
#                  # If closest_point_idx is an array, use the first index
#                  if isinstance(closest_point_idx, np.ndarray):
#                      closest_point_idx = closest_point_idx[0]
                 
                 
#                  downsampled_points.append([lidar_data.x[closest_point_idx], lidar_data.y[closest_point_idx], lidar_data.z[closest_point_idx]])
    
    
#     # Create a DataFrame and save to CSV
#     downsampled_df = pd.DataFrame(downsampled_points, columns=['x', 'y', 'z'])
#     base_name = os.path.splitext(os.path.basename(input_file))[0]
#     output_file = os.path.join(output_folder, f"{base_name}_downsampled.csv")
#     downsampled_df.to_csv(output_file, index=False)
#     print(f"Downsampled points for {input_file} saved to {output_file}")




# # Main function to manage multiple files
# def process_file(input_folder, grid_size, output_folder):
#     input_files = glob.glob(os.path.join(input_folder, "*.las")) + glob.glob(os.path.join(input_folder, "*.laz"))
#     print(f"Found {len(input_files)} files to process in {input_folder}.")


#     # Process files in parallel
#     with ProcessPoolExecutor() as executor:
#         futures = [executor.submit(downsample_lidar, input_file, grid_size, output_folder) for input_file in input_files]
#         for future in futures:
#             future.result()  # Collect results to handle exceptions






# ##############################################################################
# # Execute here

# if __name__ == '__main__':
#     # Grid cell size in meters
#     grid_size = 10  
    
#     # input and output file paths
#     input_folder = r"B:\\Thesis Project\\Reference Data\\Baker Bay\\wa2015_usace_ncmp_sand_island_Job1010668_LAS"  
#     output_folder = r"B:\Thesis Project\Spyder\Tests"
    
    
#     # Call the function and get the downsampled points as a DataFrame
#     process_file(input_folder, grid_size, output_folder)
    
#     print("All Done!")


import laspy
import numpy as np
import pandas as pd
import os
import glob
from rtree import index
from concurrent.futures import ProcessPoolExecutor

# Function to process a single grid cell (find the closest point)
def process_grid_cell(grid_cell, x, y, z, idx):
    x_cell_min, x_cell_max, y_cell_min, y_cell_max = grid_cell
    # Query points within the current grid cell from the R-tree
    point_indices = list(idx.intersection((x_cell_min, y_cell_min, x_cell_max, y_cell_max)))

    grid_points = []
    if point_indices:
        # Extract the points within the cell
        cell_points = np.column_stack((x[point_indices], y[point_indices], z[point_indices]))

        # Calculate the centroid of the current grid cell
        centroid_x = np.mean(cell_points[:, 0])
        centroid_y = np.mean(cell_points[:, 1])

        # Find the closest point to the centroid using vectorized distance calculation
        distances = np.sqrt((cell_points[:, 0] - centroid_x) ** 2 + (cell_points[:, 1] - centroid_y) ** 2)
        closest_point_idx = np.argmin(distances)

        # Get the closest point's x, y, z coordinates
        closest_point = cell_points[closest_point_idx]
        grid_points.append(closest_point)

    return grid_points

# Function to downsample points from a lidar file
def downsample_lidar(input_file, grid_size, output_folder):
    # Read the .las or .laz file
    lidar_data = laspy.read(input_file)
    print(f"Processing file: {input_file}")
    
    # Extract the x, y, z coordinates from the lidar data
    x = lidar_data.x
    y = lidar_data.y
    z = lidar_data.z
    
    # Calculate grid edges based on the given grid size
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Create a 2D grid using the user-defined cell size (grid_size)
    x_bins = np.arange(x_min, x_max, grid_size)
    y_bins = np.arange(y_min, y_max, grid_size)

    # Initialize R-tree for spatial indexing
    idx = index.Index()

    # Insert points into R-tree
    for i, (xi, yi) in enumerate(zip(x, y)):
        idx.insert(i, (xi, yi, xi, yi))  # Bounding box is just the point itself (xi, yi)

    grid_points = []

    # Prepare the grid cells (each grid cell defined by its x and y range)
    grid_cells = []
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            x_cell_min, x_cell_max = x_bins[i], x_bins[i + 1]
            y_cell_min, y_cell_max = y_bins[j], y_bins[j + 1]
            grid_cells.append((x_cell_min, x_cell_max, y_cell_min, y_cell_max))

    # Prepare the arguments to pass to the executor (grid_cells, x, y, z, idx)
    args = [(cell, x, y, z, idx) for cell in grid_cells]

    # Parallelize grid cell processing using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_grid_cell, *zip(*args))  # Unzip arguments and pass them as separate lists

    # Collect downsampled points from the results
    for result in results:
        grid_points.extend(result)

    # Convert the downsampled points into a DataFrame
    downsampled_df = pd.DataFrame(grid_points, columns=['x', 'y', 'z'])
    
    # Save the downsampled points to a CSV file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_folder, f"{base_name}_downsampled.csv")
    downsampled_df.to_csv(output_file, index=False)
    print(f"Downsampled points saved to {output_file}")

# Function to process multiple files
def process_files(input_folder, grid_size, output_folder):
    # Get all .las and .laz files from the input folder
    input_files = glob.glob(os.path.join(input_folder, "*.las")) + glob.glob(os.path.join(input_folder, "*.laz"))
    print(f"Found {len(input_files)} files to process in {input_folder}.")
    
    # Process each file
    for input_file in input_files:
        downsample_lidar(input_file, grid_size, output_folder)

# Main execution
if __name__ == "__main__":
    # Set grid size (cell size) in meters
    grid_size = 10  # Adjust as needed
    
    # Specify input folder containing LAS/LAZ files
    input_folder = r"B:\Thesis Project\Reference Data\Baker Bay\wa2015_usace_ncmp_sand_island_Job1010668_LAS"  # Modify with your folder path
    
    # Specify output folder for the CSV files
    output_folder = r"B:\Thesis Project\Spyder\Tests"  # Modify with your output folder path
    
    # Call the function to process the files
    process_files(input_folder, grid_size, output_folder)
    
    print("Processing completed!")




