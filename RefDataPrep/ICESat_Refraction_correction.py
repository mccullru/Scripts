# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:07:44 2025

@author: mccullru
"""

import numpy as np
import h5py
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pyproj

##############################################################################################################
##############################################################################################################
# Imports a h5 file and reads the point coords, ref_azimuth, and ref_elev and saves them to a csv


# # Input folder containing HDF5 files
# input_folder = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\full_h5_files\others"

# # Output folder for CSV files
# output_folder = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\new_icesat\other"
# os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# # Loop through all HDF5 files in the input folder
# for h5_filename in os.listdir(input_folder):
#     if h5_filename.endswith(".h5"):
#         h5_filepath = os.path.join(input_folder, h5_filename)
        
#         # Extract the suffix (e.g., "gt1l", "gt3r", etc...) from the filename
#         # Assuming suffix is the last part before .h5 (Had to add the suffix in manually)
#         suffix = h5_filename.split("_")[-1].split(".")[0]  # Assuming suffix is the last part before .h5
        
#         # Define dataset paths using the suffix
#         base_path = f"{suffix}/geolocation"
#         datasets = ["ref_azimuth", "ref_elev", "reference_photon_lat", "reference_photon_lon"]
        
#         try:
#             # Read the HDF5 file
#             with h5py.File(h5_filepath, 'r') as hdf:
                
#                 # Extract datasets
#                 data = {}
#                 for ds in datasets:
#                     dataset_path = f"{base_path}/{ds}"
#                     if dataset_path in hdf:
#                         data[ds] = np.array(hdf[dataset_path], dtype=np.float64)
#                     else:
#                         print(f"Warning: {dataset_path} not found in {h5_filename}")
#                         data[ds] = np.array([])  # Empty array if dataset is missing
                
#                 # Convert data to a DataFrame
#                 h5_df = pd.DataFrame(data)
                
#                 # Define output CSV file path
#                 output_csv = os.path.join(output_folder, h5_filename.replace(".h5", "_info.csv"))
                
#                 # Save DataFrame to CSV
#                 h5_df.to_csv(output_csv, index=False, float_format='%.15g')
                
#                 print(f"Processed {h5_filename} -> {output_csv}")
        
#         except Exception as e:
#             print(f"Error processing {h5_filename}: {e}")

# print("Processing complete.")


##############################################################################################################
##############################################################################################################

# Project itrf14 geodetic coordinates to a UTM projection
# Takes in both orig and new icesat tracks, and a csv with the track name and its corresponding UTM zones

# import numpy as np
# import h5py
# import os
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point
# import pyproj

# def find_matching_file(file_name, folder):
#     # Extract the common part of the file name (between ATL03_ and the next underscore)
#     suffix = file_name.split("ATL03_")[-1].split("_")[0]
    
#     for f in os.listdir(folder):
#         # Check if the common part (suffix) matches any file
#         if f.split("ATL03_")[-1].split("_")[0] == suffix:
#             return os.path.join(folder, f)
#     return None

    

# def project_to_utm(df, EPSG_code):
#     """Project lat/lon (ITRF2014) to UTM coordinates."""
#     itrf = pyproj.CRS("EPSG:7912")  # ITRF2014
#     utm_crs = pyproj.CRS(f"EPSG:{EPSG_code}")  # EPSG code
    
#     print(utm_crs)
    
#     transformer = pyproj.Transformer.from_crs(itrf, utm_crs, always_xy=True)
    
#     print("Dataframe columns:", df.columns)
    
#     # Check if the columns exist and select the right ones
#     if 'Longitude' in df.columns and 'Latitude' in df.columns:
#         lon_col = 'Longitude'
#         lat_col = 'Latitude'
#     elif 'reference_photon_lon' in df.columns and 'reference_photon_lat' in df.columns:
#         lon_col = 'reference_photon_lon'
#         lat_col = 'reference_photon_lat'
#     else:
#         raise ValueError("Neither 'Longitude'/'Latitude' nor 'reference_photon_lon'/'reference_photon_lat' columns found.")
    
#     # Perform the transformation
#     df['Easting'], df['Northing'] = transformer.transform(df[lon_col], df[lat_col])
#     return df


# def process_files(folder1, folder2, utm_csv):
#     """Process CSVs, project coordinates, and save outputs."""
#     utm_zones = pd.read_csv(utm_csv, dtype={'Name': str, 'UTM Zone': int, 'EPSG code': int})
    
#     for file1 in os.listdir(folder1):
#         if file1.endswith(".csv"):
#             file1_path = os.path.join(folder1, file1)
#             file2_path = find_matching_file(file1, folder2)
            
#             if file2_path is None:
#                 print(f"No matching file found for {file1}")
#                 continue
            
#             # Directly search for the full file name (without modification)
#             file_name = file1.split(".")[0]  # Remove extension if needed
#             epsg_code = utm_zones.loc[utm_zones['Name'] == file_name, 'EPSG code'].values
             
            
#             if len(epsg_code) == 0:
#                 print(f"No EPSG code found for {file_name}")
#                 continue
#             epsg_code = epsg_code[0]
            
           
#             df1 = pd.read_csv(file1_path)
#             df2 = pd.read_csv(file2_path)
            
#             df1 = project_to_utm(df1, epsg_code)
#             df2 = project_to_utm(df2, epsg_code)
            
#             output1 = file1_path.replace(".csv", "_utm.csv")
#             output2 = file2_path.replace(".csv", "_utm.csv")
#             df1.to_csv(output1, index=False)
#             df2.to_csv(output2, index=False)
#             print(f"Saved: \n{output1}\n{output2}")
#         else:
#             # Skip non-CSV files (e.g., .h5)
#             print(f"Skipping non-CSV file: {file1}")

# # Example usage
# folder1 = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\orig_icesat\other"    # original IceSat track
# folder2 = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\new_icesat\other"    # new Icesat track
# utm_csv = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\UTM_epsg_codes.csv"
# process_files(folder1, folder2, utm_csv)


##############################################################################################################
##############################################################################################################

# Filters out everything except the bathy points (code 40)

# import numpy as np
# import h5py
# import os
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point
# import pyproj

# def filter_manual_label(folder):
#     """Filter rows with Manual_Label value of 40 from all CSV files in a folder."""
#     for file in os.listdir(folder):
#         if file.endswith("_utm.csv"):
#             file_path = os.path.join(folder, file)
#             # Read the CSV file into a DataFrame
#             df = pd.read_csv(file_path)
            
#             # Filter rows where 'Manual_Label' is 40
#             df_filtered = df[df['Manual_Label'] == 40]
            
#             # Save the filtered DataFrame to a new CSV file
#             output_file = file_path.replace(".csv", "_filtered.csv")
#             df_filtered.to_csv(output_file, index=False)
#             print(f"Saved filtered data to: {output_file}")
#         else:
#             # Skip non-CSV files
#             print(f"Skipping non-CSV file: {file}")

# # Example usage
# folder = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\orig_icesat\other"  # Replace with your folder path
# filter_manual_label(folder)


##############################################################################################################
##############################################################################################################
# Outputs a csv that has the lat, long, and ortho height from the original icesat track and adds the ref_elev
# and ref_azimuth from the nearest newer ICESat track

# import numpy as np
# import h5py
# import os
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point


# # Function to find matching files based on the prefix
# def find_matching_file(folder, prefix):
#     for f in os.listdir(folder):
#         if f.startswith(prefix):
#             return os.path.join(folder, f)
#     return None

# def process_folder(orig_folder, new_folder, output_folder):
#     for orig_file in os.listdir(orig_folder):
#         if orig_file.endswith("_filtered.csv"):
#             orig_file_path = os.path.join(orig_folder, orig_file)

#             # Extract prefix (everything before the first underscore)
#             orig_prefix = orig_file.split('_')[0]
#             print(f"Processing orig file: {orig_file} with prefix: {orig_prefix}")

#             # Find the matching file in the new folder
#             matching_new_file_path = find_matching_file(new_folder, orig_prefix)

#             if matching_new_file_path:
#                 print(f"Found matching new file: {matching_new_file_path}")

#                 # Read the original and new track data
#                 orig_track = pd.read_csv(orig_file_path)
#                 new_track = pd.read_csv(matching_new_file_path)

#                 # Convert the data into GeoDataFrames (assuming lat, lon columns)
#                 orig_track['geometry'] = [Point(lon, lat) for lon, lat in zip(orig_track['Longitude'], orig_track['Latitude'])]
#                 new_track['geometry'] = [Point(lon, lat) for lon, lat in zip(new_track['reference_photon_lon'], new_track['reference_photon_lat'])]

#                 gdf_orig = gpd.GeoDataFrame(orig_track, geometry='geometry')
#                 gdf_new = gpd.GeoDataFrame(new_track, geometry='geometry')

#                 # Set CRS
#                 gdf_orig = gdf_orig.set_crs("EPSG:4326", allow_override=True)
#                 gdf_new = gdf_new.set_crs("EPSG:4326", allow_override=True)

#                 # Perform spatial join
#                 joined = gpd.sjoin_nearest(gdf_orig, gdf_new, how="left", distance_col="distance")

#                 # Apply distance filter (e.g., 10 meters)
#                 joined_filtered = joined[joined['distance'] <= 10]

#                 # Drop unnecessary columns
#                 joined_filtered = joined_filtered.drop(columns=["distance", "geometry"])

#                 # Define output file path
#                 output_path = os.path.join(output_folder, f"{orig_file.split('.')[0]}_merged.csv")

#                 # Save to CSV
#                 joined_filtered.to_csv(output_path, index=False)
#                 print(f"Saved merged file: {output_path}")

#             else:
#                 print(f"No matching file found for {orig_file}")

# # Define input and output folders
# orig_folder = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\orig_icesat\other"
# new_folder = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\new_icesat\other"
# output_folder = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\merged_tracks\other"

# # Run the script
# process_folder(orig_folder, new_folder, output_folder)

##############################################################################################################
##############################################################################################################

# Finds average sea surface height 

# import numpy as np
# import h5py
# import os
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point

# def process_folder(input_folder, output_file):
#     results = []

#     # Loop through files in the input folder
#     for file in os.listdir(input_folder):
#         if file.endswith("_new.csv"):  # Only process files ending in "_new.csv"
#             file_path = os.path.join(input_folder, file)
#             try:
#                 # Read the CSV file
#                 df = pd.read_csv(file_path)

#                 # Filter rows where Manual_Label is 41
#                 filtered_df = df[df['Manual_Label'] == 41]

#                 # Calculate the mean of Geoid_Corrected_Ortho_Height
#                 average_height = filtered_df['Geoid_Corrected_Ortho_Height'].mean()

#                 # Append results (filename, mean value)
#                 results.append([file, average_height])

#                 print(f"Processed: {file}, Mean Height: {average_height}")

#             except Exception as e:
#                 print(f"Error processing {file}: {e}")

#     # Create DataFrame and save to CSV
#     results_df = pd.DataFrame(results, columns=["Filename", "Mean_Geoid_Height"])
#     results_df.to_csv(output_file, index=False)

#     print(f"\nProcessing complete. Results saved to {output_file}")

# # Example usage
# input_folder = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\orig_icesat\other"
# output_file = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\geoid_sea_surface_means.csv"

# process_folder(input_folder, output_file)



##############################################################################################################
##############################################################################################################
# Actual Refraction correction


 # ICESat-2 refraction correction implented as outlined in Parrish, et al. 
 # 2019 for correcting photon depth data.

 # Highly recommended to reference elevations to geoid datum to remove sea
 # surface variations.

 # https://www.mdpi.com/2072-4292/11/14/1634

 # Code Author: 
 # Jonathan Markel
 # Graduate Research Assistant
 # 3D Geospatial Laboratory
 # The University of Texas at Austin
 # jonathanmarkel@gmail.com

 # Parameters
 # ----------
 # W : float, or nx1 array of float
 #     Elevation of the water surface.

 # Z : nx1 array of float
 #     Elevation of seabed photon data. Highly recommend use of geoid heights.

 # ref_az : nx1 array of float
 #     Photon-rate reference photon azimuth data. Should be pulled from ATL03
 #     data parameter 'ref_azimuth'. Must be same size as seabed Z array.

 # ref_el : nx1 array of float
 #     Photon-rate reference photon azimuth data. Should be pulled from ATL03
 #     data parameter 'ref_elev'. Must be same size as seabed Z array.

 # n1 : float, optional
 #     Refractive index of air. The default is 1.00029.

 # n2 : float, optional
 #     Refractive index of water. Recommended to use 1.34116 for saltwater 
 #     and 1.33469 for freshwater. The default is 1.34116.

 # Returns
 # -------
 # dE : nx1 array of float
 #     Easting offset of seabed photons.

 # dN : nx1 array of float
 #     Northing offset of seabed photons.

 # dZ : nx1 array of float
 #     Vertical offset of seabed photons.


# import numpy as np
# import h5py
# import os
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point

# def photon_refraction(W, Z, ref_az, ref_el, n1=1.00029, n2=1.34116):
#     D = W - Z
#     theta_1 = (np.pi / 2) - ref_el
#     theta_2 = np.arcsin(n1 * np.sin(theta_1) / n2)
#     phi = theta_1 - theta_2
#     S = D / np.cos(theta_1)
#     R = S * n1 / n2
#     P = np.sqrt(R**2 + S**2 - 2*R*S*np.cos(theta_1 - theta_2))
#     gamma = (np.pi / 2) - theta_1
#     alpha = np.arcsin(R * np.sin(phi) / P)
#     beta = gamma - alpha
#     dY = P * np.cos(beta)
#     dZ = P * np.sin(beta)
#     kappa = ref_az
#     dE = dY * np.sin(kappa)
#     dN = dY * np.cos(kappa)
#     return dE, dN, dZ


# def extract_base_name(Filename):
#     """Extracts everything before the first underscore in a filename."""
#     return Filename.split("_")[0]



# def process_refraction(input_folder, water_levels_csv):

#     # Read water level reference CSV
#     water_levels_df = pd.read_csv(water_levels_csv)
#     water_levels_df["base_name"] = water_levels_df["Filename"].apply(extract_base_name)  # Ensure matching
    
#     for Filename in os.listdir(input_folder):
#         if Filename.endswith("_merged.csv"):  
#             input_path = os.path.join(input_folder, Filename)
            
#             # Read input CSV
#             df = pd.read_csv(input_path)

#             # Extract base name
#             base_name = extract_base_name(Filename)

#             # Find matching row in water levels
#             W_row = water_levels_df[water_levels_df["base_name"] == base_name]
#             if W_row.empty:
#                 print(f"No matching water level for {Filename}, skipping.")
#                 continue
            
#             W = W_row["Mean_Geoid_Height"].values[0]  

#             # Extract required columns
#             Z = df["Geoid_Corrected_Ortho_Height"].values  
#             ref_az = df["ref_azimuth"].values  
#             ref_el = df["ref_elev"].values  

#             # Apply refraction correction
#             dE, dN, dZ = photon_refraction(W, Z, ref_az, ref_el)

#             # Add results to DataFrame
#             df["dE"] = dE
#             df["dN"] = dN
#             df["dZ"] = dZ

#             # Select required columns for output
#             output_df = df[["Easting", "Northing", "Geoid_Corrected_Ortho_Height", "dE", "dN", "dZ"]]

#             # Define output file path (same folder, modified filename)
#             output_filename = f"{base_name}_corrections.csv"
#             output_path = os.path.join(input_folder, output_filename)

#             # Save the corrected data
#             output_df.to_csv(output_path, index=False)

#             print(f"Processed {Filename} -> Saved as {output_filename}")

# # Define paths
# input_folder = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\merged_tracks\other"
# water_levels_csv = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\geoid_sea_surface_means.csv"

# # Run the processing function
# process_refraction(input_folder, water_levels_csv)


##############################################################################################################
##############################################################################################################

# Apply refraction corrections to icesat data

import numpy as np
import h5py
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Input folder containing the correction CSV files
input_folder = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\merged_tracks\other"

# Output folder for corrected CSV files
output_folder = r"E:\Thesis Stuff\ReferenceData\Refraction Correction\Refraction_Corrected_Photons"  

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all CSV files in the input folder ending with '_corrections.csv'
for filename in os.listdir(input_folder):
    if filename.endswith("_corrections.csv"):
        # Construct the full input file path
        input_csv = os.path.join(input_folder, filename)
        
        # Read the input CSV file into a DataFrame
        df = pd.read_csv(input_csv)
        
        # Add corrections to the easting, northing, and height values
        df['Easting'] = df['Easting'] + df['dE']
        df['Northing'] = df['Northing'] + df['dN']
        df['Geoid_Corrected_Ortho_Height'] = df['Geoid_Corrected_Ortho_Height'] + df['dZ']
        
        # Select only the corrected columns
        output_df = df[['Easting', 'Northing', 'Geoid_Corrected_Ortho_Height']]
        
        # Construct the output file name by replacing '_corrections' with '_corrected'
        output_filename = filename.replace('_corrections', '_corrected')
        output_csv = os.path.join(output_folder, output_filename)
        
        # Save the corrected values to a new CSV file
        output_df.to_csv(output_csv, index=False)
        
        print(f"Corrected file saved: {output_csv}")




















