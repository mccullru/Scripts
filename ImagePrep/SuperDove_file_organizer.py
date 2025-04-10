# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 18:07:49 2025

@author: mccullru
"""

"""
Organizes all the SuperDove files that are downloaded from NASA CSDA site. Need them to have following order:
    AOI directory -> folders of all img ID -> tiff and xml file
    
Will unzip and organize all files/folders in order to be read by Acolite

"""

import zipfile
import os
import shutil

zip_folder = r"E:\Thesis Stuff\AcoliteWithPython\Uncorrected_Imagery\SD_SouthPort"
output_folder = r"E:\Thesis Stuff\AcoliteWithPython\Uncorrected_Imagery\SD_SouthPort"
target_folder = "planet/"  # Relative path inside the zip



# Find all .zip files in the folder
zip_files = [f for f in os.listdir(zip_folder) if f.lower().endswith(".zip")]

if not zip_files:
    print("No ZIP files found in the folder.")
else:
    print(f"Found {len(zip_files)} ZIP file(s). Starting extraction...\n")

    # First loop to extract ZIP files
    for zip_file in zip_files:
        zip_path = os.path.join(zip_folder, zip_file)
        print(f"Processing: {zip_file}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = [m for m in zip_ref.namelist() if m.startswith(target_folder)]

            if not members:
                print(f"  No '{target_folder}' folder found in {zip_file}")
                continue

            for member in members:
                filename = os.path.relpath(member, target_folder)
                target_path = os.path.join(output_folder, filename)

                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                if not member.endswith('/'):
                    print(f"  Extracting: {filename} -> {target_path}")
                    with open(target_path, 'wb') as f:
                        f.write(zip_ref.read(member))

    print("All extractions complete.\n")

    # Second loop to move files and clean up
    print("Starting to organize files...\n")

    for root, dirs, files in os.walk(output_folder):
        # Look for PSScene folders
        for dir_name in dirs:
            if dir_name.startswith('PSScene'):
                ps_scene_folder = os.path.join(root, dir_name)
                assets_folder = os.path.join(ps_scene_folder, 'assets')

                # Check if assets folder exists
                if os.path.exists(assets_folder):
                    print(f"  Found assets folder: {assets_folder}")
                    
                    # Inspect all subfolders and try to move .tif and .xml files
                    for subfolder in os.listdir(assets_folder):
                        subfolder_path = os.path.join(assets_folder, subfolder)
                        if os.path.isdir(subfolder_path):
                            print(f"    Inspecting subfolder: {subfolder_path}")

                            # Look for .tif and .xml files in the subfolder
                            for file in os.listdir(subfolder_path):
                                if file.endswith('.tif') or file.endswith('.xml'):
                                    source_path = os.path.join(subfolder_path, file)
                                    target_path = os.path.join(ps_scene_folder, file)

                                    # Move if the file doesn't exist already
                                    if not os.path.exists(target_path):
                                        shutil.move(source_path, target_path)
                                        print(f"    Moved: {file} from {source_path} to {target_path}")
                                    else:
                                        print(f"    File already exists: {file}, skipping move.")

                    # After moving files, delete the 'assets' folder if empty
                    try:
                        shutil.rmtree(assets_folder)  # Remove the 'assets' folder
                        print(f"  Deleted 'assets' folder: {assets_folder}")
                    except Exception as e:
                        print(f"    Error deleting 'assets' folder: {e}")
                else:
                    print(f"  No 'assets' folder found in {ps_scene_folder}")

    print("File organization and cleanup complete.")
                        
                        
    
    
    
    
    
    