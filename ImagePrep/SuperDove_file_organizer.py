# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 18:07:49 2025

@author: mccullru
"""


import zipfile
import os
import shutil


"""
Organizes all the SuperDove files that are downloaded from NASA CSDA site. Need them to have following order:
    AOI directory -> folders of all img ID -> tiff and xml file
    
Will unzip and organize all files/folders in order to be read by Acolite

"""

# zip_folder = r"E:\Thesis Stuff\AcoliteWithPython\Uncorrected_Imagery\All_SuperDove\additional imagery\BumBum"
# output_folder = r"E:\Thesis Stuff\AcoliteWithPython\Uncorrected_Imagery\All_SuperDove\additional imagery\BumBum"
# target_folder = "planet/"  # Relative path inside the zip



# # Find all .zip files in the folder
# zip_files = [f for f in os.listdir(zip_folder) if f.lower().endswith(".zip")]

# if not zip_files:
#     print("No ZIP files found in the folder.")
# else:
#     print(f"Found {len(zip_files)} ZIP file(s). Starting extraction...\n")

#     # First loop to extract ZIP files
#     for zip_file in zip_files:
#         zip_path = os.path.join(zip_folder, zip_file)
#         print(f"Processing: {zip_file}")

#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             members = [m for m in zip_ref.namelist() if m.startswith(target_folder)]

#             if not members:
#                 print(f"  No '{target_folder}' folder found in {zip_file}")
#                 continue

#             for member in members:
#                 filename = os.path.relpath(member, target_folder)
#                 target_path = os.path.join(output_folder, filename)

#                 os.makedirs(os.path.dirname(target_path), exist_ok=True)

#                 if not member.endswith('/'):
#                     print(f"  Extracting: {filename} -> {target_path}")
#                     with open(target_path, 'wb') as f:
#                         f.write(zip_ref.read(member))

#     print("All extractions complete.\n")

#     # Second loop to move files and clean up
#     print("Starting to organize files...\n")

#     for root, dirs, files in os.walk(output_folder):
#         # Look for PSScene folders
#         for dir_name in dirs:
#             if dir_name.startswith('PSScene'):
#                 ps_scene_folder = os.path.join(root, dir_name)
#                 assets_folder = os.path.join(ps_scene_folder, 'assets')

#                 # Check if assets folder exists
#                 if os.path.exists(assets_folder):
#                     print(f"  Found assets folder: {assets_folder}")
                    
#                     # Inspect all subfolders and try to move .tif and .xml files
#                     for subfolder in os.listdir(assets_folder):
#                         subfolder_path = os.path.join(assets_folder, subfolder)
#                         if os.path.isdir(subfolder_path):
#                             print(f"    Inspecting subfolder: {subfolder_path}")

#                             # Look for .tif and .xml files in the subfolder
#                             for file in os.listdir(subfolder_path):
#                                 if file.endswith('.tif') or file.endswith('.xml'):
#                                     source_path = os.path.join(subfolder_path, file)
#                                     target_path = os.path.join(ps_scene_folder, file)

#                                     # Move if the file doesn't exist already
#                                     if not os.path.exists(target_path):
#                                         shutil.move(source_path, target_path)
#                                         print(f"    Moved: {file} from {source_path} to {target_path}")
#                                     else:
#                                         print(f"    File already exists: {file}, skipping move.")

#                     # After moving files, delete the 'assets' folder if empty
#                     try:
#                         shutil.rmtree(assets_folder)  # Remove the 'assets' folder
#                         print(f"  Deleted 'assets' folder: {assets_folder}")
#                     except Exception as e:
#                         print(f"    Error deleting 'assets' folder: {e}")
#                 else:
#                     print(f"  No 'assets' folder found in {ps_scene_folder}")

#     print("File organization and cleanup complete.")
                  

        
###############################################################################################################
###############################################################################################################

""" organizes the files if they are already unzipped """

                        
    # --- Configuration ---
# Set this to the base folder where your PSScene folders are located.
# For example, if you have E:\Thesis Stuff\AcoliteWithPython\Uncorrected_Imagery\All_SuperDove\additional imagery\BumBum\PSScene_12345,
# then base_folder should be E:\Thesis Stuff\AcoliteWithPython\Uncorrected_Imagery\All_SuperDove\additional imagery\BumBum
base_folder = r"E:\Thesis Stuff\AcoliteWithPython\Uncorrected_Imagery\All_SuperDove\additional imagery\Gyali"


print(f"Starting file organization in: {base_folder}\n")

# Use os.walk to traverse the directory structure
for root, dirs, files in os.walk(base_folder):
    # We are looking for folders that start with 'PSScene'
    # By modifying `dirs` in place, we can tell os.walk not to descend into specific subdirectories
    # to avoid redundant processing, but in this case, we need to go into 'assets'.
    
    # Iterate over a copy of dirs to allow modification during iteration if needed,
    # though os.walk handles dir traversal well.
    for dir_name in list(dirs): # Iterate over a copy to allow removal
        if dir_name.startswith('PSScene'):
            ps_scene_folder = os.path.join(root, dir_name)
            assets_folder = os.path.join(ps_scene_folder, 'assets')

            # Check if assets folder exists within the PSScene folder
            if os.path.exists(assets_folder) and os.path.isdir(assets_folder):
                print(f"  Found assets folder: {assets_folder}")
                
                # List to keep track of any files that couldn't be moved in this PSScene folder
                failed_moves_in_psscene = []

                # Iterate through items inside the 'assets' folder
                # These should be 'ortho_analytic_8b' and 'ortho_analytic_8b_xml'
                for sub_item_name in os.listdir(assets_folder):
                    sub_item_path = os.path.join(assets_folder, sub_item_name)

                    if os.path.isdir(sub_item_path):
                        # This is expected to be 'ortho_analytic_8b' or 'ortho_analytic_8b_xml'
                        print(f"    Inspecting sub-directory: {sub_item_path}")

                        # Look for .tif and .xml files directly inside these sub-directories
                        for file_name in os.listdir(sub_item_path):
                            if file_name.endswith('.tif') or file_name.endswith('.xml'):
                                source_path = os.path.join(sub_item_path, file_name)
                                target_path = os.path.join(ps_scene_folder, file_name)

                                try:
                                    if not os.path.exists(target_path):
                                        shutil.move(source_path, target_path)
                                        print(f"      Moved: '{file_name}' to '{ps_scene_folder}'")
                                    else:
                                        print(f"      File '{file_name}' already exists in '{ps_scene_folder}', skipping move.")
                                        # Optionally, you could delete the source file if it's a duplicate and you want to ensure cleanup
                                        # os.remove(source_path)
                                        # print(f"      Deleted duplicate source file: {source_path}")
                                except Exception as e:
                                    print(f"      ERROR: Could not move '{file_name}' from '{source_path}' to '{target_path}': {e}")
                                    failed_moves_in_psscene.append(source_path)
                            else:
                                print(f"      Skipping non-tif/xml file: {file_name} in {sub_item_path}")
                    elif os.path.isfile(sub_item_path):
                        print(f"    Skipping unexpected file in assets folder: {sub_item_path}")
                
                # --- Cleanup 'assets' folder and its content ---
                # After attempting to move all files from assets' subfolders,
                # remove the entire 'assets' folder and its (now empty) subfolders.
                try:
                    # shutil.rmtree will remove the directory and all its contents (even if not empty).
                    # This is more robust than os.rmdir which only removes empty directories.
                    shutil.rmtree(assets_folder)
                    print(f"  Deleted 'assets' folder and its content: {assets_folder}")
                except Exception as e:
                    print(f"  ERROR: Could not delete 'assets' folder '{assets_folder}': {e}")
                    failed_moves_in_psscene.append(assets_folder) # Mark assets folder as failed if it can't be deleted

                if failed_moves_in_psscene:
                    print(f"\n  Summary for {ps_scene_folder}:")
                    print(f"  {len(failed_moves_in_psscene)} item(s) could not be moved or deleted:")
                    for item in failed_moves_in_psscene:
                        print(f"    - {item}")
            else:
                print(f"  No 'assets' folder found or it's not a directory in {ps_scene_folder}. Skipping.")

print("\n--- File organization and cleanup complete. ---")
    
    
    
    
    