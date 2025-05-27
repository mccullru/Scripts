# -*- coding: utf-8 -*-
"""
Created on Mon May 26 10:13:28 2025

@author: mccullru
"""


""" Deletes all files within the main workspace folders 
    NOTE: The RGB files folder also has a separate deletion function in SDB_Time
"""


import os

def delete_files_in_folders(folder_list, delete_subfolders_content=False, delete_empty_subfolders=False):
    """
    Deletes all files within a list of specified folders.
    Optionally, it can also delete files within subfolders and the empty subfolders themselves.

    Args:
        folder_list (list): A list of strings, where each string is a path to a folder.
        delete_subfolders_content (bool): If True, also delete files within immediate subfolders.
                                          If False (default), only files in the top-level
                                          of each specified folder are deleted.
        delete_empty_subfolders (bool): If True (and delete_subfolders_content is True),
                                        also deletes subfolders after their content is removed.
                                        This does NOT delete the top-level specified folders.
    """
    if not isinstance(folder_list, list):
        print("Error: Please provide a list of folder paths.")
        return

    for folder_path in folder_list:
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found or is not a directory: {folder_path}. Skipping.")
            continue

        print(f"\nProcessing folder: {folder_path}")
        item_count = 0
        deleted_files_count = 0
        error_count = 0

        # Iterate through items in the current folder_path
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)

            if os.path.isfile(item_path):
                try:
                    os.remove(item_path)
                    print(f"  Deleted file: {item_path}")
                    deleted_files_count += 1
                except Exception as e:
                    print(f"  Error deleting file {item_path}: {e}")
                    error_count += 1
            elif os.path.isdir(item_path) and delete_subfolders_content:
                print(f"  Entering subfolder: {item_path}")
                sub_deleted_count, sub_error_count = delete_files_in_folder_recursive(item_path, delete_empty_subfolders)
                deleted_files_count += sub_deleted_count
                error_count += sub_error_count
                if delete_empty_subfolders:
                    try:
                        if not os.listdir(item_path): # Check if subfolder is now empty
                            os.rmdir(item_path) # os.rmdir only removes empty directories
                            print(f"  Deleted empty subfolder: {item_path}")
                        else:
                            print(f"  Subfolder not empty, not deleted: {item_path}")
                    except Exception as e:
                        print(f"  Error deleting subfolder {item_path}: {e}")
                        error_count += 1
            elif os.path.isdir(item_path):
                print(f"  Skipping subfolder (delete_subfolders_content is False): {item_path}")
            item_count +=1

        if item_count == 0:
            print(f"Folder {folder_path} was empty.")
        else:
            print(f"Finished processing {folder_path}. Deleted {deleted_files_count} files. Encountered {error_count} errors.")

def delete_files_in_folder_recursive(folder_path, delete_empty_subfolders_flag):
    """
    Helper function to recursively delete files in a folder and its subfolders.

    Args:
        folder_path (str): Path to the folder.
        delete_empty_subfolders_flag (bool): If True, delete subfolders if they become empty.

    Returns:
        tuple: (number_of_deleted_files, number_of_errors)
    """
    deleted_count = 0
    error_count = 0
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)
        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
                #print(f"    Deleted file (in subfolder): {item_path}")
                deleted_count += 1
            except Exception as e:
                print(f"    Error deleting file (in subfolder) {item_path}: {e}")
                error_count += 1
        elif os.path.isdir(item_path):
            # Recursive call for sub-subfolders etc.
            sub_deleted, sub_error = delete_files_in_folder_recursive(item_path, delete_empty_subfolders_flag)
            deleted_count += sub_deleted
            error_count += sub_error
            if delete_empty_subfolders_flag:
                try:
                    if not os.listdir(item_path):
                        os.rmdir(item_path)
                        print(f"    Deleted empty subfolder: {item_path}")
                    else:
                        print(f"    Subfolder not empty, not deleted: {item_path}")
                except Exception as e:
                    print(f"    Error deleting subfolder {item_path}: {e}")
                    error_count += 1
    return deleted_count, error_count


# --- How to use the function ---
if __name__ == "__main__":
    
    # !! CRITICAL: DEFINE YOUR FOLDER PATHS HERE !!

    folders_to_clear = [
        r"E:\Thesis Stuff\pSDB",
        r"E:\Thesis Stuff\pSDB_ExtractedPts",
        r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results",
        r"E:\Thesis Stuff\pSDB_ExtractedPts_Results",
        r"E:\Thesis Stuff\RGBCompositOutput",
        r"E:\Thesis Stuff\RGBCompositOutput_ODWbinarymasks",
        r"E:\Thesis Stuff\SDB",
        r"E:\Thesis Stuff\SDB_ExtractedPts",
        r"E:\Thesis Stuff\SDB_ExtractedPts_Results",
        r"E:\Thesis Stuff\SDB_ExtractedPts_maxR2_results"
    ]

    if not folders_to_clear:
        print("Warning: The 'folders_to_clear' list is empty. No folders will be processed.")
        print("Please edit the script to specify the folders you want to clear.")
    else:
        # Confirm with the user before proceeding
        print("WARNING: This script will permanently delete files from the specified folders.")
        print("Folders to be processed:")
        for folder in folders_to_clear:
            print(f" - {folder}")

        # Set to True if you want to also delete files within immediate subfolders
        process_subfolders_content = False # Default: only top-level files in listed folders
        # Set to True if you want to delete empty subfolders (only if process_subfolders_content is True)
        remove_empty_subfolders = False # Default: do not remove subfolders

        # Example options:
        # To delete files in top-level and also in immediate subfolders, and then remove empty subfolders:
        # process_subfolders_content = True
        # remove_empty_subfolders = True

        # To delete only files in the top-level of the listed folders:
        # process_subfolders_content = False
        # remove_empty_subfolders = False (this setting won't matter if content isn't processed)


        confirmation = input("Are you absolutely sure you want to proceed? (yes/no): ").lower()

        if confirmation == 'yes':
            print("Proceeding with file deletion...")
            delete_files_in_folders(folders_to_clear, process_subfolders_content, remove_empty_subfolders)
            print("\n--- Deletion process finished. ---")
        else:
            print("File deletion cancelled by the user.")
            
            
            
            
            
            
            
            
            
            