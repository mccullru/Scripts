# -*- coding: utf-8 -*-
"""
Created on Mon May 26 10:13:28 2025

@author: mccullru
"""


""" Deletes all files within the main workspace folders 
    NOTE: The RGB files folder also has a separate deletion function in SDB_Time. Check that if funny business occurs
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

    Returns:
        list: A list of file/folder paths that could not be deleted.
    """
    if not isinstance(folder_list, list):
        print("Error: Please provide a list of folder paths.")
        return []

    all_failed_deletions = []

    for folder_path in folder_list:
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found or is not a directory: {folder_path}. Skipping.")
            all_failed_deletions.append(folder_path) # Mark folder itself as failed if not found
            continue

        print(f"\nProcessing folder: {folder_path}")
        item_count = 0
        deleted_files_count = 0
        folder_errors_encountered = [] 

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
                    folder_errors_encountered.append(item_path)
            elif os.path.isdir(item_path) and delete_subfolders_content:
                print(f"  Entering subfolder: {item_path}")
                
                # The recursive function now returns a list of its own failed deletions
                sub_deleted_count, sub_failed_items = delete_files_in_folder_recursive(item_path, delete_empty_subfolders)
                deleted_files_count += sub_deleted_count
                folder_errors_encountered.extend(sub_failed_items) # Add subfolder failures

                if delete_empty_subfolders:
                    try:
                        if not os.listdir(item_path): # Check if subfolder is now empty
                            os.rmdir(item_path) # os.rmdir only removes empty directories
                            print(f"  Deleted empty subfolder: {item_path}")
                        else:
                            print(f"  Subfolder not empty, not deleted: {item_path}")
                            # If we tried to delete and it wasn't empty, that's fine, not an error
                    except Exception as e:
                        print(f"  Error deleting subfolder {item_path}: {e}")
                        folder_errors_encountered.append(item_path)
            elif os.path.isdir(item_path):
                print(f"  Skipping subfolder (delete_subfolders_content is False): {item_path}")
            item_count +=1

        if item_count == 0:
            print(f"Folder {folder_path} was empty.")
        else:
            print(f"Finished processing {folder_path}. Deleted {deleted_files_count} files. Encountered {len(folder_errors_encountered)} errors.")
            if folder_errors_encountered:
                print(f"  Failed deletions in {folder_path}: {len(folder_errors_encountered)} items.")
                

        all_failed_deletions.extend(folder_errors_encountered) # Accumulate errors from this folder

    return all_failed_deletions

def delete_files_in_folder_recursive(folder_path, delete_empty_subfolders_flag):

    deleted_count = 0
    failed_items_in_recursion = []

    # Get a list of items to iterate over. Using a copy to avoid issues if items are deleted.
    items_in_folder = list(os.listdir(folder_path)) 

    for item_name in items_in_folder:
        item_path = os.path.join(folder_path, item_name)

        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
                deleted_count += 1
            
            except Exception as e:
                print(f"    Error deleting file (in subfolder) {item_path}: {e}")
                failed_items_in_recursion.append(item_path)
        elif os.path.isdir(item_path):
           
            # Recursive call for sub-subfolders etc.
            sub_deleted, sub_failed = delete_files_in_folder_recursive(item_path, delete_empty_subfolders_flag)
            deleted_count += sub_deleted
            failed_items_in_recursion.extend(sub_failed)

            if delete_empty_subfolders_flag:
                try:
                    
                    # Check if subfolder is now empty *after* recursive deletion
                    if not os.listdir(item_path):
                        os.rmdir(item_path)
                        print(f"    Deleted empty subfolder: {item_path}")
                
                except OSError as e: # Catch OSError specifically for rmdir issues
                    print(f"    Error deleting subfolder {item_path}: {e}")
                    failed_items_in_recursion.append(item_path)
                except Exception as e: # Catch any other unexpected errors
                    print(f"    Unexpected error deleting subfolder {item_path}: {e}")
                    failed_items_in_recursion.append(item_path)

    return deleted_count, failed_items_in_recursion


if __name__ == "__main__":
    
    # !! DEFINE YOUR FOLDER PATHS HERE !!

    folders_to_clear = [
        
        ## SDB_Time
        r"E:\Thesis Stuff\pSDB",
        r"E:\Thesis Stuff\pSDB_ExtractedPts",
        r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results",
        r"E:\Thesis Stuff\RGBCompositOutput",
        r"E:\Thesis Stuff\SDB",
        r"E:\Thesis Stuff\SDB_ExtractedPts",
        r"E:\Thesis Stuff\SDB_ExtractedPts_maxR2_results",
        r"E:\Thesis Stuff\pSDB_ExtractedPts_maxR2_results\Flagged Results",
        
        ## Figures
        r"E:\Thesis Stuff\Figures\Heatmap_Plots",
        r"E:\Thesis Stuff\Figures\Individual_Histograms",
        r"E:\Thesis Stuff\Figures\KDE_Plots",
        r"E:\Thesis Stuff\Figures\Average_Depth_Ranges",
        r"E:\Thesis Stuff\Figures\Summary_Stats",
        r"E:\Thesis Stuff\Figures",
        
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
        process_subfolders_content = False 
        
        # Set to True if you want to delete empty subfolders 
        remove_empty_subfolders = False 


        confirmation = input("Are you absolutely sure you want to proceed? (yes/no): ").lower()

        if confirmation == 'yes':
            print("Proceeding with file deletion...")
            
            # Capture the list of failed deletions
            unsuccessful_deletions = delete_files_in_folders(folders_to_clear, process_subfolders_content, remove_empty_subfolders)
            print("\n--- Deletion process finished. ---")

            if unsuccessful_deletions:
                print("\n--- Items that could NOT be deleted: ---")
                for item_path in unsuccessful_deletions:
                    print(f" - {item_path}")
                print(f"\n!!! Total items that could not be deleted: {len(unsuccessful_deletions)} !!!")
            else:
                print("\nAll specified files and folders (or their content) were successfully deleted.")
        else:
            print("File deletion cancelled by the user.")
            
            
            
            
            
            
            
            
            