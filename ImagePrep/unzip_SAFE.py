# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 11:01:40 2025

@author: mccullru
"""

import os
import zipfile

def unzip_safe_files(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".SAFE.zip"):
                zip_path = os.path.join(subdir, file)
                safe_folder_name = file.replace(".zip", "")
                safe_extract_path = os.path.join(subdir, safe_folder_name)

                # Skip if already unzipped correctly
                if os.path.isdir(safe_extract_path) and "manifest.safe" in os.listdir(safe_extract_path):
                    print(f"Already extracted: {safe_folder_name}")
                    continue

                try:
                    print(f"Unzipping: {zip_path}")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        # List contents of zip
                        members = zip_ref.namelist()

                        # If the zip already contains a .SAFE folder, extract to parent
                        if members[0].endswith(".SAFE/") or ".SAFE/" in members[0]:
                            zip_ref.extractall(subdir)
                        else:
                            # If zip doesn't contain .SAFE/ as top-level, extract normally
                            zip_ref.extractall(safe_extract_path)

                    print(f"Extracted to: {subdir}")
                except Exception as e:
                    print(f"Failed to unzip {zip_path}: {e}")

# Example usage
unzip_safe_files(r"E:\Thesis Stuff\AcoliteWithPython\Uncorrected_Imagery\All_Sentinel2")