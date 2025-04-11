
"""

This code performs all the atmospheric correction for both SD and S2 images in the correct format (SAFE for
Sentinel and PSSscene for SuperDove).                                                                                                    
                                                                                                  


"""


import os
from pathlib import Path
import sys
import re
import zipfile
import shutil



##############################################################################################################
##############################################################################################################
"""
unzips PlanetScope Folders and renames the subfolder the same name as the zipped file so they don't overwrite 
each other with the same name

"""

# def extract_and_rename(zip_folder):
#     """
#     Extracts all zip files in a folder, renames the extracted subfolders
#     to match their corresponding zip file names, and keeps them in the same directory.

#     Args:
#         zip_folder (str): Path to the folder containing zip files.

#     Returns:
#         None
#     """
#     for file in os.listdir(zip_folder):
#         if file.endswith(".zip"):
#             zip_path = os.path.join(zip_folder, file)
#             zip_name = os.path.splitext(file)[0]  # Remove .zip extension

#             temp_extract_dir = os.path.join(zip_folder, zip_name + "_temp")  # Temporary extraction folder

#             # Extract the zip file
#             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                 zip_ref.extractall(temp_extract_dir)

#             # Find the first subfolder inside the extracted contents
#             extracted_contents = os.listdir(temp_extract_dir)
#             subfolder = None

#             for item in extracted_contents:
#                 item_path = os.path.join(temp_extract_dir, item)
#                 if os.path.isdir(item_path):  # Look for the first directory inside
#                     subfolder = item_path
#                     break

#             if not subfolder:
#                 print(f"No subfolder found in {zip_path}, skipping...")
#                 shutil.rmtree(temp_extract_dir, ignore_errors=True)
#                 continue

#             # Rename and move the extracted folder within the same directory
#             final_extracted_path = os.path.join(zip_folder, zip_name)
#             shutil.move(subfolder, final_extracted_path)  # Move and rename

#             # Cleanup: Remove the temporary extraction directory
#             shutil.rmtree(temp_extract_dir, ignore_errors=True)

#             print(f"Extracted and renamed: {final_extracted_path}")

# # Example Usage:
# zip_folder_path = r"E:\test"  # Folder containing zip files
# extract_and_rename(zip_folder_path)



##############################################################################################################
##############################################################################################################

# Atmospheric correction

def create_scene_settings_file(acolite_settings_file, proc_dir, scene_basename, geojson_aoi, scene):

    """
        Creates acolite customized settings files (.txt) from the template 
        provided for each scene.
        
        Args:
            acolite_settings_file: path to acolite settings template file (str)
            proc_dir: output directory (str)
            scene_basename: scene id or name (str)
            geojson_aoi: path to geojson file to use as a subset (str)
            scene: path to PlanetScope Superdove L1 TOA file 
            i.e. AnalyticMS.tif, AnalyticMS_clip.tif, Analytic radiance 
            (TOAR) - 8 band data from the Explorer, or analytic_8b_udm2 from the
            API (PSScene)
            
        Returns:
            scene_settings: path to customized scene settings file (str)
            out_dir: path to acolite processed data for scene (str)


        Notes:
            - PlanetScope folders need to be unzipped
            - All RGB band numbers are accounted for in the settings file (all #'s for R, G, and B bands for 
                                                                            both SD and S2')
            - Input folder has to have the subfolders of each image (Safe folder for S2 and PSScene for SD)
            
    """

    with open(acolite_settings_file, 'r') as f:
        settings = f.readlines()
        #scene_basename = key
        out_dir = os.path.join(proc_dir, scene_basename)
        
        #Update each settings file using the base template 
        for n, i in enumerate(settings):
            if i.startswith('inputfile='):
                settings[n] = r'inputfile={}{}'.format(scene , '\n') 
            elif i.startswith('output='):
                settings[n] = r'output={}{}'.format(out_dir, '\n')
            elif i.startswith('polygon='):
                if geojson_aoi:
                    settings[n] = r'polygon={}{}'.format(geojson_aoi, '\n')
                else:
                    settings[n] ="polygon=None \n"
            elif i.startswith('limit='):
                    settings[n] = "limit=None \n"
            
    #Write updated settings file
    scene_settings = os.path.join(proc_dir, scene_basename + '_acolite_settings.txt')
    with open(scene_settings, "w") as f:
            f.write(''.join(settings))
    return scene_settings, out_dir
    
    
def run_acolite_module(acolite_settings, acolite_dir):
    """
        Imports and runs acolite atmospheric correction processor
        on a PlanetScope SuperDove scene.
        
        Args:
            acolite_settings: path to acolite settings for the specific scene (str)
            acolite_dir: path to acolite directory (str)
            
        Returns:
            returncode: 0
            output: success or error info from acolite

    """
    
    sys.path.append(str(acolite_dir))
    import acolite as ac
    output = ac.acolite.acolite_run(acolite_settings)
    returncode = 0
    return returncode, output
    

def main(raw_dir, proc_dir, acolite_dir, acolite_settings_file, geojson_aoi):
    
    # Regex patterns for identifying satellites
    search_regex = {
        'SENTINEL2': re.compile(r'\.SAFE$'),
        #'PLANETDOVE': re.compile(r'.*PSScene.*', re.IGNORECASE),
        'PLANETDOVE': re.compile(r'.*PSScene.*', re.IGNORECASE)
    }

    # Image collection dictionary
    image_collection = {}
    
    # Iterate through the raw directory
    for scene_path in Path(raw_dir).iterdir():
        if scene_path.is_dir():
            for satellite, regex in search_regex.items():
                if regex.search(str(scene_path)):  
                    #print(f"Detected {satellite}")
                    image_collection[scene_path.name] = str(scene_path)
                    #break  
        
    for key,scene in image_collection.items():
        scene_settings, out_dir = create_scene_settings_file(acolite_settings_file, proc_dir, key, geojson_aoi, scene)    

    settings_files= list(Path(proc_dir).rglob('*txt'))
    
    for i in range(len(settings_files)):
            run_acolite_module(str(settings_files[i]), acolite_dir)
         
    print("Images succesfully preprocessed with Acolite atmospheric correction processor")



if __name__ == '__main__':
    
    raw_dir = r"E:\Thesis Stuff\AcoliteWithPython\Uncorrected_Imagery\SD_Anegada"    # dir with folder(s) of raw imagery
    proc_dir = r"E:\Thesis Stuff\AcoliteWithPython\Corrected_Imagery\All_SuperDove\SD_Anegada_output_0.075"   # output directory
    
    # Acolite directory path
    # Example path: E:\downloads_\SatBathy2.1.7-Beta\SatBathy\satbathy\acolite-main
    acolite_dir = r"E:\Thesis Stuff\AcoliteWithPython\acolite-main"

    acolite_settings_file = r"E:\Thesis Stuff\AcoliteWithPython\acolite_settings_planet.txt"  # path to settings file
    
    # path to geojson (needs to be in wgs84), only really required for S2 safe files which have not been clipped
    geojson_aoi = r"B:\Thesis Project\AOIs\Final_AOIs\Anegada.geojson"

    main(raw_dir, proc_dir, acolite_dir, acolite_settings_file, geojson_aoi)    













