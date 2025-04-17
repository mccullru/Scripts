# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:32:54 2025

@author: mccullru
"""

"""
The first part of this code uses an API to download S2 safe files from copernicus sentinel hub. Have to first create a token in 
the Sentinel hub dashboard (current one will expire June 24, 2025), that it calls to. Also need to find a 
unique identifier UUID for each file that can be called instead of the file ID name

The second part creates a list of all image ids that will be used to determine the total amount of imagery at each 
location in a year

"""

import requests
import pandas as pd
import os
import time


##############################################################################################################
##############################################################################################################


# # Your Copernicus Open Access Hub credentials
# username = "mccullru@oregonstate.edu"
# password = "Hurt0226917!"

# # Directory to input file name
# file_path = r"B:\Thesis Project\Raw Imagery\ImageIDs\Individual_AOI_Lists\Sentinel2\Bombah_Libya_S2.xlsx"
# sheet_name = "best"

# # Where the files will be saved
# download_dir = r"E:\Thesis Stuff\AcoliteWithPython\Uncorrected_Imagery\All_Sentinel2\S2_Bombah"

# # Read the specific sheet if the list of best images is on a specific one
# df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl", header=None)

# # list needs to be in first column
# best_img_ID = df.iloc[:,0]
# print('Best Image IDs\n',best_img_ID)

# # renames the IDs to safe_file_ids just because
# safe_file_ids = best_img_ID

# # Authenticates and gets access to the token created in sentinel hub
# def get_keycloak(username: str, password: str) -> str:
#     data = {
#         "client_id": "cdse-public",
#         "username": username,
#         "password": password,
#         "grant_type": "password",
#     }
#     try:
#         r = requests.post(
#             "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
#             data=data,
#         )
#         r.raise_for_status()
#     except Exception as e:
#         raise Exception(
#             f"Keycloak token creation failed. Reponse from the server was: {r.text}"
#         )
#     return r.json()["access_token"]

# # Assigns access token
# keycloak_token = get_keycloak(username, password)

# # Create requests session object to maintain authentication accross multiple requests
# session = requests.Session()
# session.headers.update({"Authorization": f"Bearer {keycloak_token}"})

# # Apparently the safe file ID is not specific enough, need to get the corresponding Universally Unique 
# # Identifier (UUID) for each ID
# def get_uuid_from_name(product_name, session):
#     """Retrieve the UUID for a given Sentinel-2 product name."""
#     search_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq '{product_name}'"

#     response = session.get(search_url)
    
#     if response.status_code == 200:
#         data = response.json()
#         if data["value"]:
#             return data["value"][0]["Id"]  # Extract the UUID
#         else:
#             print(f"Product {product_name} not found.")
#             return None
#     else:
#         print(f" Error retrieving product UUID (Status {response.status_code}): {response.text}")
#         return None


# # download S2 safe files and save to a directory
# for safe_file_id in safe_file_ids:
#     print(f"\nGetting UUID For: {safe_file_id}")
#     try:
        
#         # Convert product name to UUID
#         uuid = get_uuid_from_name(safe_file_id, session)
#         #print('UUID:\n', uuid)
        
#         if not uuid:
#             continue  # Skip if UUID not found
        
#         # construct the download URL using the UUIDs
#         url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({uuid})/$value"
        
#         response = session.get(url, allow_redirects=False)
        
#         while response.status_code in (301, 302, 303, 307):
#             url = response.headers.get("Location")
#             response = session.get(url, allow_redirects=False)

#         print(f"\nDownloading: {safe_file_id}")
        
        
#         # Save file to the download directory with streaming
#         file_path = os.path.join(download_dir, f"{safe_file_id}.zip")
#         with session.get(url, stream=True, verify=False) as r:
#             r.raise_for_status()
#             with open(file_path, "wb") as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     if chunk:
#                         f.write(chunk)
        
        
#         print(f"Successfully Downloaded: {safe_file_id}")
    
#         if response.status_code != 200:
#             print(f"Failed to download {safe_file_id}, Status: {response.status_code}, Response: {response.text}")
    
#     except Exception as e:
#         print(f"Failed to download {safe_file_id}. Error: {e}")

#     time.sleep(20)

##############################################################################################################
##############################################################################################################
        


































