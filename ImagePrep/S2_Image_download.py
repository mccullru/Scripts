# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:32:54 2025

@author: mccullru
"""

"""
The first part of this code uses an API to download S2 safe files from copernicus sentinel hub. Have to first create a token in 
the Sentinel hub dashboard (current one will expire June 24, 2025), that it calls to. Also need to find a 
unique identifier UUID for each file that can be called instead of the file ID name. It technically works, but
it is sooo slow and times out after 1-3 images are downloaded so kind of useless for bulk download which was 
fun to learn

The second part creates a list of all image ids that will be used to determine the total amount of imagery at each 
location in a year

"""

import requests
import pandas as pd
import os
import time
import json
import csv
import pandas as pd

# Copernicus Open Access Hub credentials
username = "mccullru@oregonstate.edu"
password = "Hurt0226917!"

##############################################################################################################
##############################################################################################################


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


# Load your GeoJSON file
with open(r"B:\Thesis Project\AOIs\Final_AOIs\Hyannis.geojson") as f:
    geojson = json.load(f)

# Extract the coordinates of the bounding box
coordinates = geojson['features'][0]['geometry']['coordinates'][0]

# Find the minimum and maximum latitude and longitude
min_lon = min([coord[0] for coord in coordinates])
max_lon = max([coord[0] for coord in coordinates])
min_lat = min([coord[1] for coord in coordinates])
max_lat = max([coord[1] for coord in coordinates])

# Construct the query URL
url = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"
params = {
    "cloudCover": "[0,100]",
    "startDate": "2023-01-01T00:00:00Z",
    "completionDate": "2023-12-31T23:59:59Z",
    "productType": "L1C",
    "maxRecords": "500",
    "box": f"{min_lon},{min_lat},{max_lon},{max_lat}"  # Use the bounding box coordinates
}

# Send the GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response JSON
    data = response.json()
    
    # Output the count of images
    image_count = len(data['features'])
    print(f"Total number of images found: {image_count}")
    
    image_data = []
    
    
    # Output image IDs and their acquisition dates
    print("\nImage IDs and Acquisition Dates:")
    for item in data['features']:
        image_id = item['id']
        acquisition_date = item['properties']['startDate']
        
        image_data.append({"ID": image_id, "Date": acquisition_date})
        
        #print(f"ID: {image_id}, Date: {acquisition_date}")

    
    df = pd.DataFrame(image_data)
    
    df_unique = df.drop_duplicates(subset='Date')
    
    
    # Print unique dates and their count
    print(f"\nTotal unique acquisition dates: {len(df_unique)}")
    print(df_unique)

else:
    print(f"Error: {response.status_code}")


















