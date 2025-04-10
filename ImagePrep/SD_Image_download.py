# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:48:38 2024

@author: mccullru
"""

# Script that imports TOAR images from PlanetLabs and Copernicus


#API key = PLAK5caebf70644b4981bd7a7598035eb6ae
# https://github.com/planetlabs/notebooks
# API tutorial: jupyter-notebooks/Data-API


import os
import json
import requests
#import geojsonio
import time
from pyproj import Proj, Transformer




# Authentication
###############################################################################

# if your Planet API Key is not set as an environment variable, you can paste it below
if os.environ.get('PL_API_KEY', ''):
    API_KEY = os.environ.get('PL_API_KEY', '')
else:
    API_KEY = 'PLAK5caebf70644b4981bd7a7598035eb6ae'

    # construct auth tuple for use in the requests library
BASIC_AUTH = (API_KEY, '')


# Request
##############################################################################

# Setup Planet Data API base URL
URL = "https://api.planet.com/data/v1"

# Setup the session
session = requests.Session()

# Authenticate
session.auth = (API_KEY, "")

# Make a GET request to the Planet Data API
res = session.get(URL)

# If the code is 200, that is good
print("GET REQUEST:\n",res)

# Response status code
res.status_code

# Response Body
res.text


# Print formatted JSON response
print("\nJSON RESPONSE:\n", res.json())


# Print the value of the item-types key from _links
# Prints a URL that shows all item types (PSScene, REOrthoTile, REScene)
# API key is both username and password
print("\nITEM TYPES URL:\n", res.json()["_links"]["item-types"])




# Searching with Filters
###############################################################################

# Specify the sensors/satellites or "item types" to include in our results
item_types = ["PSScene"]


## Create Filters ##

# DateRangeFilter
# Finds imagery that was acquired within certain dates
date_filter = {
    "type": "DateRangeFilter", # Type of filter -> Date Range
    "field_name": "acquired", # The field to filter on: "acquired" -> Date on which the "image was taken"
    "config": {
        "gte": "2022-01-01T00:00:00.000Z", # "gte" -> Greater than or equal to
        "lte": "2022-12-31T00:00:00.000Z"  # "lte" -> Less than or equal to
    }
}

# RangeFilter
# finds imagery that has a metadata that matches the field name and config
cloud_filter = {
    "type": "RangeFilter",
    "field_name": "cloud_cover",
    "config": {
        "lte": 1
        }
    }

# StringInFilter
# Finds imagery that has a string in the metadata that matches the config
PSB_SD_filter = {
    "type": "StringInFilter",
    "field_name": "instrument",
    "config": ["PSB.SD"]
    }


# Load the GeoJSON file
# NOTE: Created the GeoJSON file in ArcPro with the "Features to JSON tool,
#       Need to check "Output to GeoJSON" and "Project to WGS_1984" 
geojson_path = r"E:\Thesis Stuff\AOI\Homer_small.geojson"

with open(geojson_path, 'r') as f:
    geojson_data = json.load(f)

geometry = geojson_data['features'][0]['geometry']


# GeometryFilter
# Finds data contained within a given geometry, needs to be in GeoJSON format
GeometryFilter = {
    "type": "GeometryFilter",
    "field_name": "geometry",
    "config": geometry}


combined_filter = {
    "type": "AndFilter",
    "config": [date_filter, 
               cloud_filter,
               PSB_SD_filter,
               GeometryFilter]}


#print("\ncombined filter:\n", combined_filter)




# Stats stuff?
###############################################################################

# Setup the stats URL
stats_url = "{}/stats".format(URL)

# Print the stats URL
print("\nSTATS URL:\n" ,stats_url)



# Construct the request.
############################################################################### 
request = {
    "item_types" : item_types,
    "interval" : "year",
    "filter" : combined_filter
    
}

# Send the POST request to the API stats endpoint
res = session.post(stats_url, json=request)

#Print response
print("\nPOST REQUEST:\n", res.json())



# Quick Search
###############################################################################

all_features = []

# Setup the quick search endpoint url
quick_url = "{}/quick-search".format(URL)

# Setup Item Types
item_types = ["PSScene"]


# Setup the request
request = {
    "item_types" : item_types,
    "filter" : combined_filter}



# Fetch all features using pagination
all_features = []  # List to store all retrieved features
search_after = None  # Initialize pagination token

while True:
    # Add search_after parameter if it's not the first request
    if search_after:
        request["search"] = {"search_after": search_after}  # Set pagination token

    # Make a POST request to get a batch of features
    res = session.post(quick_url, json=request)

    if res.status_code != 200:
        print(f"Request failed with status code: {res.status_code}")
        print("Error response:", res.text)
        break  # Stop if there's an error

    data = res.json()
    
    # Extract features from this batch
    features_batch = data.get("features", [])
    if not features_batch:
        break  # No more results

    all_features.extend(features_batch)

    # Print progress
    print(f"Retrieved {len(all_features)} features so far...")

    # Get search_after from the last feature
    search_after = features_batch[-1].get("search_after")
    if not search_after:
        break  # No more pagination token, exit loop

# Print final count of features
print(f"\nTotal number of features retrieved: {len(all_features)}")

# Loop over all the features from the response
print("\nFeature IDs:")
for f in all_features:
    print(f["id"])





# # Send the POST request to the API quick search endpoint
# res = session.post(quick_url, json=request)

# # Assign the response to a variable
# geojson = res.json()

# # Print the response
# #print("\nGEOJSON:\n", geojson)


# # Assign a features variable 
# features = geojson["features"]

# # Get the number of features present in the response
# print("\nNUMBER OF FEATURES:\n", len(features))



# # Loop over all the features from the response
# print("\nFeature IDs:")
# for f in features:
#     # Print the ID for each feature
#     print(f["id"])

# # Print the first feature
# #print("\nFIRST FEATURE:\n", features[0])




# Check if the request was successful
############################################################################### 
if res.status_code == 200:
    print("\nRequest successful!\n")
    
    # Pretty print the JSON response
    #print("\nJSON RESPONSE:\n", json.dumps(res.json(), indent=4))
else:
    print("\nRequest failed with status code:", res.status_code, "\n")
    print("Error response:", res.text)



# # Permissions
# ###############################################################################

# # Assign a variable to the search result features (items)
# features = geojson["features"]

# # Get the first result's feature
# feature = features[0]

# # Print the first Feature's ID
# print("\nFEATURE ID:\n", feature["id"])

# # Print the permissions
# print("\nPERMISSIONS:\n", feature["_permissions"])


# # Get the assets link for the item
# assets_url = feature["_links"]["assets"]

# # Print the assets link
# print("\nASSETS URL:\n", assets_url)
# # Send a GET request to the assets url for the item (Get the list of available assets for the item)
# res = session.get(assets_url)

# # Assign a variable to the response
# assets = res.json()
# # Print the asset types that are available
# print("\nASSETS KEY:\n", assets.keys())

# # Assign a variable to the visual asset from the item's assets
# ortho_analytic4b = assets["ortho_analytic_4b"]

# # Print the visual asset data
# print("\nVISUAL ASSET DATA: ORTHO_ANALYTIC4B:\n", ortho_analytic4b)



# # Activating Assets
# ###############################################################################

# # Setup the activation url for the basic_analytic_4b asset
# activation_url = ortho_analytic4b["_links"]["activate"]

# # Send a request to the activation url to activate the item
# res = session.get(activation_url)

# # Print the response from the activation request
#     # 202: Request has been accepted and activation will begin
#     # 204: The asset is already active and no further action is needed
#     # 401: The user does not have permissions to download this file
# print("\nACTIVATION REQUEST STATUS CODE:\n", res.status_code)



# # Clipping Assets
# ###############################################################################

# # Construct clip API payload

# # clip_payload = {
# #     "tools": [
# #         {
# #             "type": "clip",
# #             "parameters": {
# #                 "aoi": geometry
# #             }
# #         }
# #     ]
# # }

# # Define the payload from your example
# clip_payload = {
#     "name": "clip_example",
#     "source_type": "scenes",
#     "products": [
#         {
#             "item_ids": [
#                 "20220129_181454_66_2434"  # Replace this with your scene ID
#             ],
#             "item_type": "PSScene",
#             "product_bundle": "ortho_analytic_4b"  # Match your intended product
#         }
#     ],
#     "tools": [
#         {
#             "clip": {
#                 "aoi": geometry
#                     }
#             }
#     ]
# }



# # define url for orders api
# clip_url = "https://api.planet.com/compute/ops/clips/v2"

# # send request
# order_response = session.post(clip_url, json=clip_payload)
# print('\nORDER STATUS CODE:\n', order_response.status_code)
# print('\nORDER ID:\n', order_response.json()['id'])




# # print("\nCLIP PAYLOAD:\n", clip_payload)

# # # Request clip of scene (This will take some time to complete)
# # request = requests.post('https://api.planet.com/compute/ops/clips/v1', 
# #                         auth=(BASIC_AUTH), json=clip_payload)

# # print("\nREQUEST STATUS CODE:\n", request)
# # print("\nREQUEST RESPONSE TEXT:\n", request.text)

# # clip_url = request.json()['_links']['_self']



# # # Poll API to monitor clip status. Once finished, download and upzip the scene
# # clip_succeeded = False
# # while not clip_succeeded:

# #     # Poll API
# #     check_state_request = requests.get(clip_url, auth=(API_KEY, ''))
    
# #     # If clipping process succeeded , we are done
# #     if check_state_request.json()['state'] == 'succeeded':
# #         clip_download_url = check_state_request.json()['_links']['results'][0]
# #         clip_succeeded = True
# #         print("Clip of scene succeeded and is ready to download") 
    
# #     # Still activating. Wait 1 second and check again.
# #     else:
# #         print("...Still waiting for clipping to complete...")
# #         time.sleep(1)



# #Downloading Assets
# ##############################################################################

# # # Assign a variable to the visual asset's location endpoint
# # location_url = ortho_analytic_4b["location"]

# # # Print the location endpoint
# # print("\nLOCATION URL:\n", location_url)









# #############################################################################################################
# #############################################################################################################
# # No filtering, only have scene IDs for input which will then be clipped and downloaded


# import os
# import json
# import requests
# #import geojsonio
# #import time
# #from pyproj import Proj, Transformer

# import requests
# #import time
# import json

# # Your Planet API key
# API_KEY = "PLAK5caebf70644b4981bd7a7598035eb6ae"

# # The GeoJSON AOI for clipping
# geojson_path = "B:\Thesis Project\ArcGIS Pro\Fieldwork Sites\AOISallysBend_FeaturesToJSON.geojson"

# with open(geojson_path, 'r') as f:
#     geojson_data = json.load(f)

# geometry = geojson_data['features'][0]['geometry']



# # List of scene IDs to process
# scene_ids = [
#     "20220129_181454_66_2434",
#     "20220127_190512_88_2413",
#     "20220125_181445_12_242d"
# ]

# # Define the clipping payload template
# def create_clip_payload(scene_id, aoi):
#     return {
#         "name": f"clip_{scene_id}",
#         "source_type": "scenes",
#         "products": [
#             {
#                 "item_ids": [scene_id],
#                 "item_type": "PSScene",
#                 "product_bundle": "ortho_analytic_4b"
#             }
#         ],
#         "tools": [
#             {
#                 "clip": {
#                     "aoi": aoi
#                 }
#             }
#         ]
#     }

# # Define the Clip API URL
# CLIP_URL = "https://api.planet.com/compute/ops/clips/v1"

# # Submit clipping jobs and store job URLs
# clip_jobs = []
# for scene_id in scene_ids:
#     payload = create_clip_payload(scene_id, geometry)
#     response = requests.post(CLIP_URL, auth=(API_KEY, ""), json=payload)
    
#     if response.status_code == 200:
#         job_url = response.json()["_links"]["_self"]
#         print(f"\nSubmitted clipping job for {scene_id}. Monitoring at: {job_url}\n")
#         clip_jobs.append({"scene_id": scene_id, "job_url": job_url})
#     else:
#         print(f"\nFailed to submit clipping job for {scene_id}: {response.text}\n")

# # Monitor jobs and download completed assets
# # def download_asset(asset_url, output_file):
# #     response = requests.get(asset_url, auth=(API_KEY, ""), stream=True)
# #     if response.status_code == 200:
# #         with open(output_file, "wb") as f:
# #             for chunk in response.iter_content(chunk_size=1024):
# #                 f.write(chunk)
# #         print(f"Downloaded: {output_file}")
# #     else:
# #         print(f"Failed to download asset: {response.text}")

# # for job in clip_jobs:
# #     scene_id = job["scene_id"]
# #     job_url = job["job_url"]
    
# #     while True:
# #         # Check job status
# #         response = requests.get(job_url, auth=(API_KEY, ""))
# #         if response.status_code == 200:
# #             status = response.json()["state"]
# #             if status == "succeeded":
# #                 print(f"Clipping job for {scene_id} succeeded.")
# #                 asset_url = response.json()["_links"]["results"][0]["location"]
# #                 output_file = f"{scene_id}_clipped.tif"
# #                 download_asset(asset_url, output_file)
# #                 break
# #             elif status == "failed":
# #                 print(f"Clipping job for {scene_id} failed.")
# #                 break
# #             else:
# #                 print(f"Job for {scene_id} is {status}. Waiting...")
# #                 time.sleep(30)  # Wait before checking again
# #         else:
# #             print(f"Error checking job status for {scene_id}: {response.text}")
# #             break





