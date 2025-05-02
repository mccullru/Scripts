# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:05:56 2025

@author: mccullru
"""


import arcpy
import os


def batch_crop_rasters(raster_workspace, polygon_feature_class, name_field, output_workspace):
    """
    Crops multiple rasters to corresponding polygon boundaries based on matching names.

    Args:
        raster_workspace (str): Path to the directory containing the rasters.
        polygon_feature_class (str): Path to the polygon feature class.
        name_field (str): Name of the field in the feature class containing the raster names.
        output_workspace (str): Path to the directory where cropped rasters will be saved.
    """

    try:
        # Create the output workspace if it doesn't exist
        if not os.path.exists(output_workspace):
            os.makedirs(output_workspace)

        # Get the spatial reference from ONLY THE FIRST RASTER so they all need to be from same AOI 
        first_raster = arcpy.ListRasters("*", "All")[0]  # Get the first raster in the workspace
        first_raster_path = os.path.join(raster_workspace, first_raster)
        raster_spatial_reference = arcpy.Describe(first_raster_path).spatialReference

        # Project the polygon feature class to match the rasters' spatial reference
        output_polygon_feature_class = os.path.join(os.path.dirname(polygon_feature_class), "polygons_projected.gdb", os.path.basename(polygon_feature_class).split(".")[0] + "_projected")

        arcpy.Project_management(polygon_feature_class, output_polygon_feature_class, raster_spatial_reference)

        print(f"Projected polygon feature class to: {output_polygon_feature_class}")

        # Create a search cursor to iterate through the polygons
        with arcpy.da.SearchCursor(output_polygon_feature_class, ["SHAPE@", name_field]) as cursor:
            for row in cursor:
                polygon_geometry = row[0]
                raster_name = row[1]

                # Construct the full path to the corresponding raster
                input_raster = os.path.join(raster_workspace, f"{raster_name}.tif")  # Adjust extension if needed

                # Construct the output raster name
                output_raster = os.path.join(output_workspace, f"{raster_name}_cropped.tif")

                # Check if the input raster exists
                if arcpy.Exists(input_raster):
                    # Clip the raster
                    arcpy.Clip_management(input_raster, "#", output_raster, polygon_geometry, "NONE", "NO_MAINTAIN_EXTENT")
                    print(f"Successfully cropped {raster_name}")
                else:
                    print(f"Warning: Raster {input_raster} not found for polygon {raster_name}")

        print("Batch cropping process complete.")

    except arcpy.ExecuteError:
        print(f"ArcGIS Pro error: {arcpy.GetMessages()}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Set the workspace where your rasters are located
    raster_workspace = "C:/path/to/your/rasters"  # Replace with your actual path

    # Set the path to your polygon feature class
    polygon_feature_class = "C:/path/to/your/geodatabase.gdb/your_polygons"  # Replace with your actual path

    # Set the field name in the polygon feature class that contains the raster names
    name_field = "RasterName"  # Replace with the actual field name

    # Set the output workspace for the cropped rasters
    output_workspace = "C:/path/to/your/cropped_rasters"  # Replace with your actual path

    batch_crop_rasters(raster_workspace, polygon_feature_class, name_field, output_workspace)



