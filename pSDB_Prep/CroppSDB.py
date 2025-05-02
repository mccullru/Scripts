# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:05:56 2025

@author: mccullru
"""

"""
!!! ONLY WORKS IN ARCPRO ENVIRONMENT!!!

This is just as a backup

"""



import arcpy
import os
import time

def batch_maskout_by_filename(raster_folder, polygon_folder, output_folder, raster_ext=".tif"):
    """
    Masks out areas within rasters using correspondingly named polygon
    shapefiles. Cells inside the polygon mask are set to NoData.
    REQUIRES SPATIAL ANALYST EXTENSION.

    Args:
        raster_folder (str): Folder containing input raster files.
        polygon_folder (str): Folder containing individual polygon shapefiles.
                               Each shapefile name (e.g., 'XYZ.shp') should match
                               the base name of a raster (e.g., 'XYZ.tif').
                               ASSUMES polygons and rasters share the same CRS.
        output_folder (str): Folder where masked rasters will be saved.
        raster_ext (str): The file extension for your raster files (e.g., ".tif", ".img").
                          Include the dot.
    """
    print("--- Starting Batch Mask-Out by Matching Filenames ---")
    print(f"Raster Folder: {raster_folder}")
    print(f"Polygon Shapefile Folder: {polygon_folder}")
    print(f"Output Folder: {output_folder}")
    print(f"Raster Extension: '{raster_ext}'")
    print("IMPORTANT: Assuming input polygons and rasters share the same Coordinate Reference System.")
    print("IMPORTANT: Requires ArcGIS Spatial Analyst Extension.")

    # --- Check for Spatial Analyst Extension ---
    try:
        if arcpy.CheckExtension("Spatial") == "Available":
            arcpy.CheckOutExtension("Spatial")
            print("  - Spatial Analyst extension checked out.")
        else:
            print("Error: Spatial Analyst extension is not available or cannot be checked out.")
            return
    except Exception as e:
        print(f"Error checking out Spatial Analyst extension: {e}")
        return

    # --- Basic input validation ---
    if not os.path.isdir(raster_folder):
        print(f"Error: Raster folder not found: '{raster_folder}'")
        arcpy.CheckInExtension("Spatial") # Check extension back in
        return
    if not os.path.isdir(polygon_folder):
        print(f"Error: Polygon folder not found: '{polygon_folder}'")
        arcpy.CheckInExtension("Spatial")
        return
    if not raster_ext.startswith('.'):
        print(f"Warning: raster_ext ('{raster_ext}') might need a leading dot (e.g., '.tif'). Adding one.")
        raster_ext = '.' + raster_ext.lstrip('.')

    # --- Create output folder if needed ---
    if not os.path.exists(output_folder):
        print(f"Creating output folder: {output_folder}")
        os.makedirs(output_folder)

    processed_count = 0
    error_count = 0
    skipped_count = 0
    raster_not_found_count = 0

    # Store original environment settings
    orig_extent = arcpy.env.extent
    orig_snapRaster = arcpy.env.snapRaster
    orig_cellSize = arcpy.env.cellSize
    orig_mask = arcpy.env.mask # Just in case

    try: # Use try...finally to ensure environment settings are reset
        # --- Iterate through polygon shapefiles ---
        print(f"Scanning polygon folder '{polygon_folder}' for shapefiles...")
        for poly_filename in os.listdir(polygon_folder):
            if poly_filename.lower().endswith(".shp"):
                polygon_shapefile_path = os.path.join(polygon_folder, poly_filename)
                base_name = os.path.splitext(poly_filename)[0]

                # Construct paths
                raster_filename = f"{base_name}{raster_ext}"
                raster_path = os.path.join(raster_folder, raster_filename)
                output_raster_basename = f"{base_name}_masked{raster_ext}"
                output_raster_path = os.path.join(output_folder, output_raster_basename)

                print(f"\nChecking for match: Polygon '{poly_filename}' -> Raster '{raster_filename}'")

                # Check if the corresponding raster exists
                if arcpy.Exists(raster_path):
                    print(f"  Found matching raster: '{raster_filename}'. Masking...")
                    temp_poly_raster = None # Variable to store temp raster path
                    try:
                        # --- Set environment for alignment ---
                        print(f"    Setting raster analysis environment based on: {raster_filename}")
                        arcpy.env.snapRaster = raster_path
                        arcpy.env.extent = raster_path
                        arcpy.env.cellSize = raster_path
                        arcpy.env.mask = None # Ensure no other mask is active

                        # --- Rasterize the polygon ---
                        # Define temp raster path in scratch GDB
                        # Add timestamp for uniqueness in case of script restart/parallel issues
                        timestamp = int(time.time() * 1000)
                        temp_raster_name_base = f"{arcpy.ValidateTableName(base_name, arcpy.env.scratchGDB)}_poly_{timestamp}"
                        temp_poly_raster = os.path.join(arcpy.env.scratchGDB, temp_raster_name_base)

                        print(f"    Rasterizing polygon to temporary raster: {temp_poly_raster}")
                        # Use a field that exists, or default behavior if no suitable field. OBJECTID usually exists.
                        # Use cell size from the source raster for perfect alignment.
                        cell_size_from_raster = arcpy.Describe(raster_path).meanCellHeight # Or width
                        arcpy.conversion.PolygonToRaster(in_features=polygon_shapefile_path,
                                                        value_field="Name", # Or any field that exists
                                                        out_rasterdataset=temp_poly_raster,
                                                        cell_assignment="CELL_CENTER", # Or MAXIMUM_AREA
                                                        priority_field="NONE",
                                                        cellsize=cell_size_from_raster) # Crucial for alignment

                        # --- Perform Masking using SetNull ---
                        print(f"    Applying mask using SetNull...")
                        # Create Raster objects for Spatial Analyst tools
                        inRas = arcpy.Raster(raster_path)
                        polyRas = arcpy.Raster(temp_poly_raster)

                        # SetNull where polyRas is 1 (or any value assigned by PolygonToRaster)
                        # Assuming PolygonToRaster assigns a value (like the OBJECTID) inside, and NoData outside
                        # Use IsNull to check where the polygon raster *doesn't* exist (outside)
                        # Con(IsNull(polyRas), inRas) keeps values outside polygon, sets inside to NoData
                        outRas = arcpy.sa.Con(arcpy.sa.IsNull(polyRas), inRas)
                        # Alternative using SetNull: SetNull(polyRas >= 0, inRas) -> assumes NoData is < 0, might vary
                        # Safer with Con(IsNull...)

                        # --- Save the output raster ---
                        print(f"    Saving masked output to: {output_raster_basename}")
                        outRas.save(output_raster_path)

                        print(f"  Successfully created: {output_raster_basename}")
                        processed_count += 1

                    except arcpy.ExecuteError:
                        print(f"  ERROR processing {raster_filename}: {arcpy.GetMessages(2)}")
                        error_count += 1
                    except Exception as e:
                        print(f"  UNEXPECTED ERROR processing {raster_filename}: {e}")
                        error_count += 1
                    finally:
                        # --- Clean up temporary polygon raster ---
                        if temp_poly_raster and arcpy.Exists(temp_poly_raster):
                            try:
                                print(f"    Deleting temporary raster: {temp_poly_raster}")
                                arcpy.management.Delete(temp_poly_raster)
                            except Exception as e_del:
                                print(f"    Warning: Failed to delete temporary raster {temp_poly_raster}: {e_del}")

                else:
                    print(f"  Warning: Matching raster '{raster_filename}' not found in '{raster_folder}'.")
                    raster_not_found_count += 1

            # elif os.path.isfile(os.path.join(polygon_folder, poly_filename)):
            #      skipped_count += 1

    finally: # Ensure environment is reset and extension checked in
        # --- Reset original environment settings ---
        arcpy.env.extent = orig_extent
        arcpy.env.snapRaster = orig_snapRaster
        arcpy.env.cellSize = orig_cellSize
        arcpy.env.mask = orig_mask
        # Check Spatial Analyst back in
        arcpy.CheckInExtension("Spatial")
        print("\nEnvironment settings reset. Spatial Analyst extension checked in.")

    # --- Summary ---
    print("\n--- Batch mask-out process complete ---")
    print(f"Successfully processed: {processed_count} rasters")
    print(f"Matching rasters not found: {raster_not_found_count}")
    print(f"Errors during processing: {error_count}")
    # if skipped_count > 0: print(f"Files skipped in polygon folder (not .shp): {skipped_count}")


if __name__ == "__main__":

    # Folder containing your input rasters (e.g., .tif files)
    raster_input_folder = r"E:\Thesis Stuff\Cropping_test\Inputs\Rasters"

    # Folder containing your individual polygon shapefiles (masks)
    polygon_input_folder = r"E:\Thesis Stuff\Cropping_test\Inputs\Masks"

    # Folder where the masked output rasters should be saved
    masked_output_folder = r"E:\Thesis Stuff\Cropping_test\Outputs"

    # Extension of your RASTER files
    raster_file_extension = ".tif"

    # === Run the function ===
    batch_maskout_by_filename(raster_input_folder, polygon_input_folder, masked_output_folder, raster_file_extension)



