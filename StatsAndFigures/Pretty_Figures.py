# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:47:25 2025

@author: mccullru
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import mplcursors

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde, skew, kurtosis


##############################################################################################################
##############################################################################################################

"""Histogram scatter heat map"""


# def heatscatter(ax, x, y,
#                 bins, title, cmap,
#                 xlabel, ylabel, identity_line=False,
#                 xlim=None, ylim=None, # User-provided limits, can contain None
#                 **kwargs):
#     x_arr = np.asarray(x)
#     y_arr = np.asarray(y)
    
#     finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
#     x_finite = x_arr[finite_mask]
#     y_finite = y_arr[finite_mask]

#     # Apply user-specified limits (or None for autoscaling) to the axes.
#     ax.set_xlim(xlim) 
#     ax.set_ylim(ylim)

#     # Retrieve the actual numeric limits that Matplotlib has set.
#     current_plot_xlim = ax.get_xlim() 
#     current_plot_ylim = ax.get_ylim()

#     # Set titles and labels.
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     if len(x_finite) == 0 or len(y_finite) == 0:
#         # Generalized warning message
#         print(f"  Warning for plot '{title}': No finite data points left after filtering NaNs/Infs. Cannot plot hist2d.")
#         if identity_line:
#             lim_min_plot = max(current_plot_xlim[0], current_plot_ylim[0])
#             lim_max_plot = min(current_plot_xlim[1], current_plot_ylim[1])
#             if lim_min_plot < lim_max_plot:
#                 ax.plot([lim_min_plot, lim_max_plot], [lim_min_plot, lim_max_plot], 'k--', linewidth=1)
#         return None

#     # Step 3: Use the resolved current_plot_xlim and current_plot_ylim for the 'range' of hist2d.
#     hs = ax.hist2d(x_finite, y_finite, bins=bins, cmin=1, cmap=cmap, 
#                     range=[current_plot_xlim, current_plot_ylim], # Pass resolved limits
#                     **kwargs)
    
#     if identity_line:
#         lim_min_plot = max(current_plot_xlim[0], current_plot_ylim[0])
#         lim_max_plot = min(current_plot_xlim[1], current_plot_ylim[1])
#         if lim_min_plot < lim_max_plot:
#             ax.plot([lim_min_plot, lim_max_plot], [lim_min_plot, lim_max_plot], 'k--', linewidth=1)
    
#     return hs


# # Define FOLDER path containing CSV files and column names
# csv_folder_path = r"E:\Thesis Stuff\zquick save\SDB_ExtractedPts_s2" 
# x_column_name = "Raster_Value"
# y_column_name = "Geoid_Corrected_Ortho_Height"

# # Define desired axis limits - these will be applied to EACH plot
# manual_xlim = (0, 10)
# manual_ylim = (0, 10)


# # Find all CSV files in the specified folder
# csv_files = glob.glob(os.path.join(csv_folder_path, "*.csv"))

# if not csv_files:
#     print(f"Error: No CSV files found in the folder: {csv_folder_path}")
# else:
#     print(f"Found {len(csv_files)} CSV files. Generating a separate heatmap for each.\n")

#     for file_path in csv_files:
#         print(f"Processing file: {os.path.basename(file_path)}")
#         try:
#             data_df = pd.read_csv(file_path)
            
#             if not (x_column_name in data_df.columns and y_column_name in data_df.columns):
#                 print(f"  Warning: Skipping file. One or both specified columns ('{x_column_name}', '{y_column_name}') not found.")
#                 print(f"    Available columns in this file: {data_df.columns.tolist()}")
#                 continue # Skip to the next file

#             x_data_from_file = pd.to_numeric(data_df[x_column_name], errors='coerce').values
#             y_data_from_file = pd.to_numeric(data_df[y_column_name], errors='coerce').values
            
#             # Check if there's any valid numeric data to plot for this file after coercion
#             # np.isfinite checks for not NaN and not Inf.
#             valid_points_mask = np.isfinite(x_data_from_file) & np.isfinite(y_data_from_file)
#             if not np.any(valid_points_mask): # If no points are finite
#                   print(f"  Warning: No valid numeric data pairs found in columns '{x_column_name}' and '{y_column_name}' for file {os.path.basename(file_path)}. Skipping plot.")
#                   continue

#             # Create a new figure and axis for each plot
#             fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figsize as needed

#             plot_title = f"Density Heatmap: {os.path.basename(file_path)}"

#             # Call your heatscatter function with data from the current file
#             hist_output = heatscatter(ax, x_data_from_file, y_data_from_file,
#                                       bins=100, # Or adjust as needed for individual plots
#                                       title=plot_title,
#                                       cmap='viridis',
#                                       xlabel=f"{x_column_name} Values",
#                                       ylabel=f"{y_column_name} Values",
#                                       identity_line=True,
#                                       xlim=manual_xlim, 
#                                       ylim=manual_ylim  
#                                       )

#             if hist_output is not None:
#                 # hist_output from ax.hist2d is (counts, xedges, yedges, image)
#                 # The image (QuadMesh) is the fourth element, needed for plt.colorbar.
#                 if isinstance(hist_output, tuple) and len(hist_output) == 4:
#                     plt.colorbar(hist_output[3], ax=ax, label='Counts per Bin')
#                 else:
  
#                     print(f"  Note: Could not create colorbar for {os.path.basename(file_path)}. hist_output type: {type(hist_output)}")
            
#             plt.tight_layout()
#             plt.show() 

#         except FileNotFoundError: # This specific error should be rare if glob.glob worked
#             print(f"  Error: CSV file not found at {file_path} (unexpected).")
#         except pd.errors.EmptyDataError:
#             print(f"  Warning: Skipping empty CSV file: {os.path.basename(file_path)}")
#         except Exception as e:
#             print(f"  An error occurred while processing file {os.path.basename(file_path)}: {e}")
#             # import traceback # Uncomment for detailed debugging
#             # traceback.print_exc() # Uncomment for detailed debugging
#         print("-" * 30) # Separator between file processing logs

# if csv_files: 
#     print("\nFinished processing all files.")

##############################################################################################################
##############################################################################################################


""" A single plot with multiple Kernel Density Estimates (KDE) which represent the histogram values without
cluttering the chart. Also adds an average line from all the plotted lines """


""" 
!! NOTE !!: If you want to be able to use the interactive plot settings to look at individual line info, have
            to go to Preferences > IPython console > Graphics tab > Backend: "Inline" for plot placement on right, 
            "Automatic" for a separate window to pop-up and allow interactions

!! OTHER WEIRD NOTE !!: When in interactive mode, in order to look at the name for each individual line, don't
                        just click on it, you have to hold the mouse button down and hover over the line.
                        It's weird, I know

"""


# Configuration 
# Specify the folder containing your input CSV files
input_csv_folder_path = r"E:\Thesis Stuff\SDB_ExtractedPts"

AOI = 'Marathon' 

SDB_type = 'SDBall' 

#Sensor = 'SuperDove' 
Sensor = 'Sentinel-2'

output_plot_folder_path = r"B:\Thesis Project\SDB_Time\Results_main\Marathon\Sentinel-2\Final5\Extracted Pts\New folder\test_output"
combined_plot_filename = "combined_error_kdes_with_average_stats.png"

# Define column names
ref_col = "Geoid_Corrected_Ortho_Height"
SDB_col = "Raster_Value"
error_col_name = "Error"

# Plot settings
plot_xlim = (-4, 4)
plot_ylim = (0, 800)

# Individual KDE line style
individual_kde_color = 'gray'
individual_kde_linestyle = '-'
individual_kde_linewidth = 1.0

# Define fixed_bin_width for consistent scaling
fixed_bin_width = 0.1 # 
# This 'fixed_bin_width' will serve as the 'common_bin_width_for_scaling'



# Create Output Folder if it Doesn't Exist ---
if not os.path.exists(output_plot_folder_path):
    os.makedirs(output_plot_folder_path)
    print(f"Created output folder: {output_plot_folder_path}")

# Load Data from ALL CSV Files ---
all_error_data_for_overall_range = []
datasets_for_kde = []

csv_files = glob.glob(os.path.join(input_csv_folder_path, "*.csv"))

if not csv_files:
    print(f"No CSV files found in the folder: {input_csv_folder_path}")
else:
    print(f"Found {len(csv_files)} CSV files to process in: {input_csv_folder_path}")

for i, csv_file_path in enumerate(csv_files):
    print(f"\n--- Reading file: {csv_file_path} ---")
    base_filename = os.path.basename(csv_file_path)
    filename_no_ext = os.path.splitext(base_filename)[0]

    try:
        df = pd.read_csv(csv_file_path)
        if ref_col not in df.columns or SDB_col not in df.columns:
            print(f"Warning: Required columns ('{ref_col}', '{SDB_col}') not found in {base_filename}. Skipping.")
            continue
        df[ref_col] = pd.to_numeric(df[ref_col], errors='coerce')
        df[SDB_col] = pd.to_numeric(df[SDB_col], errors='coerce')
        df = df.dropna(subset=[ref_col, SDB_col])
        if df.empty:
            print(f"Warning: No valid numeric data after coercion/NaN removal in {base_filename}. Skipping.")
            continue

        df[error_col_name] = df[SDB_col] - df[ref_col]
        error_data = df[error_col_name].dropna()

        if error_data.empty:
            print(f"Warning: No error data to process for {base_filename} after dropna. Skipping.")
            continue
        if len(error_data) < 2:
            print(f"Warning: Not enough data points ({len(error_data)}) for KDE in {base_filename}. Skipping.")
            continue

        all_error_data_for_overall_range.extend(error_data.values)
        datasets_for_kde.append({'label': filename_no_ext, 'data': error_data, 'N': len(error_data)})
        print(f"Loaded {len(error_data)} error values from '{base_filename}'.")
    except Exception as e:
        print(f"An error occurred while processing {csv_file_path}: {e}")


# Plot Multiple Scaled KDEs on a Single Figure ---
if datasets_for_kde: # Make sure datasets_for_kde is populated before this block
    fig, ax = plt.subplots(figsize=(12, 7))

    common_bin_width_for_scaling = fixed_bin_width
    print(f"Using fixed common bin_width for KDE scaling: {common_bin_width_for_scaling:.4f}")

    max_scaled_kde_y = 0
    all_scaled_kde_y_arrays = []
    common_kde_x = None
    
    plotted_lines_info = [] # For mplcursors tooltips
    interactive_kde_lines = [] # For click-to-highlight line interaction

    # Define highlight styles (used when a line is clicked)
    HIGHLIGHT_COLOR = 'blue'
    HIGHLIGHT_LINEWIDTH = 3.0 # Make it noticeably thicker
    HIGHLIGHT_ALPHA = 1.0     # Fully opaque
    HIGHLIGHT_ZORDER = 10     # Ensure highlighted line is on top
    # Note: DIMMED_ALPHA and DIMMED_LINEWIDTH_FACTOR are removed as per your request
    # to revert non-highlighted lines to original, not dim them further.

    for i, dataset_info in enumerate(datasets_for_kde):
        label = dataset_info['label']
        error_values = dataset_info['data']
        N = dataset_info['N']

        try:
            bw_factor = 0.25
            kde = gaussian_kde(error_values, bw_method=bw_factor)

            if common_kde_x is None:
                x_min_kde = plot_xlim[0] - (plot_xlim[1] - plot_xlim[0]) * 0.1
                x_max_kde = plot_xlim[1] + (plot_xlim[1] - plot_xlim[0]) * 0.1
                common_kde_x = np.linspace(x_min_kde, x_max_kde, 400)

            kde_y_density = kde(common_kde_x)
            scaled_kde_y = kde_y_density * N * common_bin_width_for_scaling
            all_scaled_kde_y_arrays.append(scaled_kde_y)

            if scaled_kde_y.size > 0:
                current_max_y = np.max(scaled_kde_y)
                if current_max_y > max_scaled_kde_y:
                    max_scaled_kde_y = current_max_y
            
            # Plot individual KDE and make it pickable
            line, = ax.plot(common_kde_x, scaled_kde_y,
                            color=individual_kde_color,
                            linestyle=individual_kde_linestyle,
                            linewidth=individual_kde_linewidth,
                            alpha=0.7, # Original alpha for individual lines
                            picker=True,
                            pickradius=5) 
            
            plotted_lines_info.append({'line': line, 'label': label.replace("_", " ")})
            interactive_kde_lines.append({
                'line': line,
                'original_color': individual_kde_color,
                'original_linewidth': individual_kde_linewidth,
                'original_alpha': 0.7, # Store initial alpha
                'original_zorder': line.get_zorder()
            })

        except Exception as e_kde:
            print(f"Warning: Could not compute or plot KDE for {label}. Error: {e_kde}")

    average_line_handle = None
    average_kde_stats_text = "Average KDE Stats:\nN/A" # Initialize

    if all_scaled_kde_y_arrays and common_kde_x is not None:
        if len(all_scaled_kde_y_arrays) > 0:
            stacked_kdes = np.array(all_scaled_kde_y_arrays)
            average_kde_y = np.mean(stacked_kdes, axis=0)

            if average_kde_y.size > 0:
                current_max_avg_y = np.max(average_kde_y)
                if current_max_avg_y > max_scaled_kde_y:
                    max_scaled_kde_y = current_max_avg_y
            
            avg_line_plots = ax.plot(common_kde_x, average_kde_y,
                            color='red',
                            linestyle='-.',
                            linewidth=2.5,
                            label=f'Average KDE (of {len(all_scaled_kde_y_arrays)} datasets)',
                            zorder=len(interactive_kde_lines) + 10) # Ensure average is well above individuals
            if avg_line_plots:
                average_line_handle = avg_line_plots[0]
                plotted_lines_info.append({'line': average_line_handle, 'label': average_line_handle.get_label()})
            print("Plotted Average KDE line.")
            
            if all_error_data_for_overall_range:
                pooled_errors = np.array(all_error_data_for_overall_range)
                avg_kde_mean = np.mean(pooled_errors); avg_kde_min = np.min(pooled_errors)
                avg_kde_max = np.max(pooled_errors); avg_kde_std = np.std(pooled_errors)
                avg_kde_kurtosis = kurtosis(pooled_errors, fisher=True)
                avg_kde_skewness = skew(pooled_errors)
                average_kde_stats_text = (
                    f"Summary Stats (All Data):\n\n"
                    f"Mean = {avg_kde_mean:.3f} m\nMin = {avg_kde_min:.2f} m\nMax = {avg_kde_max:.2f} m\n"
                    f"Std Dev = {avg_kde_std:.2f} m\nSkewness = {avg_kde_skewness:.2f}\nKurtosis = {avg_kde_kurtosis:.2f}\n"
                    f"Total Pts = {len(pooled_errors)}"
                )
            else: 
                average_kde_stats_text = "Average KDE Stats:\n(No pooled data for stats)"

    # --- Axis and Title Setup ---
    ax.set_xlabel(error_col_name + " (m)", fontsize=14) # Use defined error_col_name
    ax.set_ylabel("Smoothed Frequency Estimate (Scaled Density)", fontsize=14)
    ax.set_title(f"Distribution of {Sensor} {SDB_type} Error Values: {AOI}", fontsize=16)
    ax.set_xlim(plot_xlim)
    if plot_ylim is not None: 
        ax.set_ylim(plot_ylim)
    elif max_scaled_kde_y > 0 : 
        ax.set_ylim(0, max_scaled_kde_y * 1.1)
    else: 
        ax.set_ylim(0, 10) # Fallback if no data

    if all_scaled_kde_y_arrays : # Check if there's data before trying to place text
        text_x_pos = plot_xlim[0] + (plot_xlim[1] - plot_xlim[0]) * 0.02
        current_ax_ylim = ax.get_ylim() # Get current y-limits after they are set
        text_y_pos = current_ax_ylim[1] * 0.98 
        ax.text(text_x_pos, text_y_pos, average_kde_stats_text,
                fontsize=12, color='black', ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.75))

    # --- Event Handler for Highlighting Lines (Pick Event) ---
    def on_pick(event):
        picked_line_artist = event.artist
        
        is_interactive_target = False
        for item_check in interactive_kde_lines:
            if item_check['line'] == picked_line_artist:
                is_interactive_target = True
                break
        
        if not is_interactive_target:
            return 

        for item in interactive_kde_lines:
            line_obj = item['line']
            if line_obj == picked_line_artist:
                line_obj.set_color(HIGHLIGHT_COLOR)
                line_obj.set_linewidth(HIGHLIGHT_LINEWIDTH)
                line_obj.set_alpha(HIGHLIGHT_ALPHA)
                line_obj.set_zorder(HIGHLIGHT_ZORDER)
            else:
                # Revert other individual KDE lines to their original style
                line_obj.set_color(item['original_color'])
                line_obj.set_linewidth(item['original_linewidth'])
                line_obj.set_alpha(item['original_alpha']) # Use stored original alpha
                line_obj.set_zorder(item['original_zorder'])
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    # --- End Pick Event Handler ---

    # --- Event Handler for Mouse Wheel Zoom ---
    def on_scroll(event):
        if event.inaxes != ax: # Process scroll only if cursor is over the axes
            return
        base_scale = 1.1 # Zoom sensitivity
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata 
        ydata = event.ydata

        if xdata is None or ydata is None: # If outside data limits, zoom to center
            xdata = (cur_xlim[0] + cur_xlim[1]) / 2
            ydata = (cur_ylim[0] + cur_ylim[1]) / 2

        if event.button == 'up': # Scroll up, zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down': # Scroll down, zoom out
            scale_factor = base_scale
        else: # Other scroll events
            return

        # New x-axis limits centered on mouse
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        ax.set_xlim([xdata - new_width * relx, xdata + new_width * (1 - relx)])

        # New y-axis limits centered on mouse
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
        ax.set_ylim([ydata - new_height * rely, ydata + new_height * (1 - rely)])

        ax.figure.canvas.draw_idle() # Redraw the plot

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    # --- End Mouse Wheel Zoom Handler ---

    # Setup mplcursors (for click tooltips)
    line_to_label_map = {item['line']: item['label'] for item in plotted_lines_info if item['line'] is not None}
    lines_for_cursor_interaction = [item['line'] for item in plotted_lines_info if item['line'] is not None]

    if lines_for_cursor_interaction:
        cursor = mplcursors.cursor(lines_for_cursor_interaction, hover=False) # hover=False for click
        
        @cursor.connect("add")
        def on_add(sel):
            label_for_line = line_to_label_map.get(sel.artist, "Unknown Line")
            # Customize tooltip text based on whether it's the average line or an individual file
            if average_line_handle and sel.artist == average_line_handle:
                 sel.annotation.set_text(f"{label_for_line}\nError: {sel.target[0]:.2f}\nAvg. Scaled Density: {sel.target[1]:.2f}")
            else:
                 sel.annotation.set_text(f"File: {label_for_line}\nError: {sel.target[0]:.2f}\nScaled Density: {sel.target[1]:.2f}")
            sel.annotation.get_bbox_patch().set(alpha=0.85, facecolor='lightyellow')
    
    # Legend (only for the average line, as individual lines are too many)
    if average_line_handle:
        ax.legend(handles=[average_line_handle], fontsize=12, loc='best')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_plot_full_path = os.path.join(output_plot_folder_path, combined_plot_filename)
    
    # try:
    #     plt.savefig(output_plot_full_path) 
    #     print(f"\nSUCCESS: Combined KDE plot saved to: {output_plot_full_path}")
    # except Exception as e_save:
    #     print(f"\nERROR: Failed to save combined KDE plot {output_plot_full_path}. Error: {e_save}")
    
    plt.show()
    # plt.close(fig) # Close the figure if running in a loop or non-interactively

else:
    print("\nNo datasets were successfully processed to generate a combined KDE plot.")

# This print statement was outside the if block, moved it for clarity or remove if not needed at the end
# print("\n--- All CSV files processed. ---")


##############################################################################################################
##############################################################################################################



"""Linear Regressions"""

# def process_single_csv(input_file, output_folder):
#     """
#     Processes a single CSV file, performs linear regression, 
#     and saves the results in the output folder.
#     """
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # Check if the input file exists
#     if not os.path.exists(input_file):
#         print(f"Error: {input_file} does not exist.")
#         return

#     print(f"Processing {input_file}...")

#     # Read the CSV file
#     data = pd.read_csv(input_file)

#     # Drop rows where 'Raster_Value' is blank
#     data = data.dropna(subset=['Raster_Value'])

#     # Perform linear regression
#     x = data[['Raster_Value']].values
#     y = data['Geoid_Corrected_Ortho_Height'].values
#     model = LinearRegression()
#     model.fit(x, y)

#     # Calculate regression metrics
#     y_pred = model.predict(x)
#     r2 = r2_score(y, y_pred)  # Scikit-learn R² calculation
#     rmse = np.sqrt(mean_squared_error(y, y_pred))

#     # Create the line of best fit equation
#     coef = model.coef_[0]
#     intercept = model.intercept_
#     equation = f"y = {coef:.4f}x + {intercept:.4f}"

#     # Calculate perpendicular distances
#     distances = np.abs(coef * x.flatten() - y + intercept) / np.sqrt(coef**2 + 1)

#     # Compute statistics for distances
#     min_dist = np.min(distances)
#     max_dist = np.max(distances)
#     mean_dist = np.mean(distances)
#     std_dist = np.std(distances)

#     # Append the results to the list
#     results = {
#         "Image Name": os.path.basename(input_file),
#         "R^2": r2,
#         "RMSE": rmse,
#         "Line of Best Fit": equation,
#         "m1": coef,
#         "m0": intercept,
#         "min perp dist": min_dist,
#         "max perp dist": max_dist,
#         "mean perp dist": mean_dist,
#         "std dev perp dist": std_dist
#     }

#     # Regression Plot
#     plt.figure(figsize=(8, 6))
#     plt.scatter(x, y, color='blue', alpha=0.7)
#     plt.plot(x, y_pred, color='red', linewidth=2)
    
#     plt.title('Sentinel-2 Linear Regression', fontsize=30)
    
#     plt.xlabel("pSDB values", fontsize=24)
#     plt.ylabel("Reference Depth (m)", fontsize=24)
#     plt.legend()
#     plt.grid(True)
    
#     #plt.xlim(.9, None)
#     #plt.ylim(None, 0.9)

#     # Add R^2 and RMSE as text on the plot
#     min_x = np.min(x)
#     max_y = np.max(y)
#     min_y = np.min(y)
#     mean_y = np.mean(y)
#     plt.text(min_x, 10, f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}", fontsize=18, color='black', ha='left',
#               bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

#     # Invert both axes so 0 is bottom left, and up and right are negative
#     #plt.gca().invert_xaxis()
#     #plt.gca().invert_yaxis()

#     # Save the regression plot in the output folder
#     plot_filename = f"{os.path.splitext(os.path.basename(input_file))[0]}_LR_plot_better.png"
#     plot_path = os.path.join(output_folder, plot_filename)
#     plt.savefig(plot_path)
#     plt.close()



# input_file = r"E:\Thesis Stuff\pSDB_ExtractedPts\SD_PlanetScope_2274_2022_06_03_21_10_58_L2W__RGB_pSDBred_extracted.csv"
# output_folder = r"E:\Thesis Stuff\pSDB_ExtractedPts"
# process_single_csv(input_file, output_folder)


##############################################################################################################
##############################################################################################################

""" Individual Error Histograms """

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import glob
# from scipy.stats import gaussian_kde

# # Specify the folder containing your input CSV files
# input_csv_folder_path = r"B:\Thesis Project\SDB_Time\Results_main\Hyannis\Sentinel-2\Extracted Pts\SDB\merged"

# # Specify the folder where you want to save the output plots
# output_plot_folder_path = r"B:\Thesis Project\SDB_Time\Results_main\Marathon\SuperDove\Final5\Extracted Pts\New folder\test_output" 
# # Define column names
# ref_col = "Geoid_Corrected_Ortho_Height"
# SDB_col = "Raster_Value"
# error_col_name = "Error" 

# #  Define a FIXED BIN WIDTH ---
# fixed_bin_width = 0.1  # Adjust this value based on your data's typical error range and desired resolution


# # Histogram plot settings
# hist_xlim = (-2, 2)
# hist_ylim = (0, 45) 

# # --- Create Output Folder if it Doesn't Exist ---
# if not os.path.exists(output_plot_folder_path):
#     os.makedirs(output_plot_folder_path)
#     print(f"Created output folder: {output_plot_folder_path}")

# # --- Find and Process CSV Files ---
# csv_files = glob.glob(os.path.join(input_csv_folder_path, "*.csv"))

# if not csv_files:
#     print(f"No CSV files found in the folder: {input_csv_folder_path}")
# else:
#     print(f"Found {len(csv_files)} CSV files to process in: {input_csv_folder_path}")

# for csv_file_path in csv_files:
#     print(f"\n--- Processing file: {csv_file_path} ---")
#     base_filename = os.path.basename(csv_file_path)
#     filename_no_ext = os.path.splitext(base_filename)[0]
#     fig = None # Initialize fig here to ensure it's defined for error handling

#     try:
#         df = pd.read_csv(csv_file_path)

#         if ref_col not in df.columns or SDB_col not in df.columns:
#             print(f"Warning: Required columns ('{ref_col}', '{SDB_col}') not found in {base_filename}. Skipping.")
#             continue

#         df[ref_col] = pd.to_numeric(df[ref_col], errors='coerce')
#         df[SDB_col] = pd.to_numeric(df[SDB_col], errors='coerce')
#         df = df.dropna(subset=[ref_col, SDB_col])

#         if df.empty:
#             print(f"Warning: No valid numeric data after coercion and NaN removal in {base_filename}. Skipping.")
#             continue

#         df[error_col_name] = df[ref_col] - df[SDB_col]
        
#         # Ensure there's data for histogram after error calculation and potential new NaNs
#         error_data_for_hist = df[error_col_name].dropna()
#         if error_data_for_hist.empty:
#             print(f"Warning: No error data to plot for {base_filename} after dropna. Skipping plot generation.")
#             continue


#         stats = error_data_for_hist.describe()
#         if error_data_for_hist.empty: # Should be caught by above, but good for safety
#             stats["RMSE"] = np.nan
#         else:
#             stats["RMSE"] = (error_data_for_hist.astype(float) ** 2).mean() ** 0.5

#         print(f"Statistics for {base_filename}:")
#         print(stats)

#         fig, ax = plt.subplots(figsize=(10, 6))
        
#         # --- MODIFICATION: Define bins based on fixed_bin_width ---
#         min_val = error_data_for_hist.min()
#         max_val = error_data_for_hist.max()
#         # Ensure bins cover the data range appropriately, even if min_val equals max_val
#         if np.isclose(min_val, max_val):
#             bins_array = np.array([min_val - fixed_bin_width/2, max_val + fixed_bin_width/2])
#         else:
#             bins_array = np.arange(min_val, max_val + fixed_bin_width, fixed_bin_width)
#         if len(bins_array) < 2: # Ensure at least one bin
#             bins_array = np.array([min_val - fixed_bin_width/2, max_val + fixed_bin_width/2])

#         counts, bin_edges, patches = ax.hist(error_data_for_hist, bins=bins_array, edgecolor='black', alpha=0.7, label=f'{error_col_name} Counts')
#         # --- END MODIFICATION ---
        
#         # Add KDE Plot
#         if len(error_data_for_hist) > 1: # KDE needs at least 2 points
#             try:
#                 # Create KDE
#                 kde = gaussian_kde(error_data_for_hist)
#                 # Create x-values for plotting the KDE curve (cover range of histogram)
#                 kde_x = np.linspace(bin_edges[0], bin_edges[-1], 200) # Use histogram bin edges for range
#                 kde_y = kde(kde_x)

#                 # Scale KDE to match histogram counts
#                 # (Area under KDE is 1, area under hist is N * bin_width)
#                 bin_width = bin_edges[1] - bin_edges[0]
#                 N = len(error_data_for_hist)
#                 scaled_kde_y = kde_y * N * bin_width

#                 # Plot KDE on the same axis
#                 ax.plot(kde_x, scaled_kde_y, color='red', linestyle='--', linewidth=2, label='KDE')
#             except Exception as e_kde:
#                 print(f"Warning: Could not compute or plot KDE for {base_filename}. Error: {e_kde}")
        
#         # --- ADD THIS TO CHECK BIN WIDTH ---
#         if len(bin_edges) > 1: # Check if there are at least two bin edges
#             bin_width_current_hist = bin_edges[1] - bin_edges[0]
#             print(f"Bin width for {base_filename}: \n{bin_width_current_hist:.4f}")
#         else:
#             print(f"Could not determine bin width for {base_filename} (not enough bin_edges).")
#             # --- END ADDITION ---
        
        
#         ax.set_xlabel(f"{error_col_name} (m)", fontsize=14)
#         ax.set_ylabel("Count", fontsize=14)
#         plot_title = f"{filename_no_ext.replace('_', ' ')} Error Distribution"
#         ax.set_title(plot_title, fontsize=16)
#         ax.grid(axis='y', linestyle='--', alpha=0.7)
        
#         ax.set_xlim(hist_xlim)
#         if hist_ylim:
#             ax.set_ylim(hist_ylim)
#         else:
#             # Dynamic y-limit if hist_ylim is not set or is (0,0) which can happen
#             max_y_val = 0
#             if len(counts) > 0:
#                 max_y_val = max(counts)
#             if 'scaled_kde_y' in locals() and len(scaled_kde_y) > 0 : # Check if KDE was plotted
#                   max_y_val = max(max_y_val, np.max(scaled_kde_y))
#             ax.set_ylim(0, max_y_val * 1.1 if max_y_val > 0 else 10)


#         stats_text = (f"Mean = {stats.get('mean', np.nan):.3f} m\n"
#                       f"Min = {stats.get('min', np.nan):.2f} m\n"
#                       f"Max = {stats.get('max', np.nan):.2f} m\n"
#                       f"Std Dev = {stats.get('std', np.nan):.2f} m\n"
#                       f"RMSE = {stats.get('RMSE', np.nan):.2f} m\n"
#                       f"Count = {stats.get('count', 0):.0f}")
        
#         current_text_x = hist_xlim[0] + (hist_xlim[1] - hist_xlim[0]) * 0.05
#         current_text_y = (ax.get_ylim()[1]) * 0.95
        
#         ax.text(current_text_x, current_text_y, stats_text, fontsize=10, color='black', ha='left', va='top',
#                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        
#         ax.legend() # Add legend to show histogram and KDE labels
#         plt.tight_layout()

#         output_plot_filename = f"{filename_no_ext}_error_histogram_kde.png" 
#         output_plot_full_path = os.path.join(output_plot_folder_path, output_plot_filename)
        
#         #plt.savefig(output_plot_full_path) # Uncommented this
#         #print(f"Saved plot to: {output_plot_full_path}")

#         plt.show() 
#         plt.close(fig)

#     except FileNotFoundError:
#         print(f"Error: File not found at {csv_file_path}. Skipping.")
#     except pd.errors.EmptyDataError:
#         print(f"Warning: CSV file {base_filename} is empty. Skipping.")
#     except KeyError as e:
#         print(f"Warning: Column missing in {base_filename} - {e}. Skipping.")
#     except Exception as e:
#         print(f"An error occurred while processing {csv_file_path}: {e}")
#         if fig and plt.fignum_exists(fig.number): # Ensure fig exists before trying to close
#             plt.close(fig)

# print("\n--- All CSV files processed. ---")


##############################################################################################################
##############################################################################################################



""" Single Histogram chart for viewing multiple SuperDove and Sentinel-2 error values for a single site """


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd # For reading CSV files
# import os # For path manipulation
# import glob # For finding files

# # --- 1. Configuration ---
# # Specify the folder containing your CSV files
# csv_folder_path = r"B:\Thesis Project\SDB_Time\Results_main\Marathon\SuperDove\Final5\Extracted Pts\New folder"  # <--- !!! REPLACE THIS !!!

# # Specify the column name in your CSV files that contains the error values
# error_column_name = "Error"  # <--- !!! REPLACE THIS if your column name is different !!!

# # Optional: Define colors and linestyles if you have a fixed number of expected files
# # or want to cycle through them.
# colors = ['blue', 'darkorange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (5, 1)), (0, (1, 1))]

# # --- 2. Load Data from CSV Files ---
# all_error_data = [] # To store data from all files for common bin calculation
# data_to_plot = []   # To store tuples of (label, error_values_array)

# csv_files = glob.glob(os.path.join(csv_folder_path, "*.csv"))

# if not csv_files:
#     print(f"No CSV files found in the folder: {csv_folder_path}")
#     # You might want to exit or raise an error here if no files are found
#     # For now, it will just produce an empty plot.
# else:
#     print(f"Found {len(csv_files)} CSV files to process in: {csv_folder_path}")

# for i, file_path in enumerate(csv_files):
#     try:
#         df = pd.read_csv(file_path)
#         if error_column_name in df.columns:
#             # Remove NaNs and convert to numpy array
#             error_values = df[error_column_name].dropna().values.astype(float)
#             if error_values.size > 0:
#                 all_error_data.append(error_values)
#                 # Create a label from the filename (without extension)
#                 file_name = os.path.basename(file_path)
#                 label = os.path.splitext(file_name)[0]
#                 data_to_plot.append((label, error_values))
#                 print(f"Loaded {len(error_values)} error values from '{file_name}' (column: '{error_column_name}').")
#             else:
#                 print(f"Warning: No valid error values found in column '{error_column_name}' in file '{file_path}'.")
#         else:
#             print(f"Warning: Column '{error_column_name}' not found in file '{file_path}'. Skipping this file.")
#     except Exception as e:
#         print(f"Error processing file '{file_path}': {e}")

# # --- 3. Plot Histograms as Outlines (Steps) ---
# if data_to_plot: # Only proceed if there's data to plot
#     plt.figure(figsize=(12, 7)) # Adjust figure size if needed

#     # Determine common bins based on all loaded data
#     if all_error_data:
#         concatenated_errors = np.concatenate(all_error_data)
#         min_bin = 0.0 # RMSE/errors are typically non-negative
#         # Consider a robust max_bin, e.g., percentile, or just max if outliers are okay
#         max_bin = np.percentile(concatenated_errors, 99.5) if concatenated_errors.size > 0 else 1.0
#         if max_bin <= min_bin: # handle cases with very little data or constant values
#             max_bin = min_bin + 1.0
#         num_bins = 30 # Adjust number of bins as desired
#         bins = np.linspace(min_bin, max_bin, num_bins + 1)
#     else: # Fallback if no data was loaded at all (shouldn't happen if data_to_plot is not empty)
#         bins = np.linspace(0, 1, num_bins + 1)


#     # Plot histograms using histtype='step'
#     for i, (label, error_values) in enumerate(data_to_plot):
#         plt.hist(error_values, bins=bins,
#                   #density=True,          # Use density for fair comparison if sample sizes vary
#                   histtype='step',       # Draw outline
#                   linewidth=1.5,         # Line thickness
#                   linestyle=linestyles[i % len(linestyles)], # Cycle through linestyles
#                   color=colors[i % len(colors)],             # Cycle through colors
#                   label=f'{label} (N={len(error_values)})')

#     # Add plot labels and title
#     plt.xlabel(f"{error_column_name} Values", fontsize=12) # Use the actual column name
#     plt.ylabel("Point Count", fontsize=12)
#     plt.title(f"Distribution of {error_column_name} Values (Outlines)", fontsize=14)
#     plt.legend(fontsize=10)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout() # Adjust layout
#     plt.show()
# else:
#     print("No data loaded to plot.")




##############################################################################################################
##############################################################################################################

"""
Creates box and whisker plots out of all the RMSE values (1 for each image) for each AOI, and overlays all R^2
values from each regression line over each box plot. The purpose of this is to visualize not only the R^2 values 
(which can't tell you everything anyway), but the RMSE as well to get a better picture. 


"""

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib.cm as cm

# # --- 1. Sample Data Creation (Replace this with loading your actual data) ---
# np.random.seed(42)
# sites = [f'Site_{i}' for i in range(1, 4)]
# sensors = ['Sentinel-2', 'SuperDove']
# algorithms = ['pSDBred', 'pSDBgreen']
# data_list = []
# for site in sites:
#     for sensor in sensors:
#         n_images = 5 if sensor == 'Sentinel-2' else 20
#         for algo in algorithms:
#             for i in range(n_images):
#                 base_rmse = 0.8 if sensor == 'Sentinel-2' else 1.1
#                 base_r2 = 0.85 if sensor == 'Sentinel-2' else 0.75
#                 rmse = max(0.1, base_rmse + np.random.randn() * 0.3 + (0.1 if algo == 'pSDBred' else -0.1))
#                 r2 = min(0.99, max(0.1, base_r2 + np.random.randn() * 0.1 + (0.05 if algo == 'pSDBred' else -0.05)))
#                 data_list.append({'Site': site, 'Sensor': sensor, 'Algorithm': algo, 'RMSE': rmse, 'R2': r2})
# results_df = pd.DataFrame(data_list)
# results_df['Group'] = results_df['Sensor'] + ' - ' + results_df['Algorithm'] # Example grouping
# print("Sample DataFrame head:")
# print(results_df.head())
# # --- End Sample Data Creation ---


# # --- 2. Prepare Data for Matplotlib Boxplot ---
# grouping_variable = 'Group' # Column to group by on x-axis
# y_metric = 'RMSE'           # Column for boxplot y-axis
# color_metric = 'R2'         # Column to map to color

# unique_groups = sorted(results_df[grouping_variable].unique())
# # Data for boxplot: list of arrays, one array of RMSE values per group
# data_to_plot = [results_df.loc[results_df[grouping_variable] == grp, y_metric].dropna().values
#                 for grp in unique_groups]
# # X positions for the boxes
# box_positions = np.arange(len(unique_groups)) + 1


# # --- 3. Create the Plot ---
# fig, ax = plt.subplots(figsize=(10, 7)) # Adjust size as needed

# # Create the box plots
# # patch_artist=True allows filling boxes with color (optional)
# # showfliers=False prevents boxplot from drawing outliers, scatter will show them
# bp = ax.boxplot(data_to_plot, positions=box_positions, showfliers=False, patch_artist=True,
#                 boxprops=dict(facecolor='lightblue', alpha=0.6), # Example box styling
#                 medianprops=dict(color='red', linewidth=1.5),
#                 whiskerprops=dict(color='blue'),
#                 capprops=dict(color='blue'))

# # --- 4. Prepare for Scatter Overlay ---
# # Create a mapping from group name to x position
# group_to_xpos = {group: pos for group, pos in zip(unique_groups, box_positions)}
# # Get x position for each row in the original dataframe
# x_scatter = results_df[grouping_variable].map(group_to_xpos)

# # Add jitter (small random horizontal shift) to x positions to reduce overlap
# jitter_amount = 0.08
# #x_scatter_jitter = x_scatter + np.random.normal(0, jitter_amount, size=len(results_df))

# # Get the y values and color values for the scatter plot
# y_scatter = results_df[y_metric]
# color_values = results_df[color_metric]

# # --- 5. Normalize R2 values and choose colormap ---
# norm = mcolors.Normalize(vmin=color_values.min(), vmax=color_values.max())
# cmap = cm.RdYlGn # Choose a colormap (e.g., viridis, plasma, coolwarm)

# # --- 6. Plot Scatter Overlay ---
# scatter_plot = ax.scatter(x_scatter, y_scatter,
#                           c=color_values, # Use R2 values for color
#                           cmap=cmap,      # Apply the colormap
#                           norm=norm,      # Apply the normalization
#                           alpha=0.9,      # Point transparency 
#                           #edgecolor='k',
#                           s=50,           # Adjust marker size if needed
#                           zorder=3)       # Draw scatter on top of boxes

# # --- 7. Add Color Bar ---
# cbar = fig.colorbar(scatter_plot, ax=ax)
# cbar.set_label(f'{color_metric} Value', rotation=270, labelpad=15)

# # --- 8. Customize and Show Plot ---
# ax.set_xticks(box_positions) # Set ticks at the box positions
# ax.set_xticklabels(unique_groups, rotation=45, ha='right') # Set group names as labels
# ax.set_title(f'Distribution of {y_metric} with points colored by {color_metric}', fontsize=16)
# ax.set_xlabel("Group", fontsize=12)
# ax.set_ylabel(y_metric, fontsize=12)
# ax.grid(axis='y', linestyle='--', alpha=0.6)

# plt.tight_layout() # Adjust layout
# plt.show()








