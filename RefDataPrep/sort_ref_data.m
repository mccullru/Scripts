
clc
clear

% Reads in multiple LiDAR files, creates a grid over each file with cells
% of a provided size, chooses one point at random within each cell, saves
% that point and its information, then saves the combination of all LiDAR
% files provided into one file. The purpose is to downsample the very large
% LiDAR files that will be used as reference data for creating SDB (for
% either calibration or accuracy checks) into fewer randomly chosen points
% that are equally distributed throughout the area. 



% % Define the input directory containing the LiDAR files
% inputFolder = "B:\Thesis Project\Reference Data\Hyannis Harbour, MA\2018";  % Change this to your actual folder path
% fileExtension = '*.txt';  % Specify the file extension of the LiDAR files
% 
% fileList = dir(fullfile(inputFolder, fileExtension));
% 
% 
% % Initialize a cell array to store all selected points
% allSelectedPoints = [];  % An empty array to hold all selected points
% 
% % Loop through each LiDAR file in the folder
% for i = 1:length(fileList)
%     tic
%     
%     % Construct the full file path
%     inputFile = fullfile(fileList(i).folder, fileList(i).name);
%     
%     % Read the LiDAR file, skipping the header (first row)
%     file = readmatrix(inputFile, 'NumHeaderLines', 1);
% 
%     % Extract X, Y, Z coordinates
%     X = file(:,1);
%     Y = file(:,2);
%     Z = file(:,3);
% 
%     % Define the grid cell size
%     cellSize = 100;
% 
%     % Determine the minimum and maximum X and Y values to set grid extents
%     minX = min(X);
%     maxX = max(X);
%     minY = min(Y);
%     maxY = max(Y);
% 
%     % Create the grid indices for each point
%     rowIndex = floor((X - minX) / cellSize) + 1;  % Row index based on X coordinates
%     colIndex = floor((Y - minY) / cellSize) + 1;  % Column index based on Y coordinates
% 
%     % Create a unique identifier for each grid cell
%     gridIDs = rowIndex + 1i * colIndex;  % Use complex numbers to combine row and col
% 
%     % Identify unique grid cells
%     uniqueGridIDs = unique(gridIDs);
% 
%     % Initialize an array to store the selected points for this file
%     selectedPoints = zeros(length(uniqueGridIDs), 3);
% 
%     % Loop over each grid cell and randomly select one point
%     for j = 1:length(uniqueGridIDs)
%         % Find points in the current grid cell
%         pointsInCell = (gridIDs == uniqueGridIDs(j));
% 
%         % Randomly select one point from the cell
%         pointIdx = find(pointsInCell);
%         randomIdx = pointIdx(randi(length(pointIdx)));  % Randomly select index
% 
%         % Store the selected point (X, Y, Z)
%         selectedPoints(j, :) = [X(randomIdx), Y(randomIdx), Z(randomIdx)];
%     end
% 
%     % Concatenate selected points from this file into the overall list
%     allSelectedPoints = [allSelectedPoints; selectedPoints];  % Combine points into a single array
% 
%     disp(['Processed file: ' inputFile]);
% 
%     toc
% end
% 
% % Clip all the points to selected goemetry
% 
% polygon = "B:\Thesis Project\AOIs\Hyannis Harbour, MA\HyannisAOI.geojson";
% 
% % Read the GeoJSON file
% geometry_Data = fileread(polygon);
% 
% % Parse the GeoJSON
% geojson = jsondecode(geometry_Data);
% 
% % Extract coordinates of the polygon (assuming one polygon in GeoJSON)
% polygonCoordinates = geojson.features.geometry.coordinates;
% 
% % Convert to X and Y arrays (adjust for nested arrays if needed)
% polygonX = polygonCoordinates(:,1);  % Extract X (longitude)
% polygonY = polygonCoordinates(:,2);  % Extract Y (latitude)
% 
% Easting = allSelectedPoints(:,1);
% Northing = allSelectedPoints(:,2);
% 
% [isInside, ~] = inpolygon(Easting, Northing, polygonX, polygonY);
% 
% clippedPoints = allSelectedPoints(isInside, :);
% 
% 
% 
% % Define the output file name for combined output
% outputFile = fullfile(inputFolder, 'combined_and_downsampled_points.csv');  % Output file name
% 
% % Write the header (assuming the original header is "X Y Z")
% header = {'Easting(m)', 'Northing(m)', 'Elevation(m)'};  % Modify this to match the header format in your input files
% 
% writematrix(allSelectedPoints, outputFile, 'WriteMode','overwrite','Delimiter', ',');
% 
% 
% % Open the output file for writing
% fileID = fopen(outputFile, 'r+');
% content = fread(fileID, '*char')';
% fseek(fileID, 0, 'bof');
% fprintf(fileID, '%s\n', strjoin(header, ','));  % Write the header to the file
% fwrite(fileID, content);
% fclose(fileID);
% 
% disp(['All selected points have been combined and saved to ' outputFile]);


%{
Splits data into a calibration dataset (70%) and an accuraccy assessment 
dataset (30%). Uses all input points, and places each point in a bin
which corresponds to a depth range so points are stratified. 
%}


% Define the input folder containing the CSV files
inputFolder = "E:\Thesis Stuff\ReferenceData\Topobathy";  % Change to your folder path

% Define the output folder
outputFolder = "E:\Thesis Stuff\ReferenceData\Topobathy";  % Change to your desired output folder path

% Get list of all CSV files in the folder
fileList = dir(fullfile(inputFolder, '*.csv'));

% Loop through each file
for k = 1:length(fileList)
    % Read the CSV file
    inputFile = fullfile(fileList(k).folder, fileList(k).name);
    data = readmatrix(inputFile);
    
    % Extract columns (assuming order: Easting, Northing, Elevation)
    Easting = data(:,1);
    Northing = data(:,2);
    Elevation = data(:,3);
    
    % Define number of bins (adjust based on depth range)
    numBins = 10;  % Example: 10 bins
    [~, edges] = histcounts(Elevation, numBins);  % Get bin edges
    binIdx = discretize(Elevation, edges);  % Assign each point to a bin
    
    % Initialize training and calibration datasets
    calData = [];
    accData = [];

    % Stratified splitting: Loop through each depth bin
    for i = 1:numBins
        % Get indices of points in the current bin
        binPoints = find(binIdx == i);
        
        % Shuffle indices to ensure randomness
        binPoints = binPoints(randperm(length(binPoints)));
        
        % Determine split sizes (70% Accuracy, 30% Calibration)
        numTrain = round(0.7 * length(binPoints));
        
        % Select points for each subset
        calData = [calData; Easting(binPoints(1:numTrain)), Northing(binPoints(1:numTrain)), Elevation(binPoints(1:numTrain))];
        accData = [accData; Easting(binPoints(numTrain+1:end)), Northing(binPoints(numTrain+1:end)), Elevation(binPoints(numTrain+1:end))];
    end
    
    % Create new file names by appending suffixes
    [~, baseName, ext] = fileparts(fileList(k).name);
    calFileName = fullfile(outputFolder, [baseName, '_calibration_pts', ext]);
    accFileName = fullfile(outputFolder, [baseName, '_accuracy_pts', ext]);

    % Save stratified datasets to CSV
    writematrix(calData, calFileName);
    writematrix(accData, accFileName);
    
    disp(['Processed file: ', fileList(k).name]);
end

disp('Stratified splitting complete for all files: 70% Accuracy, 30% Calibration.');






















