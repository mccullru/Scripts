%%
clc; clear; close all;

data = readmatrix("B:\Thesis Project\StatsAndFigures\Available_and_Usable_imgs\available_vs_usable_data.csv");
AOI_label = {'Bum Bum (4.5°)'; 'Nait (12.6°)'; 'Anegada (18.7°)'; 'Marathon (24.7°)'; 'North Fuerteventura (28.7°)'; 'Bombah (32.4°)';
    'Gyali (36.6°)'; 'South Port (39.2°)'; 'Hyannis (41.6°)'; 'Punta (45.7°)';  'Dingle (52°)'; 'Rago (55°)'; 'Homer (59.6°)'; 'Skutvik (68°)'; 'Risoysundet (69°)'};

%%
% Data column order for sd_data: [Good_col, Ok_col, Bad_col, Unusable_col]
sd_data_all = data(:,[4,3,2,5]); % SD: Good(D), Ok(C), Bad(B), Unusable(E)
s2_data_all = data(:,[8,7,6,9]); % S2: Good(H), Ok(G), Bad(F), Unusable(I)


n = size(data,1); % Make n dynamic
grouped_data = zeros(n*2, 4); 
grouped_data(1:2:end, :) = sd_data_all;
grouped_data(2:2:end, :) = s2_data_all;

% Plot
figure;
hb = bar(grouped_data, 'stacked'); % hb(1) is bottom (Good), ..., hb(4) is top (Unusable)

% Define colors to match legend and "Good is darkest" preference
sd_colors = [1 0.6 0.6;  % 1 (SD) (Bad) - Light Red
             1 0.2 0.2;  % 2 (SD) (Ok)  - Medium Red
             0.6 0 0;    % 3 (SD) (Good)- Dark Red
             0.7 0.7 0.7]; % unusable (SD) - Gray

s2_colors = [0.6 0.8 1;  % 1 (S2) - Light Blue
             0.2 0.6 1;  % 2 (S2)  - Medium Blue
             0 0 0.6;    % 3 (S2)- Dark Blue
             0.7 0.7 0.7]; % unusable (S2) - Gray

% Map stack components to correct colors ---
% Stack components (hb(k)): k=1 is bottom (Good data), k=2 (Ok data), k=3 (Bad data), k=4 (Unusable data)
% (where index 1=Bad, 2=Ok, 3=Good, 4=Unusable based on legend_labels order)
color_indices_for_stack = [3, 2, 1, 4]; % legend-based color index (3=Good, 2=Ok, 1=Bad, 4=Unusable)
                                        
for k = 1:4 
    bars = hb(k);
    CData = zeros(2*n, 3);
    
    actual_color_index = color_indices_for_stack(k);

    for i = 1:n
        CData(2*i-1,:) = sd_colors(actual_color_index,:);
        CData(2*i,:)   = s2_colors(actual_color_index,:);
    end
    bars.FaceColor = 'flat';
    bars.CData = CData;
end

% X-axis setup
xtick_positions = 1.5:2:(2*n);
xticks(xtick_positions);
xticklabels(AOI_label); 
xtickangle(45);
ylabel('Images per Year', 'FontSize',14);
xlabel('AOI (latitude)', 'FontSize',14);
title('Imagery Rates with Usability Index', 'FontSize',16);
grid on;

% Create dummy objects for the unified legend
% This part should correctly use the sd_colors/s2_colors as defined for the legend
dummy = gobjects(8,1); 

% SD Legend 
for k = 1:4 % k=1 for '1 (SD)', k=2 for '2 (SD)', k=3 for '3 (SD)', k=4 for 'unusable (SD)'
    dummy(k) = patch(NaN, NaN, sd_colors(k,:));
end

% S2 Legend
for k = 1:4
    dummy(k+4) = patch(NaN, NaN, s2_colors(k,:));
end

% Unified Legend
legend_labels = {' 1 (SD)', ' 2 (SD)', ' 3 (SD)', 'Unusable (SD)', ...
                 ' 1 (S2)', ' 2 (S2)', ' 3 (S2)', 'Unusable (S2)'}; 
lgd = legend(dummy, legend_labels, 'Location', 'northeastoutside', 'NumColumns', 2, 'FontSize',14);
lgd.Title.String = ["Usability Index", "(3=Good; 2=Ok; 1=Bad)"];
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Reorder data columns for stacking where bottom is good and top is bad
sd_data = data(:,[4,3,2]); % Assuming col 4 is Good, 3 is Ok, 2 is Bad for SD
s2_data = data(:,[8,7,6]); % Assuming col 8 is Good, 7 is Ok, 6 is Bad for S2


n = size(data,1); % Make n dynamic based on number of rows in original data

grouped_data = zeros(n*2, 3);
grouped_data(1:2:end, :) = sd_data; % sd_data now [Good_counts, Ok_counts, Bad_counts]
grouped_data(2:2:end, :) = s2_data; % s2_data now [Good_counts, Ok_counts, Bad_counts]

% Plot
figure;
hb = bar(grouped_data, 'stacked'); % hb(1) is bottom (Good), hb(2) middle (Ok), hb(3) top (Bad)

% Define colors to match legend
sd_colors = [1 0.6 0.6;  % 1 (SD) - Light Red
             1 0.2 0.2;  % 2 (SD) - Medium Red
             0.6 0 0];   % 3 (SD) - Dark Red

s2_colors = [0.6 0.8 1;  % 1 (S2) - Light Blue
             0.2 0.6 1;  % 2 (S2) - Medium Blue
             0 0 0.6;];  % 3 (S2) - Dark Blue


% Map stack components to correct colors
% Stack components (hb(k)): k=1 is bottom (Good data), k=2 is middle (Ok data), k=3 is top (Bad data)
% We need to map these to the correct color index from sd_colors (where index 1=Bad, 2=Ok, 3=Good)
color_indices_for_stack = [3, 2, 1]; % To map [Good, Ok, Bad] stack to [color_for_3, color_for_2, color_for_1]

for k = 1:3 % k refers to the component of the stack (1=bottom, 2=middle, 3=top)
    bars = hb(k); % hb(1) are the bottom segments, hb(2) middle, hb(3) top
    CData = zeros(2*n, 3);
    
    actual_color_index = color_indices_for_stack(k); % Get color for Good when k=1, Ok for k=2, Bad for k=3

    for i = 1:n
        CData(2*i-1,:) = sd_colors(actual_color_index,:);
        CData(2*i,:)   = s2_colors(actual_color_index,:);
    end
    bars.FaceColor = 'flat';
    bars.CData = CData;
end

% X-axis setup
xtick_positions = 1.5:2:(2*n);
xticks(xtick_positions);
xticklabels(AOI_label); % Ensure AOI_label is defined and has 'n' elements
xtickangle(45);
ylabel('Images Per Year');
xlabel('AOI (latitude)');
title('Imagery Rates with Usability Index');
grid on;

% Create dummy objects for the unified legend
% This part correctly uses sd_colors where sd_colors(1) is for Bad, (2) for Ok, (3) for Good
dummy = gobjects(6,1);
% SD Legend (first 3 entries)
for k = 1:3 % k=1 maps to label '1 (SD)', k=2 to '2 (SD)', k=3 to '3 (SD)'
    dummy(k) = patch(NaN, NaN, sd_colors(k,:));
end
% S2 Legend (next 3 entries)
for k = 1:3
    dummy(k+3) = patch(NaN, NaN, s2_colors(k,:));
end

% Unified Legend
legend_labels = {' 1 (SD)', ' 2 (SD)', ' 3 (SD)', ' 1 (S-2)', ' 2 (S-2)', ' 3 (S-2)'};
lgd = legend(dummy, legend_labels, 'Location', 'northeastoutside', 'NumColumns', 2);
lgd.Title.String = {"Usability Index", "(3=Good; 2=Ok; 1=Bad)"};

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Only use rating 3 data
sd_data = data(:,4);  % column for rating 3 (SD)
s2_data = data(:,8);  % column for rating 3 (S2)

n = 15;

% Combine into a 15x2 matrix: [SD, S2]
grouped_data = [sd_data, s2_data];

% Plot grouped bar chart
figure;
hb = bar(grouped_data, 'grouped');

% Set bar colors manually
sd_color = [0.6 0 0];
s2_color = [0 0 0.6];
hb(1).FaceColor = sd_color;
hb(2).FaceColor = s2_color;

% X-axis labels and formatting
xticks(1:n);
xticklabels(AOI_label);
xtickangle(45);
ylabel('Images per Year', 'FontSize',13);
xlabel('AOI (latitude)', 'FontSize',13);
title('Total Best Quality Images', 'FontSize',15);
grid on;

lgd = legend({'SD', 'S-2'}, 'Location', 'northeastoutside', 'FontSize',14);
lgd.Title.String = '3-Rated Imgs on Usability Index';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%


% --- 1. Load Data from CSV file ---
% IMPORTANT: Replace 'usability_data.csv' with the actual path to your data file.
% The CSV file must contain columns with the following headers for the script to work:
% SD_1, SD_2, SD_3, SD_all, S2_1, S2_2, S2_3, S2_all

try
    filename = "B:\Thesis Project\StatsAndFigures\Available_and_Usable_imgs\available_vs_usable_data.csv"; % <-- REPLACE WITH YOUR CSV FILE NAME/PATH
    data = readtable(filename);
    fprintf('Successfully loaded data from %s\n\n', filename);
catch ME
    if strcmp(ME.identifier, 'MATLAB:readtable:FileNotFound')
        error('Data file not found. Please make sure the file "%s" is in the correct path and update the filename variable in the script.', filename);
    else
        rethrow(ME);
    end
end

% --- Conditional Data Filtering ---
% Set the variable below to true or false to control whether the Dingle
% site is included in the final average calculations.
includeDingle = false; % <-- SET TO false TO EXCLUDE DINGLE

if ~includeDingle
    % Find the row corresponding to the 'Dingle' AOI
    % Note: Assumes the AOI column is named 'AOI' in your CSV
    if ismember('AOI', data.Properties.VariableNames)
        dingleRow = strcmp(data.AOI, 'Dingle');
        data(dingleRow, :) = []; % Remove the row
        fprintf('Excluding Dingle site from analysis.\n\n');
    else
        warning('AOI column not found. Cannot exclude Dingle site.');
    end
else
    fprintf('Including Dingle site in analysis.\n\n');
end


% --- 2. Calculate Percentage for each AOI ---
% For each row (AOI), calculate the percentage of usable images for each
% rating (1, 2, 3) relative to the total number of images for that sensor.

% SuperDove (SD) Percentages
data.SD_1_percent = (data.SD_1 ./ data.SD_all) * 100;
data.SD_2_percent = (data.SD_2 ./ data.SD_all) * 100;
data.SD_3_percent = (data.SD_3 ./ data.SD_all) * 100;

% Sentinel-2 (S2) Percentages
data.S2_1_percent = (data.S2_1 ./ data.S2_all) * 100;
data.S2_2_percent = (data.S2_2 ./ data.S2_all) * 100;
data.S2_3_percent = (data.S2_3 ./ data.S2_all) * 100;

% --- 3. Compute the Average Percentages ---
% Calculate the mean of the percentage columns.

avg_SD_1_percent = mean(data.SD_1_percent);
avg_SD_2_percent = mean(data.SD_2_percent);
avg_SD_3_percent = mean(data.SD_3_percent);

avg_S2_1_percent = mean(data.S2_1_percent);
avg_S2_2_percent = mean(data.S2_2_percent);
avg_S2_3_percent = mean(data.S2_3_percent);

% --- 4. Create the Summary Table ---
% Create a new table to display the final summarized results.

Platform = {'Sentinel-2'; 'SuperDove'};
Usability_1_Rate = [avg_S2_1_percent; avg_SD_1_percent];
Usability_2_Rate = [avg_S2_2_percent; avg_SD_2_percent];
Usability_3_Rate = [avg_S2_3_percent; avg_SD_3_percent];

% Create the summary table with raw averages
summaryTable = table(Platform, Usability_1_Rate, Usability_2_Rate, Usability_3_Rate);


% --- 5. Save and Display the Final Results ---

% Save the summary table to a new CSV file
[inputPath, ~, ~] = fileparts(filename); % Get the directory of the input file
outputFilename = 'usability_rate_summary_noDingle.csv';
outputFilePath = fullfile(inputPath, outputFilename);

try
    writetable(summaryTable, outputFilePath);
    fprintf('Summary table successfully saved to: %s\n\n', outputFilePath);
catch ME_write
    fprintf('ERROR: Could not save summary table to %s.\n', outputFilePath);
    fprintf('Reason: %s\n', ME_write.message);
end

% Display the table in the command window with formatted percentages
fprintf('Usability Rate Summary:\n\n');
fprintf('%-12s | %-25s | %-25s | %-25s\n', 'Platform', 'Usability 1 Rate (% of total)', 'Usability 2 Rate (% of total)', 'Usability 3 Rate (% of total)');
fprintf('--------------------------------------------------------------------------------------------------\n');
for i = 1:height(summaryTable)
    fprintf('%-12s | %22.0f%% | %22.0f%% | %22.0f%%\n', ...
        summaryTable.Platform{i}, ...
        round(summaryTable.Usability_1_Rate(i)), ...
        round(summaryTable.Usability_2_Rate(i)), ...
        round(summaryTable.Usability_3_Rate(i)));
end

































