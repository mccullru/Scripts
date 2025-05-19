clc; clear; close all;

data = readmatrix("B:\Thesis Project\StatsAndFigures\Available_and_Usable_imgs\available_vs_usable_data.csv");
AOI_label = {'Bum Bum (4.5°)'; 'Nait (12.6°)'; 'Anegada (18.7°)'; 'Marathon (24.7°)'; 'North Fuerteventura (28.7°)'; 'Bombah (32.4°)';
    'Gyali (36.6°)'; 'South Port (39.2°)'; 'Hyannis (41.6°)'; 'Punta (45.7°)';  'Dingle (52°)'; 'Rago (55°)'; 'Homer (59.6°)'; 'Skutvik (68°)'; 'Risoysundet (69°)'};


sd_data = data(:,2:5);
s2_data = data(:,6:9);

n = 15;

grouped_data = zeros(n*2, 4);
grouped_data(1:2:end, :) = sd_data;
grouped_data(2:2:end, :) = s2_data;

% Plot
figure;
hb = bar(grouped_data, 'stacked');

% Define SD (reds) and S2 (blues)
sd_colors = [0.6 0 0; 1 0.2 0.2; 1 0.6 0.6;.7 .7 .7];    % .7 .7 .7 for light gray
s2_colors = [0 0 0.6; 0.2 0.6 1; 0.6 0.8 1;.7 .7 .7];      

% Assign per-bar color
for k = 1:4
    bars = hb(k);
    CData = zeros(2*n, 3);
    for i = 1:n
        CData(2*i-1,:) = sd_colors(k,:);
        CData(2*i,:)   = s2_colors(k,:);
    end
    bars.FaceColor = 'flat';
    bars.CData = CData;
end

% X-axis setup
xtick_positions = 1.5:2:(2*n);
xticks(xtick_positions);
xticklabels(AOI_label);
xtickangle(45);
ylabel('Count');
xlabel('AOI (latitude)');
title('Rates of Usable Imagery');
grid on;

% Create dummy objects for the unified legend
dummy = gobjects(8,1);

% SD Legend (first 4 entries)
for k = 1:4
    dummy(k) = patch(NaN, NaN, sd_colors(k,:));
end

% S2 Legend (next 4 entries)
for k = 1:4
    dummy(k+4) = patch(NaN, NaN, s2_colors(k,:));
end

% Unified Legend
legend_labels = {' 1 (SD)', ' 2 (SD)', ' 3 (SD)', 'unusable (SD)', ' 1 (S2)', ' 2 (S2)', ' 3 (S2)', 'unusable (S2)'};
lgd = legend(dummy, legend_labels, 'Location', 'northeastoutside');
lgd.Title.String = "Img Ratings";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- MODIFICATION 1: Reorder data columns for stacking ---
% Original: sd_data column 1 = Bad(1), col 2 = Ok(2), col 3 = Good(3)
% New order for sd_data (and s2_data) columns: Good(3), Ok(2), Bad(1)
% This means the first column passed to bar() will be 'Good', stacking at the bottom.
sd_data = data(:,[4,3,2]); % Assuming col 4 is Good, 3 is Ok, 2 is Bad for SD
s2_data = data(:,[8,7,6]); % Assuming col 8 is Good, 7 is Ok, 6 is Bad for S2
% --- END MODIFICATION 1 ---

n = size(data,1); % Make n dynamic based on number of rows in original data
                  % or use the '15' if it's fixed for this specific plot
grouped_data = zeros(n*2, 3);
grouped_data(1:2:end, :) = sd_data; % sd_data now [Good_counts, Ok_counts, Bad_counts]
grouped_data(2:2:end, :) = s2_data; % s2_data now [Good_counts, Ok_counts, Bad_counts]

% Plot
figure;
hb = bar(grouped_data, 'stacked'); % hb(1) is bottom (Good), hb(2) middle (Ok), hb(3) top (Bad)

% --- MODIFICATION 2: Define colors to match legend and "Good is darkest" preference ---
% Legend Title: (3=Good; 2=Ok; 1=Bad)
% We want legend '3 (SD)' (Good) to be Dark Red.
% We want legend '1 (SD)' (Bad) to be Light Red.
sd_colors = [0.6 0 0;  % Color for legend entry '1 (SD)' (Bad) - Light Red
             1 0.2 0.2;  % Color for legend entry '2 (SD)' (Ok)  - Medium Red
             1 0.6 0.6];   % Color for legend entry '3 (SD)' (Good)- Dark Red

s2_colors = [0 0 0.6;  % Color for legend entry '1 (S2)' (Bad) - Light Blue
             0.2 0.6 1;  % Color for legend entry '2 (S2)' (Ok)  - Medium Blue
             0.6 0.8 1;];  % Color for legend entry '3 (S2)' (Good)- Dark Blue
% --- END MODIFICATION 2 ---

% Assign per-bar color
% --- MODIFICATION 3: Map stack components to correct colors ---
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
% --- END MODIFICATION 3 ---

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
legend_labels = {' 1 (SD)', ' 2 (SD)', ' 3 (SD)', ' 1 (S2)', ' 2 (S2)', ' 3 (S2)'};
% You had two lgd=legend lines, the second overwrites NumColumns. Combine them:
lgd = legend(dummy, legend_labels, 'Location', 'northeastoutside', 'NumColumns', 2);
lgd.Title.String = {"Usability Index", "(3=Good; 2=Ok; 1=Bad)"};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 
% % Only use rating 3 data
% sd_data = data(:,4);  % column for rating 3 (SD)
% s2_data = data(:,8);  % column for rating 3 (S2)
% 
% n = 15;
% 
% % Combine into a 15x2 matrix: [SD, S2]
% grouped_data = [sd_data, s2_data];
% 
% % Plot grouped bar chart
% figure;
% hb = bar(grouped_data, 'grouped');
% 
% % Set bar colors manually
% sd_color = [1 0.6 0.6];
% s2_color = [0.6 0.8 1];
% hb(1).FaceColor = sd_color;
% hb(2).FaceColor = s2_color;
% 
% % X-axis labels and formatting
% xticks(1:n);
% xticklabels(AOI_label);
% xtickangle(45);
% ylabel('Count');
% xlabel('AOI (latitude)');
% title('Total Best Imagery (3-Rated) per AOI');
% grid on;
% 
% lgd = legend({'3-Rating SD', '3-Rating S2'}, 'Location', 'northeastoutside');
% lgd.Title.String = 'Usability Index';
% 
% 












