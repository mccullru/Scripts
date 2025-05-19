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

sd_data = data(:,2:4);
s2_data = data(:,6:8);

n = 15;

grouped_data = zeros(n*2, 3);
grouped_data(1:2:end, :) = sd_data;
grouped_data(2:2:end, :) = s2_data;

% Plot
figure;
hb = bar(grouped_data, 'stacked');

% Define SD (reds) and S2 (blues)
sd_colors = [0.6 0 0; 1 0.2 0.2; 1 0.6 0.6;];
s2_colors = [0 0 0.6; 0.2 0.6 1; 0.6 0.8 1;];

% Assign per-bar color
for k = 1:3
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
ylabel('Images Per Year');
xlabel('AOI (latitude)');
title('Imagery Rates with Usability Index');
grid on;

% Create dummy objects for the unified legend
dummy = gobjects(6,1);

% SD Legend (first 3 entries)
for k = 1:3
    dummy(k) = patch(NaN, NaN, sd_colors(k,:));
end

% S2 Legend (next 3 entries)
for k = 1:3
    dummy(k+3) = patch(NaN, NaN, s2_colors(k,:));
end

% Unified Legend
legend_labels = {' 1 (SD)', ' 2 (SD)', ' 3 (SD)', ' 1 (S2)', ' 2 (S2)', ' 3 (S2)'};

% --- MINIMAL CHANGE IS HERE ---
lgd = legend(dummy, legend_labels, 'Location', 'northeastoutside', 'NumColumns', 2); % Added 'NumColumns', 2
% --- END OF MINIMAL CHANGE ---

lgd = legend(dummy, legend_labels, 'Location', 'northeastoutside');
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












