clc; clear; close all

% Load data
data_name = "E:\Thesis Stuff\pSDB_ExtractedPts\S2A_MSI_2023_01_07_11_43_34_T28RFS_L2W__RGB_pSDBgreen_extracted.csv";
data = readmatrix(data_name);
y = data(:,3);  % Reference data
x = data(:,5);  % pSDB data

% Initial values
x_max = max(x);
x_min = min(x);


% Plot the scatter plot
figure;
scatter(x, y, 'k', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
xlabel('pSDB Values (unitless)');
ylabel('Reference Bathymetry (m)');
title('pSDB Linear Regression (S2)');
grid on;


% % Define bin width
% bin_size = (x_max-x_min)/1;
% x_min = floor(min(x));
% x_max = ceil(max(x));
% 
% % Store R² values for each bin
% r2_values = [];

% % Loop through bins
% for start = x_min:bin_size:(x_max - bin_size)
%     % Find data points within the current bin
%     bin_idx = (x >= start) & (x < start + bin_size);
%     x_bin = x(bin_idx);
%     y_bin = y(bin_idx);
%     
%     if length(x_bin) > .01  % Ensure at least two points for regression
%         % Perform linear regression
%         p = polyfit(x_bin, y_bin, 1); % First-degree polynomial (linear)
%         y_fit = polyval(p, x_bin); % Compute fitted values
%         
%         % Calculate R²
%         SS_tot = sum((y_bin - mean(y_bin)).^2);
%         SS_res = sum((y_bin - y_fit).^2);
%         R2 = 1 - (SS_res / SS_tot);
%         
%         % Store R² value
%         r2_values = [r2_values; start, R2];
% 
%         % Plot regression line for this bin
%         plot(x_bin, y_fit, 'b', 'LineWidth', 2);
%     
%         % Display R² value on the plot
%         mid_x = mean(x_bin);
%         mid_y = mean(y_bin);
%         text(mid_x+.02, mid_y+.05, sprintf('R^2 = %.2f', R2), 'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold');
%     
%     
%     end
% end


% hold off;
% legend('Data Points', 'Piecewise Linear Regression');
% 
% % Display R² values for each bin
% disp('R² values for each bin:');
% disp(array2table(r2_values, 'VariableNames', {'Bin_Start', 'R2'}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Prompt user for reference value range
% if contains(lower(data_name), "green")
%     y_min_limit = 2;
%     y_max_limits = 2.5:1:10;
% 
% elseif contains(lower(data_name), "red")
%     y_min_limit = 0;
%     y_max_limits = 0.5:0.25:10;
% else
%     y_min_limit = 0;
%     y_max_limits = 0.5:0.25:15;
% end
% 
% 
% prev_R2 = -Inf;
% best_R2 = -Inf;
% best_fit_params = []; 
% best_x_range = [];
% best_y_range = [];
% r2_values = []; 
% 
% for y_max_limit = y_max_limits
%     % Filter data based on y-axis range
%     range_idx = (y >= y_min_limit) & (y <= y_max_limit);
%     x_range = x(range_idx);
%     y_range = y(range_idx);
%     
%     if length(x_range) > 1  % Ensure valid regression
%         % Perform regression on filtered data
%         p_range = polyfit(x_range, y_range, 1);
%         y_fit_range = polyval(p_range, x_range);
% 
%         % Calculate R²
%         SS_tot_range = sum((y_range - mean(y_range)).^2);
%         SS_res_range = sum((y_range - y_fit_range).^2);
%         R2_range = 1 - (SS_res_range / SS_tot_range);
%         
%         % Store R² value
%         r2_values = [r2_values; y_max_limit, R2_range];
% 
%         % If R² decreases, stop the loop
%         if R2_range < prev_R2
%             break;
%         end
% 
%         % Store best values
%         best_R2 = R2_range;
%         best_fit_params = p_range;
%         best_x_range = x_range;
%         best_y_range = y_range;
% 
%         % Update previous R²
%         prev_R2 = R2_range;
%     end
% end
% 
% % Plot best regression line
% if ~isempty(best_x_range)
%     y_best_fit = polyval(best_fit_params, best_x_range);
%     plot(best_x_range, y_best_fit, 'r', 'LineWidth', 2);
%     scatter(best_x_range, best_y_range, 'b', 'filled', 'SizeData', 8);
% 
%     % Display regression equation and best R²
%     text(min(best_x_range) + 0.1, max(best_y_range) - 4, ...
%         sprintf('y = %.2fx + %.2f\nR^2 = %.2f', best_fit_params(1), best_fit_params(2), best_R2), ...
%         'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');
% end
% 
% hold off;
% legend('Data Points', 'Best Regression Fit');
% 
% disp(r2_values)




% 
% % Filter data based on y-axis range
% 
% figure;
% range_idx = (y >= y_min_limit) & (y <= y_max_limit);
% x_range = x(range_idx);
% y_range = y(range_idx);
% 
% if length(x_range) > 1  % Ensure valid regression
%     % Perform regression on filtered data
%     p_range = polyfit(x_range, y_range, 1);
%     y_fit_range = polyval(p_range, x_range);
% 
%     % Calculate R²
%     SS_tot_range = sum((y_range - mean(y_range)).^2);
%     SS_res_range = sum((y_range - y_fit_range).^2);
%     R2_range = 1 - (SS_res_range / SS_tot_range);
% 
%     % Plot regression line
%     plot(x_range, y_fit_range, 'r', 'LineWidth', 2); 
%     scatter(x_range, y_range, 'b', SizeData=8)
%     
%     % Display regression equation and R²
%     text(min(x_range), max(y_range)-4, ...
%         sprintf('y = %.2fx + %.2f\nR^2 = %.2f', p_range(1), p_range(2), R2_range), ...
%         'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');
% end
% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% --- Assume x, y, and data_name are already defined from your preceding code ---
% x = pSDB data (positive values)
% y = Reference depth data (positive values, larger means deeper)
% data_name = Name of the data file being processed

fprintf('\nCalculating and plotting regressions for increasing depth ranges...\n');

% --- Define Depth Limits ---
% Define the absolute shallowest depth to include
if contains(lower(data_name), "green")
    depth_min_limit = 2.0; % Shallowest depth to START including
    overall_max_depth = 15.0; % Define the maximum depth to iterate up to
    step = 0.5;
    initial_depth_max = 2.5; % Start iteration from here
    disp(['Using Green limits: min=', num2str(depth_min_limit), ', iterating max from ', num2str(initial_depth_max), ' to ', num2str(overall_max_depth)]);
elseif contains(lower(data_name), "red")
    depth_min_limit = 0.0;
    overall_max_depth = 15.0;
    step = 0.5;
    initial_depth_max = 0.5;
    disp(['Using Red limits: min=', num2str(depth_min_limit), ', iterating max from ', num2str(initial_depth_max), ' to ', num2str(overall_max_depth)]);
else % Default/fallback
    depth_min_limit = 0.0;
    overall_max_depth = 15.0; % Adjusted based on previous limits
    step = 0.5;
    initial_depth_max = 0.5;
    disp(['Using Default limits: min=', num2str(depth_min_limit), ', iterating max from ', num2str(initial_depth_max), ' to ', num2str(overall_max_depth)]);
end

% Define the specific maximum depth limits to test
depth_max_limits_to_test = initial_depth_max : step : overall_max_depth;
% Ensure the initial minimum range (up to 1m) is tested if applicable
if depth_min_limit < 1.0 && initial_depth_max > 1.0
    depth_max_limits_to_test = [1.0, depth_max_limits_to_test]; % Add 1m test explicitly
elseif initial_depth_max < 1.0 % If starting below 1m, ensure 1m is included later
    if ~ismember(1.0, depth_max_limits_to_test)
         depth_max_limits_to_test = sort([depth_max_limits_to_test, 1.0]);
    end
end
if isempty(depth_max_limits_to_test)
    error('No depth ranges defined to test.');
end
% --- End Limit Definition ---

% --- Store results from each iteration ---
num_iterations = length(depth_max_limits_to_test);
iteration_results = struct('depth_limit', num2cell(depth_max_limits_to_test(:)), ... % <<< ADDED (:)
                           'R2', num2cell(nan(num_iterations, 1)), ...
                           'params', cell(num_iterations, 1), ...
                           'point_count', num2cell(zeros(num_iterations, 1)));
% ---

% --- Loop through INCREASING maximum depth limits ---
fprintf('Calculating regression for %d depth ranges...\n', num_iterations);
for k = 1:num_iterations
    depth_max_limit = iteration_results(k).depth_limit;

    % Filter data based on current y-axis range [depth_min_limit, depth_max_limit]
    range_idx = (y >= depth_min_limit) & (y <= depth_max_limit);

    x_range = x(range_idx);
    y_range = y(range_idx);
    num_points = length(x_range);
    iteration_results(k).point_count = num_points;

    fprintf('Testing range [%.2f, %.2f]: %d points. ', depth_min_limit, depth_max_limit, num_points);

    % Ensure enough points for a valid regression
    if num_points > 1
        % Perform linear regression on the filtered data subset
        p_range = polyfit(x_range, y_range, 1); % [m1, m0] slope and intercept

        % Calculate R² for this specific range
        y_fit_range = polyval(p_range, x_range);
        SS_tot_range = sum((y_range - mean(y_range)).^2);
        if SS_tot_range == 0 % Avoid division by zero if all y_range values are identical
             R2_range = 1.0;
        else
            SS_res_range = sum((y_range - y_fit_range).^2);
            % Handle potential negative R2 due to poor fit or low variance
            R2_range = max(0, 1 - (SS_res_range / SS_tot_range));
        end
        fprintf('R2 = %.4f\n', R2_range);

        % Store results for this iteration
        iteration_results(k).R2 = R2_range;
        iteration_results(k).params = p_range;

    else
        fprintf('Not enough points for regression. R2=NaN\n');
        % R2 and params remain NaN/empty as initialized
    end
end % End loop through depth_max_limits
% --- End Calculation Loop ---


% --- Find the iteration with the best R² ---
all_R2 = [iteration_results.R2];
% Handle potential NaNs when finding the max
valid_R2_idx = find(~isnan(all_R2));
if isempty(valid_R2_idx)
    warning('No valid R2 values calculated. Cannot determine best fit.');
    best_k = [];
    best_R2 = NaN;
else
    [best_R2, best_idx_in_valid] = max(all_R2(valid_R2_idx));
    best_k = valid_R2_idx(best_idx_in_valid); % Index in the original iteration_results struct
    best_fit_params = iteration_results(best_k).params;
    best_depth_limit = iteration_results(best_k).depth_limit;
    fprintf('\nBest R2 = %.4f found for max depth limit = %.2f\n', best_R2, best_depth_limit);
end
% ---


% --- Plotting ---
% Assumes figure and initial scatter plot (x, y) already exist with 'hold on'

% Define x range for plotting lines (e.g., min to max of original x data)
x_plot_range = linspace(min(x), max(x), 100)'; % Column vector for polyval if needed

% Plot all regression lines first in a default color
disp('Plotting all regression lines...');
for k = 1:num_iterations
    if ~isnan(iteration_results(k).R2) % Only plot if regression was successful
        p_current = iteration_results(k).params;
        y_plot_fit = polyval(p_current, x_plot_range);
        plot(x_plot_range, y_plot_fit, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5); % Light gray lines
    end
end

% Plot the best regression line highlighted
if ~isempty(best_k)
    fprintf('Highlighting best fit line (R2=%.4f, Max Depth=%.2f)...\n', best_R2, best_depth_limit);
    y_best_plot_fit = polyval(best_fit_params, x_plot_range);
    plot(x_plot_range, y_best_plot_fit, 'r', 'LineWidth', 2.5, 'DisplayName', 'Best R² Fit'); % Red, thicker, label for legend
else
     plot(NaN,NaN,'r', 'LineWidth', 2.5, 'DisplayName', 'Best R² Fit (None)'); % Add dummy for legend consistency
end

% --- Text Annotation for Best Fit ---
if ~isempty(best_k)
    % Adjust text position if needed based on your data range and axis direction
    text_x_pos = min(x) + (max(x) - min(x)) * 0.1;
    text_y_pos = min(y) + (max(y) - min(y)) * 0.8;
     if strcmp(get(gca, 'YDir'), 'reverse') % Adjust if Y-axis is reversed
         text_y_pos = max(y) - (max(y) - min(y)) * 0.8;
     end

    text(text_x_pos, text_y_pos, ...
        sprintf('Best Fit (Max Depth %.2f m):\ny = %.2fx + %.2f\nR^2 = %.2f', ...
                best_depth_limit, best_fit_params(1), best_fit_params(2), best_R2), ...
        'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'EdgeColor', 'k');
end
% ---

hold off; % Turn hold off after adding all lines

% Update Legend
% Find the handle for the original scatter plot if needed, assume it's the first child
h = get(gca,'children');
legend([h(end), h(1)], {'Data Points', 'Best R² Fit'}); % Assumes scatter was first, best line was last before hold off
% If scatter wasn't first, adjust index for h(end)

% --- Display all R² values ---
disp('--- R2 Values per Max Depth Limit ---');
r2_table = struct2table(iteration_results); % Convert struct to table for display
disp(r2_table(:, {'depth_limit', 'R2', 'point_count'})); % Display relevant columns
% ---





















