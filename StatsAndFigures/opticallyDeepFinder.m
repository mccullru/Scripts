clc; clear; close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Intital loading and plotting scatter plot for regressions

% Load data
data_name = "B:\Thesis Project\SDB_Time\Results_main\Anegada\SuperDove\Extracted Pts\pSDB\PlanetScope_241f_2023_03_25_13_53_31_L2W__RGB_pSDBred_extracted.csv";
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{

Creates larger and larger depth ranges to include points to create regression
lines. Goes through all possible ranges specified by users input for min,
max, and step sizes, then highlights the regression line with the highest
R^2 value. If data is bad to begin with it won't really work, but when it
does work it gets pretty close to the visual initial curve of the linear
trend which is essentially the extinction depth. 

%}


disp(['Loading data from: ', data_name]);
if ~isfile(data_name)
    error('Data CSV file not found at the specified path: %s', data_name);
end
try
    data = readmatrix(data_name);
    if size(data,2) < 5 % Ensure at least 5 columns for x and y
        error('CSV file does not have enough columns. Expecting at least 5.');
    end
    y_original = data(:,3);  % Reference data (Column 3, assumed positive depths)
    x_original = data(:,5);  % pSDB data (Column 5)

    % Ensure x and y are column vectors and numeric
    x_original = double(x_original(:));
    y_original = double(y_original(:));

    % Remove rows with NaN in either x or y from the original dataset
    nan_rows = isnan(x_original) | isnan(y_original);
    x_original(nan_rows) = [];
    y_original(nan_rows) = [];
    if isempty(x_original) || isempty(y_original)
        error('No valid data points after removing NaNs from x or y.');
    end
    fprintf('Loaded %d valid data points after initial NaN removal.\n', length(x_original));

catch ME
    error('Failed to read or process input CSV file. Error ID: %s Message: %s', ME.identifier, ME.message);
end

% Extract just the filename for contains() logic
[~, just_the_filename, ~] = fileparts(data_name);
data_name_for_conditions = string(just_the_filename); % Use this for contains check

% Define depth_min_limit for the initial scatter plot based on data_name
if contains(lower(data_name_for_conditions), "green")
    depth_min_limit_for_plot = 2.0;
elseif contains(lower(data_name_for_conditions), "red")
    depth_min_limit_for_plot = 0.0;
else
    depth_min_limit_for_plot = 0.0;
end
disp(['Initial scatter plot will only show points with Reference Depth >= ', num2str(depth_min_limit_for_plot)]);

% Filter x and y for the initial scatter plot
plot_filter_idx = (y_original >= depth_min_limit_for_plot);
x_for_scatter = x_original(plot_filter_idx);
y_for_scatter = y_original(plot_filter_idx);

if isempty(x_for_scatter)
    warning('No data points meet the initial depth_min_limit for plotting. Scatter plot will be empty.');
end

scatter_handle = scatter(x_for_scatter, y_for_scatter, 36, 'k', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Filtered Data Points');

hold on;
xlabel('pSDB Value');
ylabel('Reference Depth (m)');
title(['pSDB Linear Regression Analysis: ', strrep(just_the_filename, '_', '\_')]); % Use filename in title
grid on;


% --- Iterative Regression Section ---
fprintf('\nCalculating and plotting regressions for increasing depth ranges...\n');

% Define Depth Limits for iteration based on data_name
if contains(lower(data_name_for_conditions), "green")
    depth_min_limit_regr = 2.0;
    overall_max_depth = 20.0;
    step = 0.25;
    initial_depth_max = 2.5;
    disp(['Using Green limits for regression: min=', num2str(depth_min_limit_regr), ', iterating max from ', num2str(initial_depth_max), ' to ', num2str(overall_max_depth)]);
elseif contains(lower(data_name_for_conditions), "red")
    depth_min_limit_regr = 1.0;
    overall_max_depth = 20.0;
    step = 0.25;
    initial_depth_max = 0.5;
    disp(['Using Red limits for regression: min=', num2str(depth_min_limit_regr), ', iterating max from ', num2str(initial_depth_max), ' to ', num2str(overall_max_depth)]);
else % Default/fallback
    depth_min_limit_regr = 1.0;
    overall_max_depth = 12.0;
    step = 0.25;
    initial_depth_max = 0.5;
    disp(['Using Default limits for regression: min=', num2str(depth_min_limit_regr), ', iterating max from ', num2str(initial_depth_max), ' to ', num2str(overall_max_depth)]);
end

depth_max_limits_to_test = initial_depth_max : step : overall_max_depth;

% Ensure the explicitly requested 1.0m limit is included if the range allows
if depth_min_limit_regr < 1.0 && initial_depth_max > 1.0 % Test if 1.0 is between min and initial max
    if ~ismember(1.0, depth_max_limits_to_test)
        depth_max_limits_to_test = sort([1.0, depth_max_limits_to_test]);
    end
elseif initial_depth_max <= 1.0 % If starting at or below 1.0, make sure 1.0 is there if it's a step
     if ~ismember(1.0, depth_max_limits_to_test) && 1.0 <= overall_max_depth && 1.0 >= initial_depth_max
        depth_max_limits_to_test = sort([depth_max_limits_to_test, 1.0]);
     end
end

if isempty(depth_max_limits_to_test)
    error('No depth ranges defined to test for regression.');
end

% Store results from each iteration
num_iterations = length(depth_max_limits_to_test);
iteration_results = struct(...
    'depth_limit', num2cell(depth_max_limits_to_test(:)), ... % Corrected: ensure column vector
    'R2', num2cell(nan(num_iterations, 1)), ...
    'params', cell(num_iterations, 1), ...
    'point_count', num2cell(zeros(num_iterations, 1)), ...
    'x_min_fit', num2cell(nan(num_iterations,1)), ...
    'x_max_fit', num2cell(nan(num_iterations,1)) ...
);

fprintf('Calculating regression for %d depth ranges...\n', num_iterations);
for k = 1:num_iterations
    current_depth_max_limit = iteration_results(k).depth_limit;
    
    % Filter ORIGINAL x_original, y_original data for this iteration's regression
    range_idx = (y_original >= depth_min_limit_regr) & (y_original <= current_depth_max_limit);
    
    x_iter_range = x_original(range_idx);
    y_iter_range = y_original(range_idx);
    
    num_points = length(x_iter_range);
    iteration_results(k).point_count = num_points;

    fprintf('Testing regression range [%.2f, %.2f]: %d points. ', depth_min_limit_regr, current_depth_max_limit, num_points);
    
    if num_points > 1
        p_range = polyfit(x_iter_range, y_iter_range, 1); % [m1, m0]
        
        y_fit_iter_range = polyval(p_range, x_iter_range);
        SS_tot_range = sum((y_iter_range - mean(y_iter_range)).^2);
        if SS_tot_range < eps % Handle cases with effectively zero variance to avoid NaN R2
             R2_range = 1.0; % If all y_iter_range are the same, fit is perfect
        else
            SS_res_range = sum((y_iter_range - y_fit_iter_range).^2);
            R2_range = 1 - (SS_res_range / SS_tot_range);
            if R2_range < 0 && SS_tot_range > eps % R2 can be negative if model is worse than horizontal line
                % R2_range = 0; % Option: Cap R2 at 0
            end
        end
        fprintf('R2 = %.4f\n', R2_range);
        
        iteration_results(k).R2 = R2_range;
        iteration_results(k).params = p_range;
        if ~isempty(x_iter_range)
            iteration_results(k).x_min_fit = min(x_iter_range);
            iteration_results(k).x_max_fit = max(x_iter_range);
        end
    else
        fprintf('Not enough points for regression. R2=NaN\n');
        % R2, params, x_min_fit, x_max_fit remain NaN
    end
end


% --- Find the iteration with the best R² ---
all_R2 = [iteration_results.R2];
valid_R2_idx = find(~isnan(all_R2)); % Indices of non-NaN R2 values
best_k_overall_index = []; best_R2_val = -Inf; best_fit_params = []; best_depth_limit_final = NaN;

if ~isempty(valid_R2_idx)
    [best_R2_val, temp_idx] = max(all_R2(valid_R2_idx)); % Find max among valid R2s
    best_k_overall_index = valid_R2_idx(temp_idx);      % Get original index from iteration_results
    
    best_fit_params = iteration_results(best_k_overall_index).params;
    best_depth_limit_final = iteration_results(best_k_overall_index).depth_limit;
    fprintf('\nBest R2 = %.4f found for max depth limit = %.2f\n', best_R2_val, best_depth_limit_final);
else
    warning('No valid R2 values calculated. Cannot determine or plot best fit.');
end

% --- Plotting Regression Lines ---
disp('Plotting all regression lines (only over their respective data x-ranges)...');
for k = 1:num_iterations
    if ~isnan(iteration_results(k).R2) && ...
       ~isempty(iteration_results(k).params) && ...
       ~isnan(iteration_results(k).x_min_fit) && ...
       ~isnan(iteration_results(k).x_max_fit)

        p_current = iteration_results(k).params;
        current_x_min = iteration_results(k).x_min_fit;
        current_x_max = iteration_results(k).x_max_fit;

        if current_x_min == current_x_max % If only one unique x-value in range
            x_line_segment = [current_x_min; current_x_min]; % Create a point
        else
            x_line_segment = linspace(current_x_min, current_x_max, 20)'; % 20 points for a smooth line segment
        end
        
        y_plot_fit_segment = polyval(p_current, x_line_segment);
        plot(x_line_segment, y_plot_fit_segment, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5, 'HandleVisibility', 'off');
    end
end

% Plot the best regression line highlighted
best_line_handle = plot(NaN,NaN,'r', 'LineWidth', 2.5, 'DisplayName', 'Best R² Fit'); % Initialize for legend

if ~isempty(best_k_overall_index)
    fprintf('Highlighting best fit line (R2=%.4f, Max Depth=%.2f)...\n', best_R2_val, best_depth_limit_final);
    
    best_x_min_line = iteration_results(best_k_overall_index).x_min_fit;
    best_x_max_line = iteration_results(best_k_overall_index).x_max_fit;

    if best_x_min_line == best_x_max_line
        x_best_line_segment = [best_x_min_line; best_x_min_line];
    else
        x_best_line_segment = linspace(best_x_min_line, best_x_max_line, 20)';
    end
    
    y_best_plot_fit_segment = polyval(best_fit_params, x_best_line_segment);
    
    % Overwrite dummy handle with actual plot
    best_line_handle = plot(x_best_line_segment, y_best_plot_fit_segment, 'r', 'LineWidth', 2.5, 'DisplayName', 'Best R² Fit');
end

% --- Text Annotation for Best Fit ---
if ~isempty(best_k_overall_index)
    % Use overall min/max of *plotted* data for relative positioning
    plot_x_min = min(x_for_scatter); plot_x_max = max(x_for_scatter);
    plot_y_min = min(y_for_scatter); plot_y_max = max(y_for_scatter);

    text_x_pos = plot_x_min + (plot_x_max - plot_x_min) * 0.05; % 5% from left
    if strcmp(get(gca, 'YDir'), 'reverse')
         text_y_pos = plot_y_min + (plot_y_max - plot_y_min) * 0.05; % 5% from visual top (min y)
    else
         text_y_pos = plot_y_max - (plot_y_max - plot_y_min) * 0.05; % 5% from visual top (max y)
    end

    text(text_x_pos, text_y_pos, ...
        sprintf('Best Fit (Range: %.2f-%.2f m)\ny = %.2fx + %.2f\nR^2 = %.2f', ...
                depth_min_limit_regr, best_depth_limit_final, best_fit_params(1), best_fit_params(2), best_R2_val), ...
        'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'EdgeColor', 'k', 'VerticalAlignment', 'top');
end

hold off;
% Legend should use defined handles
if ~isempty(x_for_scatter) % Only add legend if data points were plotted
    legend([scatter_handle, best_line_handle], 'Location', 'best');
else
    legend([best_line_handle], 'Location', 'best');
end


% --- Display all R² values ---
disp('--- R2 Values per Max Depth Limit Tested ---');
r2_table = struct2table(iteration_results);

% Display relevant columns
disp(r2_table(:, {'depth_limit', 'R2', 'point_count', 'x_min_fit', 'x_max_fit'}));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{

Creates larger and larger pSDB ranges to include points to create regression
lines. Goes through all possible ranges specified by users input for min,
max, and step sizes, then highlights the regression line with the highest
R^2 value. Doesn't really work as well to find the extinction depth, I
think I will stick with teh increasing depth ranges for now as that also
just seems to work better overall.

%}

% 
% % Define how many steps to test for the upper x-limit
% %num_x_steps = 20; % Adjust number of iterations/steps (e.g., 20 steps across the range)
% x_step_size = 0.1;
% 
%
%
% disp(['Loading data from: ', data_name]);
% if ~isfile(data_name)
%     error('Data CSV file not found: %s', data_name);
% end
% 
% try
%     data = readmatrix(data_name);
%     if size(data,2) < 5 % Check based on assumed columns
%         error('CSV file needs at least 5 columns (assuming Col3=Y, Col5=X).');
%     end
%     y_original = data(:,3);  % Reference data (Column 3, positive depths)
%     x_original = data(:,5);  % pSDB data (Column 5)
% 
%     % Ensure x and y are column vectors and numeric (double)
%     x_original = double(x_original(:));
%     y_original = double(y_original(:));
% 
%     % Remove rows with NaN or Inf in either x or y from the original dataset
%     valid_rows = isfinite(x_original) & isfinite(y_original);
%     x_original = x_original(valid_rows);
%     y_original = y_original(valid_rows);
% 
%     if isempty(x_original) || isempty(y_original)
%         error('No valid (finite) data points after initial NaN/Inf removal.');
%     end
%     fprintf('Loaded %d valid data points after initial NaN/Inf removal.\n', length(x_original));
% 
% catch ME
%     error('Failed to read or process input CSV file. Error ID: %s Message: %s', ME.identifier, ME.message);
% end
% 
% % Extract just the filename for potential use in titles/conditions
% [~, just_the_filename, ~] = fileparts(data_name);
% data_name_for_conditions = string(just_the_filename); % Convert to string for 'contains'
% % --- End Data Loading ---
% 
% % --- Initial Plot Setup ---
% % NOTE: This initial plot still uses a Y-based filter for displayed points.
% % You might want to remove this filter if you want to see ALL points
% % over which the X-based regression iteration occurs.
% figure;
% 
% % Define depth_min_limit_for_plot based on data_name (kept from previous logic)
% if contains(lower(data_name_for_conditions), "green")
%     depth_min_limit_for_plot = 0.0;
% elseif contains(lower(data_name_for_conditions), "red")
%     depth_min_limit_for_plot = 0.0;
% else
%     depth_min_limit_for_plot = 0.0;
% end
% disp(['Initial scatter plot shows points with Reference Depth >= ', num2str(depth_min_limit_for_plot)]);
% 
% % Filter x and y for the initial scatter plot based on Y limit
% plot_filter_idx = (y_original >= depth_min_limit_for_plot);
% x_for_scatter = x_original(plot_filter_idx);
% y_for_scatter = y_original(plot_filter_idx);
% 
% if isempty(x_for_scatter)
%     warning('No data points meet initial depth filter for scatter plot.');
%     scatter_handle = scatter(NaN, NaN, 36, 'k', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Filtered Data Points (None)'); % Plot empty for legend
% else
%     scatter_handle = scatter(x_for_scatter, y_for_scatter, 36, 'k', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Filtered Data Points');
% end
% hold on;
% xlabel('pSDB Value');
% ylabel('Reference Depth (m)');
% title(['pSDB Linear Regression Analysis (Iterating X): ', strrep(just_the_filename, '_', '\_')]);
% grid on;
% % --- End Initial Plot Setup ---
% 
% 
% % --- Iterative Regression Block (Iterating on X) ---
% fprintf('\nCalculating regressions for increasing pSDB (X) ranges...\n');
% 
% % Define X Limits for Iteration
% x_min_overall = min(x_original);
% x_max_overall = max(x_original);
% x_min_limit_regr = x_min_overall; % Start including all points from the minimum x value
% 
% % --- MODIFIED: Define max X limits using STEP SIZE ---
% % Define where the iteration starts for the upper limit
% initial_x_max = x_min_overall + x_step_size; % Start testing one step above the minimum
% % Ensure start point isn't beyond the overall max
% if initial_x_max > x_max_overall
%     initial_x_max = x_max_overall; % If range is smaller than step, test only full range
%     x_max_limits_to_test = [x_max_overall]; % Just one iteration
% else
%     % Create array using colon operator: start : step : end
%     x_max_limits_to_test = initial_x_max : x_step_size : x_max_overall;
%     % Ensure the absolute maximum x value is included if the step doesn't land on it
%     if ~isempty(x_max_limits_to_test) && abs(x_max_limits_to_test(end) - x_max_overall) > 1e-9 % Check if last step missed max
%         x_max_limits_to_test(end+1) = x_max_overall;
%     elseif isempty(x_max_limits_to_test) % Handle case where step is too large, just test max
%          x_max_limits_to_test = [x_max_overall];
%     end
% end
% % --- END MODIFICATION ---
% 
% % --- Update num_iterations AFTER creating the test array ---
% num_iterations = length(x_max_limits_to_test);
% % ---
% 
% % --- MODIFIED: Update fprintf message ---
% fprintf('Iterating max X from %.3f to %.3f with step size approx %.3f (%d iterations).\n', ...
%         x_max_limits_to_test(1), x_max_overall, x_step_size, num_iterations);
% % --- END MODIFICATION ---
% 
% if isempty(x_max_limits_to_test) || num_iterations < 1
%     error('No x ranges defined to test with the specified step size.');
% end
% 
% % --- Store results from each iteration (Struct definition is unchanged) ---
% iteration_results = struct(...
%     'x_max_limit', num2cell(x_max_limits_to_test(:)), ...
%     'R2', num2cell(nan(num_iterations, 1)), ...
%     'params', cell(num_iterations, 1), ...
%     'point_count', num2cell(zeros(num_iterations, 1)), ...
%     'x_min_fit', num2cell(nan(num_iterations,1)), ...
%     'x_max_fit', num2cell(nan(num_iterations,1)) ...
% );
% % ---
% 
% % --- Loop through INCREASING maximum X limits ---
% fprintf('Calculating regression for %d X-ranges...\n', num_iterations);
% for k = 1:num_iterations
%     current_x_max_limit = iteration_results(k).x_max_limit;
% 
%     % --- MODIFIED: Filter ORIGINAL data based on current X range ---
%     range_idx = (x_original >= x_min_limit_regr) & (x_original <= current_x_max_limit);
% 
%     x_iter_range = x_original(range_idx); % X data for this iteration's regression
%     y_iter_range = y_original(range_idx); % Corresponding Y data
% 
%     num_points = length(x_iter_range);
%     iteration_results(k).point_count = num_points;
% 
%     fprintf('Testing X range [%.3f, %.3f]: %d points. ', x_min_limit_regr, current_x_max_limit, num_points);
% 
%     if num_points > 1
%         p_range = polyfit(x_iter_range, y_iter_range, 1); % Still fit Y ~ X
% 
%         y_fit_iter_range = polyval(p_range, x_iter_range);
%         SS_tot_range = sum((y_iter_range - mean(y_iter_range)).^2);
%         if SS_tot_range < eps
%              R2_range = 1.0;
%         else
%             SS_res_range = sum((y_iter_range - y_fit_iter_range).^2);
%             R2_range = 1 - (SS_res_range / SS_tot_range);
%             if R2_range < 0 && SS_tot_range > eps, R2_range = 0; end
%         end
%         fprintf('R2 = %.4f\n', R2_range);
% 
%         iteration_results(k).R2 = R2_range;
%         iteration_results(k).params = p_range;
%         if ~isempty(x_iter_range)
%             iteration_results(k).x_min_fit = min(x_iter_range); % Store actual x range boundaries
%             iteration_results(k).x_max_fit = max(x_iter_range);
%         end
%     else
%         fprintf('Not enough points for regression. R2=NaN\n');
%     end
% end
% % --- End Calculation Loop ---
% 
% % --- Find the iteration with the best R² ---
% all_R2 = [iteration_results.R2];
% valid_R2_idx = find(~isnan(all_R2));
% best_k_overall_index = []; best_R2_val = -Inf; best_fit_params = []; best_x_max_limit_final = NaN; % Initialize
% 
% if ~isempty(valid_R2_idx)
%     [best_R2_val, temp_idx] = max(all_R2(valid_R2_idx));
%     best_k_overall_index = valid_R2_idx(temp_idx);
%     best_fit_params = iteration_results(best_k_overall_index).params;
%     best_x_max_limit_final = iteration_results(best_k_overall_index).x_max_limit; % Get the best X limit
%     fprintf('\nBest R2 = %.4f found for max pSDB limit = %.3f\n', best_R2_val, best_x_max_limit_final);
% else
%     warning('No valid R2 values calculated. Cannot determine or plot best fit.');
% end
% % ---
% 
% % --- Plotting Regression Lines (Limited by X-Range) ---
% disp('Plotting all regression lines (only over their respective data x-ranges)...');
% for k = 1:num_iterations
%     % Check if regression was valid and x-limits were stored
%     if ~isnan(iteration_results(k).R2) && ~isempty(iteration_results(k).params) && ...
%        ~isnan(iteration_results(k).x_min_fit) && ~isnan(iteration_results(k).x_max_fit)
% 
%         p_current = iteration_results(k).params;
%         current_x_min = iteration_results(k).x_min_fit;
%         current_x_max = iteration_results(k).x_max_fit;
% 
%         if current_x_min >= current_x_max
%             x_line_segment = [current_x_min; current_x_min]; % Handle single point case
%         else
%             x_line_segment = linspace(current_x_min, current_x_max, 20)';
%         end
% 
%         y_plot_fit_segment = polyval(p_current, x_line_segment);
%         plot(x_line_segment, y_plot_fit_segment, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5, 'HandleVisibility', 'off');
%     end
% end
% 
% % Plot the best regression line highlighted (limited by its X-Range)
% best_line_handle = plot(NaN,NaN,'r', 'LineWidth', 2.5, 'DisplayName', 'Best R² Fit'); % Initialize for legend
% 
% if ~isempty(best_k_overall_index)
%     fprintf('Highlighting best fit line (R2=%.4f, Max pSDB=%.3f)...\n', best_R2_val, best_x_max_limit_final);
% 
%     % Get x limits for the best fit line
%     best_x_min_line = iteration_results(best_k_overall_index).x_min_fit;
%     best_x_max_line = iteration_results(best_k_overall_index).x_max_fit;
% 
%     if best_x_min_line >= best_x_max_line
%         x_best_line_segment = [best_x_min_line; best_x_min_line];
%     else
%         x_best_line_segment = linspace(best_x_min_line, best_x_max_line, 20)';
%     end
% 
%     y_best_plot_fit_segment = polyval(best_fit_params, x_best_line_segment);
%     % Overwrite dummy handle
%     best_line_handle = plot(x_best_line_segment, y_best_plot_fit_segment, 'r', 'LineWidth', 2.5, 'DisplayName', 'Best R² Fit');
% end
% % ---
% 
% % --- Text Annotation for Best Fit ---
% if ~isempty(best_k_overall_index)
%     % Positioning based on overall original data range (can adjust)
%     text_x_pos = min(x_original) + (max(x_original) - min(x_original)) * 0.05;
%     % Using Y range of initially plotted points for vertical positioning
%     if isempty(y_for_scatter) % Handle case where no points were plotted initially
%         plot_y_min = min(y_original); plot_y_max = max(y_original);
%     else
%         plot_y_min = min(y_for_scatter); plot_y_max = max(y_for_scatter);
%     end
%     if strcmp(get(gca, 'YDir'), 'reverse')
%          text_y_pos = plot_y_min + (plot_y_max - plot_y_min) * 0.1; % Closer to top for reversed axis
%     else
%          text_y_pos = plot_y_max - (plot_y_max - plot_y_min) * 0.1; % Closer to top for normal axis
%     end
% 
%     % MODIFIED Text to show X range
%     text(text_x_pos, text_y_pos, ...
%         sprintf('Best Fit (pSDB Range %.3f-%.3f):\ny = %.2fx + %.2f\nR^2 = %.2f', ...
%                 x_min_limit_regr, best_x_max_limit_final, best_fit_params(1), best_fit_params(2), best_R2_val), ...
%         'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'EdgeColor', 'k', 'VerticalAlignment', 'top');
% end
% % ---
% 
% hold off;
% % Update Legend
% if ~isempty(x_for_scatter) && ~isempty(best_k_overall_index)
%     legend([scatter_handle, best_line_handle], 'Location', 'best');
% elseif ~isempty(best_k_overall_index)
%      legend([best_line_handle], 'Location', 'best');
% % else no legend if no best line and no scatter points
% end
% 
% 
% % --- Display all R² values ---
% disp('--- R2 Values per Max pSDB (X) Limit Tested ---'); % Changed title
% r2_table = struct2table(iteration_results);
% % MODIFIED Display relevant columns
% disp(r2_table(:, {'x_max_limit', 'R2', 'point_count', 'x_min_fit', 'x_max_fit'}));
% % --- End Script ---



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










