clc; clear; close all

% Load data
data_name = "B:\Thesis Project\SDB_Time\Results\Marathon\Condition1_dsSD\Extracted Pts\pSDB\Marathon_PlanetScope_24c0_2023_02_14_15_07_01_L2W__RGB_ds_pSDBgreen_extracted.csv";
data = readmatrix(data_name);
y = data(:,3);  % Reference data
x = data(:,5);  % pSDB data

% Initial values
x_max = max(x);
x_min = min(x);


% Plot the scatter plot
figure;
scatter(x, y, 'k', 'filled', 'MarkerFaceAlpha', 0.3);
set(gca, 'YDir', 'reverse')
hold on;
xlabel('pSDB');
ylabel('Reference');
title('pSDB Linear Regression');
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

% Prompt user for reference value range
if contains(lower(data_name), "green")
    y_min_limit = -2;
    y_max_limits = -2.5:-0.5:-10;

elseif contains(lower(data_name), "red")
    y_min_limit = 0;
    y_max_limits = -0.5:-0.25:-10;
else
    y_min_limit = 0;
    y_max_limits = -0.5:-0.25:-15;
end


prev_R2 = -Inf;
best_R2 = -Inf;
best_fit_params = []; 
best_x_range = [];
best_y_range = [];
r2_values = []; 

for y_max_limit = y_max_limits
    % Filter data based on y-axis range
    range_idx = (y <= y_min_limit) & (y >= y_max_limit);
    x_range = x(range_idx);
    y_range = y(range_idx);
    
    if length(x_range) > 1  % Ensure valid regression
        % Perform regression on filtered data
        p_range = polyfit(x_range, y_range, 1);
        y_fit_range = polyval(p_range, x_range);

        % Calculate R²
        SS_tot_range = sum((y_range - mean(y_range)).^2);
        SS_res_range = sum((y_range - y_fit_range).^2);
        R2_range = 1 - (SS_res_range / SS_tot_range);
        
        % Store R² value
        r2_values = [r2_values; y_max_limit, R2_range];

        % If R² decreases, stop the loop
        if R2_range < prev_R2
            break;
        end

        % Store best values
        best_R2 = R2_range;
        best_fit_params = p_range;
        best_x_range = x_range;
        best_y_range = y_range;

        % Update previous R²
        prev_R2 = R2_range;
    end
end

% Plot best regression line
if ~isempty(best_x_range)
    y_best_fit = polyval(best_fit_params, best_x_range);
    plot(best_x_range, y_best_fit, 'r', 'LineWidth', 2);
    scatter(best_x_range, best_y_range, 'b', 'filled', 'SizeData', 8);

    % Display regression equation and best R²
    text(min(best_x_range) + 0.1, max(best_y_range) - 4, ...
        sprintf('y = %.2fx + %.2f\nR^2 = %.2f', best_fit_params(1), best_fit_params(2), best_R2), ...
        'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');
end

hold off;
legend('Data Points', 'Best Regression Fit');

disp(r2_values)


% % Filter data based on y-axis range
% range_idx = (y <= y_min_limit) & (y >= y_max_limit);
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









