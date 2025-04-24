clc
clear
close all

% Computes summary statistics for the SDB error values compared to the
% truth LiDAR dataset.


data1 = readmatrix('Marathon.csv');
data2 = readmatrix('Homer.csv');

% assigns data to SD and S2, make sure this is still correct for different
% datasets. Gets a little funky, pay attention to data assignments
SDred = data2(:,1);
SDgreen = data1(:,1);
S2red = data2(:,2);
S2green = data1(:,2);

% finds various summary stats for SD and S2

% SuperDove green and red stats
SDred_min = min(SDred);
SDred_max = max(SDred);
SDred_average = mean(SDred);
SDred_std = std(SDred);
SDred_numpts = length(SDred);
SDred_skew = skewness(SDred);
SDred_kurt = kurtosis(SDred);
SDred_rmse = sqrt(mean(SDred.^2));

SDgreen_min = min(SDgreen);
SDgreen_max = max(SDgreen);
SDgreen_average = mean(SDgreen);
SDgreen_std = std(SDgreen);
SDgreen_numpts = length(SDgreen);
SDgreen_skew = skewness(SDgreen);
SDgreen_kurt = kurtosis(SDgreen);
SDgreen_rmse = sqrt(mean(SDgreen.^2));


% Sentinel-2 pSDB green and red stats
S2red_min = min(SDred);
S2red_max = max(SDred);
S2red_average = mean(SDred);
S2red_std = std(SDred);
S2red_numpts = length(SDred);
S2red_skew = skewness(SDred);
S2red_kurt = kurtosis(SDred);
S2red_rmse = sqrt(mean(S2red.^2));


S2green_min = min(SDgreen);
S2green_max = max(SDgreen);
S2green_average = mean(SDgreen);
S2green_std = std(SDgreen);
S2green_numpts = length(SDgreen);
S2green_skew = skewness(SDgreen);
S2green_kurt = kurtosis(SDgreen);
S2green_rmse = sqrt(mean(S2green.^2));


%%Plots the SDBred error distributions


figure
%subplot(2,2,1)
S2red_chart = histogram(S2red, 'FaceColor','#5E5854', 'BinWidth', 0.2);
hold on
xline(S2red_average, 'LineWidth', 2, 'Color', 'r');
xline((S2red_average + S2red_std), 'LineWidth', 2, 'Color', 'b')
xline((S2red_average - S2red_std), 'LineWidth', 2, 'Color', 'b')
xlabel('Error (meters)', 'FontSize',12)
ylabel('Count', 'FontSize',12)
ylim([0,250])
xlim([-3, 3])
legend({'Error Distribution', 'Mean Error', '+/-1 Std Dev.'}, 'FontSize', 12)
title({'Sentinel-2 SDBred Error Distribution', 'Homer, AK'}, 'fontsize', 14)
annotation('textbox', [0.14, 0.78, .1, .1],'EdgeColor', 'none', 'FontSize', 13, 'string', ...
    compose('Min: %.2f\nMax: %.2f\nMean: %.2f\nStd Dev: %.2f\nRMSE: %.2f\nTotal Count: %s', ...
    S2red_min, S2red_max, S2red_average, S2red_std, S2red_rmse, num2str(length(S2red))));

figure
%%subplot(2,2,2)
SDred_chart = histogram(SDred, 'FaceColor','#5E5854', 'BinWidth', 0.2);
hold on
xline(SDred_average, 'LineWidth', 2, 'Color', 'r');
xline((SDred_average + SDred_std), 'LineWidth', 2, 'Color', 'b')
xline((SDred_average - SDred_std), 'LineWidth', 2, 'Color', 'b')
xlabel('Error (meters)', 'FontSize',12)
ylabel('Count', 'FontSize',12)
ylim([0,250])
xlim([-3, 3])
legend({'Error Distribution', 'Mean Error', '+/-1 Std Dev.'}, 'FontSize',12)
title({'SuperDove SDBred Error Distribution', 'Homer, AK'}, 'FontSize',14)
annotation('textbox', [0.14, 0.78, .1, .1],'EdgeColor', 'none', 'FontSize', 13, 'string', ...
    compose('Min: %.2f\nMax: %.2f\nMean: %.2f\nStd Dev: %.2f\nRMSE: %.2f\nTotal Count: %s', ...
    SDred_min, SDred_max, SDred_average, SDred_std, SDred_rmse, num2str(length(SDred))));

% figure
% %subplot(2,2,3)
% S2green_chart = histogram(S2green, 'FaceColor','#5E5854', 'BinWidth', 0.2);
% hold on
% xline(S2green_average, 'LineWidth', 2, 'Color', 'r');
% xline((S2green_average + S2green_std), 'LineWidth', 2, 'Color', 'b')
% xline((S2green_average - S2green_std), 'LineWidth', 2, 'Color', 'b')
% xlabel('Error (meters)', 'FontSize',12)
% ylabel('Count', 'FontSize',12)
% ylim([0,250])
% xlim([-3, 3])
% legend({'Error Distribution', 'Mean Error', '+/-1 Std Dev.'}, 'FontSize',12)
% title({'Sentinel-2 SDBgreen Error Distribution', 'Marathon, FL'}, 'FontSize',14)
% annotation('textbox', [0.14, 0.78, .1, .1],'EdgeColor', 'none', 'FontSize', 13, 'string', ...
%     compose('Min: %.2f\nMax: %.2f\nMean: %.2f\nStd Dev: %.2f\nRMSE: %.2f\nTotal Count: %s', ...
%     S2green_min, S2green_max, S2green_average, S2green_std, S2green_rmse, num2str(length(S2green))));

% figure
%%subplot(2,2,4)
% SDgreen_chart = histogram(SDgreen, 'FaceColor','#5E5854', 'BinWidth', 0.2);
% hold on
% xline(SDgreen_average, 'LineWidth', 2, 'Color', 'r');
% xline((SDgreen_average + SDgreen_std), 'LineWidth', 2, 'Color', 'b')
% xline((SDgreen_average - SDgreen_std), 'LineWidth', 2, 'Color', 'b')
% xlabel('Error (meters)', 'FontSize',12)
% ylabel('Count', 'FontSize',12)
% ylim([0,250])
% xlim([-3, 3])
% legend({'Error Distribution', 'Mean Error', '+/-1 Std Dev.'}, 'FontSize',12)
% title({'SuperDove SDBgreen Error Distribution', 'Marathon, FL'}, 'FontSize',14)
% annotation('textbox', [0.14, 0.78, .1, .1],'EdgeColor', 'none', 'FontSize', 13, 'string', ...
%     compose('Min: %.2f\nMax: %.2f\nMean: %.2f\nStd Dev: %.2f\nRMSE: %.2f\nTotal Count: %s', ...
%     SDgreen_min, SDgreen_max, SDgreen_average, SDgreen_std, SDgreen_rmse, num2str(length(SDgreen))));



%%
a = 60;
b = 60;
c = 240.00694;
d = 120.001389;

A = [1 0;0 1;-1 -1;1 1];
W = eye;
L = [a;b;c-360;d];

ATWA = inv(A'*W*A);

X = ATWA * A'*W*L



%%

l1 = 30.005556;
l2 = 50;
l3 = 20;
l4 = 40.005556;

A = [1 0 0;1 1 0;0 0 1;0 1 1];
y = [l1;l2;l3;l4];

xi = inv(A'*A)*A'*y;

e = y - A*xi;

l1_hat = l1-e(1)
l2_hat = l2-e(2)
l3_hat = l3-e(3)
l4_hat = l4-e(4)


%%


A = [0 1 1;-1 2 1;-1 1 2];
B = [70.0111;90.005556;60.005556];

X = inv(A) * B






















