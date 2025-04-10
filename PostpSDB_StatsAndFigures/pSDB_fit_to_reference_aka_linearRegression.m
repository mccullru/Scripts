%% pSDB_fit_to_reference
%  This script is used in SDB. It creates a scatterplot of reference
%  bathymetry (y axis) vs. relative SDB (pSDB) (x axis), fits a line to the 
%  data, and displays the R^2 value and equation of fit. For example, the
%  reference bathymetry could be ICESat-2 bathymetry or multibeam 
%  echousounder (MBES) data, and the pSDB could be the ratio of logs from
%  the Stumpf algorithm. The input consists of an Excel spreadsheet where 
%  the first column contains the pSDB (relative SDB), and the second column 
%  contains the reference bathymetry.
%  C. Parrish (christopher.parrish@oregonstate.edu)
%  5/22/2022
%%
clear all; close all; clc;

%% Input parameters
inFile = strcat("E:\Thesis Stuff\SDB_ExtractedPts\Marathon_S2A_MSI_2023_02_14_16_06_29_T17RMH_L2W__RGB_SDBred_extracted.csv");
[~, filename, ~] = fileparts(inFile);

%% Read the Excel file
A = readmatrix(inFile);

%% Parse the file
sdb = A(:,5);
refBathy = A(:,3);

%% Create a scatterplot
scatter(sdb,refBathy, 'o' );
xlabel('Relative SDB, pSDB (unitless ratio)','fontSize',14);
ylabel('Reference Bathymetry (m)','fontSize',14);
title([filename 'Marathon, FL' ], 'FontWeight','normal', 'FontSize',14, 'FontWeight','bold')
set(gca, 'XDir','reverse');
set(gca, 'YDIR','reverse');

%% File a line (requres statistics toolbox)
hold on;
h1 = lsline;   % least squares line fit
h1.Color = 'r';
h1.LineWidth = 1.5;

%% Get the coefficients of the line
p2 = polyfit(get(h1,'xdata'),get(h1,'ydata'),1);

%% Display the equation of the line on the plot
xl = xlim;
yl = ylim;
xt = 0.30 * (xl(2)-xl(1)) + xl(1);
yt = 0.90 * (yl(2)-yl(1)) + yl(1);
caption = sprintf('y = %f * x + %f', p2(1), p2(2));
text(xt, yt, caption, 'FontSize', 16, 'Color', 'r');

%% Now compute and display the R2 value to the screen
RegressionLine = sdb.*p2(1) + p2(2);
% RMSE between regression line and y
RMSE = sqrt(mean((refBathy-RegressionLine).^2));
% R2 between regression line and y
SS_X = sum((RegressionLine-mean(RegressionLine)).^2);
SS_Y = sum((refBathy-mean(refBathy)).^2);
SS_XY = sum((RegressionLine-mean(RegressionLine)).*(refBathy-mean(refBathy)));
R_squared = SS_XY/sqrt(SS_X*SS_Y);
disp(['RMSE: ' num2str(RMSE)])
disp(['R2: ' num2str(R_squared)])

%caption = sprintf('R2: %f        RMSE: %f', R_squared, RMSE);
annotation('textbox', [0.72, 0.77, .1, .1], 'string', compose('R2:        %.2f\nRMSE:  %.2f',R_squared, RMSE), 'FontSize', 14, 'EdgeColor', 'none');

% Positioning the caption (adjust x and y as needed)
% xPos = 0.3;  % Relative x position (0 is far left, 1 is far right)
% yPos = .97; % Relative y position (negative value positions it below)
% text(xPos, yPos, caption, 'Units', 'normalized');

%% Display some instructions to the screen
disp('To compute the final SDB bathymetric grid from the pSDB, ');
disp('perform the following computation for every pixel in the pSDB grid:');
disp(['SDB = ' num2str(p2(1)) '*pSDB + ' num2str(p2(2))]); 
disp('Note: in Arc, this can be performed using Raster Calculator');


