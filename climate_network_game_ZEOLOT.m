%% Climate Network Games for Climate

% network and country-wise data preprocessed

clc;
clearvars;

%% Cost function construction and Examination

% Part I: Mitigation Cost Function

% sample strategy space
s_sample = 0.01 : 0.01 : 1.20;

% Pre-allocate an array for marginal costs
C_marginal = zeros(size(s_sample));
% marginal mitigiation cost function
idx_valid = (s_sample <= 0.99);
C_marginal(idx_valid) = 500*(0.2 + 0.1 .* log(s_sample(idx_valid) ./ (1 - s_sample(idx_valid))));
% clamp all future points to that same marginal cost
idx_099 = find(s_sample == 0.99, 1, 'first');
if ~isempty(idx_099)
    C_marginal_099 = C_marginal(idx_099);
    
    % For s >= 0.99, clamp to the marginal cost at s=0.99
    idx_clamp = (s_sample >= 0.99);
    C_marginal(idx_clamp) = C_marginal_099;
else
    error('No s=0.99 found in s_sample; check step size or range.')
end

% cumulative mitigation cost function
C_mitigation = cumsum(C_marginal)*0.01; % Multiply the partial sum by the step size (0.01)

% plot cost functions
figure(1)
plot(s_sample, C_mitigation, 'LineWidth', 2) % thicker line
hold on
plot(s_sample, C_marginal, 'LineWidth', 2)   % thicker line
hold off
xlabel('Mitigation Strategy')
ylabel('Marginal/Cumulative Mitigation Cost $/ton-CO2')
title('Mitigation Cost vs. Strategy')
legend('Cumulative Mitigation Cost', 'Marginal Mitigation Cost', 'Location', 'best') % 'best' finds a spot automatically



% Part II: Adaptation Cost Function (by year 2100)

% by paper (Lamboll et al., 2023), remaining carbon budget before 2C is
% 1000 Gt, given 75 year between 2100 and 2025, 13.3 Gt per year on average

% Read data from data folder
% Specify the folder name and file names
dataFolder = 'data';
csvFile = 'Countries_input_data.csv';
matFile = 'influence_Matrix.mat';

% Build full file paths
csvFilePath = fullfile(dataFolder, csvFile);
matFilePath = fullfile(dataFolder, matFile);

% Read data from the CSV file
countriesData = readtable(csvFilePath);

% Load the .mat file
loadedData = load(matFilePath);
% If the variable saved in the MAT file is named 'influence_Matrix',
% you can access it like this:
influenceMatrix = loadedData.influence_Matrix;

% Sum the values in the 5th column of 'countriesData' table: kton_CO2
emission_total = sum(countriesData{:,5})/1000000; % convert kton ton Gton

% fitting the cumulative CO2 emission with temperature rise function as
% quadratic function fitting

% Using available carbon budget by the end of century: 2100.
% Point 1: (0, 1.5) Net-zero cumulative emission, 1.5 degree C T-increase
% Point 2: (13.3/emission_total, 2.0) use up carbon budget, 2.0 degree C T-increase
% This is around 1000 Gt remaining carbon budget divided by 75 years.
% Point 3: (1, 3.2) emission level does not change, 3.2 degree C T-increase

% Given data
x1 = [0, 13.3/emission_total, 1];  % x-values
y1 = [1.5, 2.0, 3.2]; % y-values

% Fit a 2nd-degree polynomial (quadratic) to the data
p1 = polyfit(x1, y1, 2);

% Evaluate the polynomial on a finer grid for plotting
x1Fit = linspace(0, 1.2, 100);
y1Fit = polyval(p1, x1Fit);

% Plot the original data and the fitted curve
figure(2);
plot(x1, y1, 'o', x1Fit, y1Fit, '-');
xlabel('Global Emission Change');
ylabel('Temperature Rise');
title('Quadratic Fit: T-rise vs. Global GHG Emission');
legend('Data points','Quadratic fit','Location','best');


% fitting the temperature rise with economic loss as function of
% quadratic function fitting

% Point 1: (0, 0.02) Net-zero cumulative emission, 1.5 degree C T-increase,
% leading to 2% economic loss

% Point 2: (2.0, 0.10) use up carbon budget, 2.0 degree C
% T-increase, leading to 10% economic loss

% Point 3: (3.2, 0.25) emission level does not change, 3.2 degree C
% T-increase, leading to 25% economic loss

% Given data
x2 = [0, 2.0, 3.2];  % x-values
y2 = [0.02, 0.10, 0.25]; % y-values

% Fit a 2nd-degree polynomial (quadratic) to the data
p2 = polyfit(x2, y2, 2);
p2_low = polyfit(x2, y2./2, 2);
p2_high = polyfit(x2, y2.*2, 2);

% Evaluate the polynomial on a finer grid for plotting
x2Fit = linspace(0, 5, 100);
y2Fit = polyval(p2, x2Fit);

% Plot the original data and the fitted curve
figure(3);
plot(x2, y2, 'o', x2Fit, y2Fit, '-');
xlabel('Temperature Rise');
ylabel('Economic Loss of GDP');
title('Quadratic Fit: GDP-loss vs. T-rise');
legend('Data points','Quadratic fit','Location','best');

% Now, directly mapping emission to economic loss
% Define a grid for Global Emission Change
x3Fit = linspace(0.01, 1.2, 120);

% First map emission to temperature
tempRise = polyval(p1, x3Fit);

% Then map the resulting temperature to economic loss
gdpLoss = polyval(p2, tempRise);

% Define upper and lower bounds
lowerBound = 0.5 * gdpLoss * 100;
upperBound = 2.0 * gdpLoss * 100;

% Create x values for the shaded region
x_fill = [x3Fit, fliplr(x3Fit)];
y_fill = [upperBound, fliplr(lowerBound)];

% Plot the direct emission -> loss mapping
figure(4);
fill(x_fill, y_fill, [0.8, 0.8, 1], 'EdgeColor', 'none'); % light blue shade
hold on;
plot(x3Fit, gdpLoss*100, '-','LineWidth', 2);
% Add vertical lines
xline(0.26, '--r', 'LineWidth', 1.5);  % 2°C threshold
xline(0.75, '--r', 'LineWidth', 1.5);  % 3°C threshold

% Add text annotations
text(0.26 + 0.01, max(upperBound)*0.75, '2°C Threshold', 'Color', 'r', ...
     'FontSize', 10, 'Rotation', 90, 'VerticalAlignment', 'top');
text(0.75 + 0.01, max(upperBound)*0.75, '3°C Threshold', 'Color', 'r', ...
     'FontSize', 10, 'Rotation', 90, 'VerticalAlignment', 'top');
xlabel('Global Normalized Emission');
ylabel('Economic Loss of GDP (%)');
title('Direct Mapping: GDP-loss vs. Global GHG Emission');


%% Main Simulation data settings

% declare all inputs

% define strategy space, 1*120 double
s_space = s_sample; % from 0.01 to 1.20, meaning emission reduction compared to current level

% Network of Influence matrix between different countries, 209*209 double
network = influenceMatrix;

% GDP of each country, 209*1 double
country_GDP = countriesData{:,2}*1000*1000*1000; % convert from billion to $

% CO2 GHG emissions of each country, 209*1 double
country_CO2 = countriesData{:,5}*1000; % convert from kton to ton

% network sensitivity of each country, 209*1 double
country_network_sens = countriesData{:,4};

%% Paras

% parameters compiling
params.s_space               = s_space;               % 1×120 vector
params.C_mitigation          = C_mitigation;          % Function handle or array
params.network               = network;               % 209×209 matrix
params.country_CO2           = country_CO2;           % 209×1
params.country_GDP           = country_GDP;           % 209×1
params.country_network_sens  = country_network_sens;  % 209×1
params.emission_total        = emission_total;        % Scalar
params.p1                    = p1;                    % Polynomial coefficients (vector)
params.p2                    = p2;                    % Polynomial coefficients (vector)

%% Strategy Implication plot

% Extend net-zero year beyond 2100, up to 2150
net_zero_year = 2025:1:2200;

% Initialize normalized emission vector
emissions = zeros(size(net_zero_year));

for i = 1:length(net_zero_year)
    y = net_zero_year(i);
    T = y - 2025;

    if y <= 2100
        % Emissions reach zero by 2100: triangle
        emissions(i) = (y - 2025) / 150;
    else
        % Emissions do NOT reach zero by 2100: trapezoid
        emissions(i) = 1 - 75 / (2 * T);
    end
end

% Plot
figure;
plot(net_zero_year, emissions, 'LineWidth', 2);
xlabel('Implied Net-Zero Year');
ylabel('Remaining Emissions (2025~2100) = 1 – Strategy');
%title('Normalized Emissions vs. Net-Zero Year');
grid on;
hold on;

% Strategy-to-emission conversions
e_2050 = 1 - (1 - (2050 - 2025)/150);  % = 0.1667
e_2060 = 1 - (1 - (2060 - 2025)/150);  % = 0.2333
e_2deg = 0.2597;                   % 2 degree C emission limit
T_2deg = 2*(2100-2025)*e_2deg;
y_2deg = 2025 + T_2deg;

% Nash Equilibrium Emission Level
e_nash = 0.7544;
T_nash = 75 / (2 * (1 - e_nash));
y_nash = 2025 + T_nash;


% Highlight 2050 and 2060 targets
xline(2050, '--b', 'Net-Zero 2050', 'LabelVerticalAlignment', 'top', 'LineWidth', 1.5);
yline(e_2050, '--b');
plot(2050, e_2050, 'ob', 'MarkerFaceColor', 'b');

xline(2060, '--m', 'Net-Zero 2060', 'LabelVerticalAlignment', 'top', 'LineWidth', 1.5);
yline(e_2060, '--m');
plot(2060, e_2060, 'om', 'MarkerFaceColor', 'm');

% Highlight 2°C target with darker green
yline(e_2deg, '--', '2°C Target - all pledges + network-induced effect', ...
    'Color', [0, 0.5, 0], 'LabelHorizontalAlignment', 'right', ...
    'LabelVerticalAlignment', 'top', 'LineWidth', 1.5);
plot(y_2deg, e_2deg, 'd', 'MarkerFaceColor', [0, 0.5, 0], 'MarkerEdgeColor', 'k');

% Plot Nash Equilibrium
yline(e_nash, '--', '3°C Nash Equilibrium (38.6 Gt/year)', ...
    'Color', [0.2, 0.3, 0.8], 'LineWidth', 1.5, ...
    'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'top');

plot(y_nash, e_nash, 'd', 'MarkerFaceColor', [0.2, 0.3, 0.8], 'MarkerEdgeColor', 'k');
%text(y_nash + 1, e_nash, sprintf('NE (~%.1f)', y_nash), 'Color', [0.2, 0.3, 0.8]);

% Reference lines
xline(2100, '--r', '2100 Horizon');
% yline(0.5, '--k', '50% Emissions Remaining');

%legend('Emission Trajectory', '2050 Target', '2060 Target', '2°C Target', 'Location', 'northeast');
ylim([0,1]);

% 2) compute the fractional positions for 0,10,…,50
tickVals = 0:10:50;                   % the Gt values you want
yt = tickVals / emission_total;      % map to [0,1] fractions

% 3) apply ticks and labels
set(gca, ...
    'YTick',      yt, ...
    'YTickLabel', arrayfun(@(x)sprintf('%d',x), tickVals, 'UniformOutput',false) ...
);

% 4) (optional) clarify units in your label
ylabel('Avg. Global Emissions by 2100 (Gt/year)');


%% Test run functions

% construct initial strategy
s_initial = 40 * ones(length(country_CO2),1);

[Cost_total, Cost_mitigation, Cost_economic, Cost_network] = computeCost(s_initial, params);

s_next = bestResponseUpdate(s_initial, params);

%% convergence check

% construct initial strategy
s_initial = randi([1 99], length(country_CO2), 1);

maxIter = 365;
[s_eq, globalCost, globalEmission] = runToConvergence(s_initial, params, maxIter);

fprintf('Global Cost at equilibrium: %.4e\n', globalCost);
fprintf('Global Emission at equilibrium: %.4f\n', globalEmission);

%% prepare for Zeolot case
% no zeolot case
zeolot_none = zeros(size(s_initial));

% USA is zeolot
zeolot_USA = zeros(size(s_initial));
zeolot_USA(200) = 1;

% China is zeolot
zeolot_China = zeros(size(s_initial));
zeolot_China(39) = 1;

% EU is Zeolot
EU_index = [11, 18, 28, 44, 47, 48, 50, 59, 65, 66, 71, 74, 86, 92, 94,...
            107, 112, 113, 120, 136, 152, 153, 155, 167, 168, 173, 180]+1;
% doble check EU country list
EU_Data = countriesData(EU_index, 1);  % Extract the first column for the specified rows
% After confirmation
zeolot_EU = zeros(size(s_initial));
zeolot_EU(EU_index) = 1;

% Other 2050
other_2050_index = [8, 11, 27, 35, 97, 103, 172, 199];
% doble check EU country list
other_2050 = countriesData(other_2050_index, 1);  % Extract the first column for the specified rows
% After confirmation
zeolot_other2050 = zeros(size(s_initial));
zeolot_other2050(other_2050_index) = 1;
disp('Countries with 2050 net-zero pledges:');
disp(other_2050);

% Other 2060
other_2060_index = [90, 157, 161];
% doble check EU country list
other_2060 = countriesData(other_2060_index, 1);  % Extract the first column for the specified rows
% After confirmation
zeolot_other2060 = zeros(size(s_initial));
zeolot_other2060(other_2060_index) = 1;
disp('Countries with 2060 net-zero pledges:');
disp(other_2060);

% Other 2070
other_2070_index = 89;
% doble check EU country list
other_2070 = countriesData(other_2070_index, 1);  % Extract the first column for the specified rows
% After confirmation
zeolot_other2070 = zeros(size(s_initial));
zeolot_other2070(other_2070_index) = 1;
disp('Countries with 2070 net-zero pledges:');
disp(other_2070);

zeolot_other = zeolot_other2050 + zeolot_other2060 + zeolot_other2070;

%% Network Graph Visualization

G_raw = influenceMatrix;
G_raw(G_raw < 0.03) = 0;
G = graph(G_raw, 'upper'); % graph object
nodeSizes = rescale(country_GDP.^0.5, 5, 50); % node properties
countryNames = countriesData{:,1};

[~, idxTop] = maxk(country_GDP, 28);   % idx = indices of top 20 countries
% Initialize all labels empty
labels = repmat({''}, numel(country_GDP), 1);
% Fill only top 20 labels
labels(idxTop) = countryNames(idxTop);

colorGroup = zeros(size(country_GDP));  % default 0 = others
colorGroup(39) = 1;             % USA
colorGroup(200) = 2;              % China
colorGroup(EU_index) = 3;        % EU

colors = [
    0.5 0.5 0.5;   % 0 -> Other countries (dark gray)
    0.8 0.6 0.0;   % 1 -> USA (dark gold)
    0.0 0.6 0.0;   % 2 -> China (dark green)
    0.0 0.4 0.7;   % 3 -> EU (steel blue)
];

% manually update the label country names
labels(200) = {'USA'};         % Index 200 is United States
labels(39)  = {'China'};        % Index 39 is China (can leave as is, or shorten if needed)
labels(103)  = {'South Korea'};
labels(137)  = {'Netherland'};
labels(157)  = {'Russia'};
labels(195) = {'Türkiye'};
labels(199) = {'UK'};
labels(198) = {'UAE'};
labels(153) = {'Poland'};

% Add space padding to all non-empty labels
for i = 1:numel(labels)
    if ~isempty(labels{i})
        labels{i} = ['  ' labels{i}];  % Add 3 spaces in front
    end
end


figure();
p = plot(G, 'Layout', 'force', 'MarkerSize', nodeSizes);
% colorbar
% Center after spreading
p.XData = (p.XData - mean(p.XData)) * 2;  
p.YData = (p.YData - mean(p.YData)) * 2;
title('Trade Network with GDP Sizes')

% Set node colors manually
nodeColors = colors(colorGroup + 1, :);  % +1 because MATLAB indices start at 1
p.NodeColor = nodeColors;

% Apply labels
% p.NodeLabel = countryNames;
p.NodeLabel = labels;
p.NodeFontSize = 10;      % Increase font size (default is 8 usually)
p.NodeFontWeight = 'bold'; % Make label text bold


%% Define Zeolots for major players

% Combination: USA + China
zeolot_USA_China = double( (zeolot_USA==1) | (zeolot_China==1) );

% Combination: USA + EU
zeolot_USA_EU = double( (zeolot_USA==1) | (zeolot_EU==1) );

% Combination: China + EU
zeolot_China_EU = double( (zeolot_China==1) | (zeolot_EU==1) );

% Combination: USA + China + EU
zeolot_USA_China_EU = double( (zeolot_USA==1) | (zeolot_China==1) | (zeolot_EU==1) );

% Construct an n x 8 matrix, where n = length(s_initial)
zeolotOptions = [zeolot_none, ...           % Column 1: non-zealot option
                 zeolot_USA, ...            % Column 2: USA only
                 zeolot_China, ...          % Column 3: China only
                 zeolot_EU, ...             % Column 4: EU only
                 zeolot_USA_China, ...      % Column 5: USA + China
                 zeolot_USA_EU, ...         % Column 6: USA + EU
                 zeolot_China_EU, ...       % Column 7: China + EU
                 zeolot_USA_China_EU];      % Column 8: USA + China + EU


%% Climate zeolot simulation

% Assume zeolotOptions is an n x 8 matrix, where each column is one zealot
% configuration option, and country_CO2, params, and maxIter are defined.

nOptions = size(zeolotOptions, 2);  % Should be 8+1 options.
nRuns = 20;                        % Number of random runs per zealot option.
maxIter = 365;

% Preallocate storage:
% For the vector output final_Cost_Country, use a cell array.
finalCostCountry_all = cell(nOptions, nRuns);
finalEquiCountry_all = cell(nOptions, nRuns);
% For the scalar outputs, use matrices.
costFinal_all    = zeros(nOptions, nRuns);
emissionFinal_all = zeros(nOptions, nRuns);

tic;  % Start timer

for opt = 1:nOptions
    % Get the current zealot configuration (column vector of zeros and ones).
    currentZealot = zeolotOptions(:, opt);
    fprintf('Running zealot option %d\n', opt);
    
    for r = 1:nRuns
        % 1) Generate a random initial strategy for each run.
        %    For non-zealot positions, choose a random integer in [1,99].
        s_initial = randi([1 99], length(country_CO2), 1);
        
        % 2) For positions flagged as zealots, set the strategy to 80.
        s_initial(currentZealot == 1) = 80;
        
        % 3) Run the best-response convergence algorithm with zealot tracking.
        %    This function returns:
        %    final_Cost_Country (vector), costFianl (scalar),
        %    emission_Final (scalar), and additional outputs.
        [final_Cost_Country, costFianl, emission_Final, s_equilibrium, costHistory, emissionHistory] = ...
            runToConvergenceZealots(s_initial, params, maxIter, currentZealot);
        
        % 4) Store the outputs.
        finalCostCountry_all{opt, r} = final_Cost_Country;
        finalEquiCountry_all{opt, r} = s_equilibrium;
        costFinal_all(opt, r)       = costFianl;
        emissionFinal_all(opt, r)     = emission_Final;
        
        % Print progress every 10 runs per option.
        if mod(r, 10) == 0
            fprintf('  Completed run %d of %d for option %d\n', r, nRuns, opt);
        end
    end
end

elapsedTime = toc;  % Stop timer
fprintf('All %d zealot options with %d runs each finished in %.2f seconds.\n', nOptions, nRuns, elapsedTime);


%% Real world 2050-2060 pledges scenario

% Combination: USA + China + EU
zeolot_2050 = double( (zeolot_USA==1) | (zeolot_EU==1) | (zeolot_other2050==1));
zeolot_2060 = double( (zeolot_China==1) | (zeolot_other2060==1) );
zeolot_2070 = double( (zeolot_other2070==1));

currentZealot = zeolot_2050 + zeolot_2060 + zeolot_2070;


nOptions = size(1, 2);  % Should be 8+1 options.
nRuns = 20;                        % Number of random runs per zealot option.
maxIter = 365;

% Preallocate storage:
% For the vector output final_Cost_Country, use a cell array.
finalCostCountry_real = cell(nOptions, nRuns);
finalEquiCountry_real = cell(nOptions, nRuns);
% For the scalar outputs, use matrices.
costFinal_real    = zeros(nOptions, nRuns);
emissionFinal_real = zeros(nOptions, nRuns);

for r = 1:nRuns
    % 1) Generate a random initial strategy for each run.
    %    For non-zealot positions, choose a random integer in [1,99].
    s_initial = randi([1 99], length(country_CO2), 1);
    
    % 2) For positions flagged as zealots, set the strategy to 83 for net-zero 2050.
    s_initial(zeolot_2050 == 1) = 83;
    % set 76 for net-zero 2060.
    s_initial(zeolot_2060 == 1) = 76;
    % set 70 for net-zero 2070.
    s_initial(zeolot_2070 == 1) = 70;
    
    % 3) Run the best-response convergence algorithm with zealot tracking.
    %    This function returns:
    %    final_Cost_Country (vector), costFianl (scalar),
    %    emission_Final (scalar), and additional outputs.
    [final_Cost_Country, costFianl, emission_Final, s_equilibrium, costHistory, emissionHistory] = ...
        runToConvergenceZealots(s_initial, params, maxIter, currentZealot);
    
    % 4) Store the outputs.
    finalCostCountry_real{nOptions, r} = final_Cost_Country;
    finalEquiCountry_real{nOptions, r} = s_equilibrium;
    costFinal_real(nOptions, r)       = costFianl;
    emissionFinal_real(nOptions, r)     = emission_Final;
    
    % Print progress every 10 runs per option.
    if mod(r, 10) == 0
        fprintf('  Completed run %d of %d for option %d\n', r, nRuns, nOptions);
    end
end

%{
%% check if zeolot have incentive to defect

% 0) Sensitivity‐boost hyperparameter
sensBoost = 1;    % 20% boost when testing boosted case

% 1) Baseline run
[finalCost_orig, ~, ~, ~, ~, ~] = ...
    runToConvergenceZealots(s_initial, params, maxIter, currentZealot);

zealotIdx = find(currentZealot);
fprintf('=== Zealot Defection Incentive Test ===\n');
fprintf('Sensitivity boost = +%.0f%%\n\n', sensBoost*100);

% 2) Loop over each zealot
for k = 1:numel(zealotIdx)
    i = zealotIdx(k);
    name     = countryNames{i};
    origCost = finalCost_orig(i);
    
    % prepare defected flag
    zealotDefect = currentZealot;
    zealotDefect(i) = 0;
    
    % ---- (a) No‐boost defection ----
    [fc_nb, ~, ~, ~, ~, ~] = ...
        runToConvergenceZealots(s_initial, params, maxIter, zealotDefect);
    cost_nb = fc_nb(i);
    pct_nb  = (cost_nb - origCost)/origCost*100;
    if pct_nb < 0
        inc_nb = 'Yes';
    else
        inc_nb = 'No';
    end
    
    % ---- (b) +boost defection ----
    paramsBoost = params;
    paramsBoost.country_network_sens(i) = ...
        params.country_network_sens(i)*(1+sensBoost);
    [fc_b, ~, ~, ~, ~, ~] = ...
        runToConvergenceZealots(s_initial, paramsBoost, maxIter, zealotDefect);
    cost_b = fc_b(i);
    pct_b  = (cost_b - origCost)/origCost*100;
    if pct_b < 0
        inc_b = 'Yes';
    else
        inc_b = 'No';
    end
    
    % ---- Print results ----
    fprintf('Country: %s\n', name);
    fprintf('  Without boost:    Δcost = %+6.2f%%   Incentive to defect? %s\n', ...
            pct_nb, inc_nb);
    fprintf('  With +%.0f%% boost: Δcost = %+6.2f%%   Incentive to defect? %s\n\n', ...
            sensBoost*100, pct_b, inc_b);
end

%% introducing new zeolots

% 0) incremental commitment
s_increment = 0;

% 1) Baseline run to get equilibrium strategies
[finalCost_orig, costFinal_orig, emissionFinal_orig, s_equilibrium, costHistory, emissionHistory] = ...
    runToConvergenceZealots(s_initial+s_increment, params, maxIter, currentZealot);
% Identify which countries “committed” by looking at equilibrium s > 70
newZealotMask = (s_equilibrium > 65);
fprintf('Total fixed-zealots (s_eq > 65): %d countries\n', sum(newZealotMask));

% 2) Now test _original_ zealots for incentive to defect
origZealotIdx = find(currentZealot);
fprintf('\n=== Original-Zealot Defection Test ===\n');
fprintf('(All others with s_eq>70 held fixed as zealots)\n\n');

for k = 1:numel(origZealotIdx)
    i = origZealotIdx(k);
    name = countryNames{i};
    
    % build the “defected” zealot mask: start from newZealotMask
    zealotDefect = newZealotMask;
    % flip off this one original zealot
    zealotDefect(i) = 0;
    
    % rerun convergence with the updated zealot mask
    [finalCost_def, ~, ~, ~, ~, ~] = ...
        runToConvergenceZealots(s_equilibrium+s_increment, params, maxIter, zealotDefect);
    
    % compute percentage change for this country
    cost_before = finalCost_orig(i);
    cost_after  = finalCost_def(i);
    pctChange   = (cost_after - cost_before) / cost_before * 100;
    
    % decide if defecting lowers cost
    if pctChange < 0
        incentive = 'Yes';
    else
        incentive = 'No';
    end
    
    % print result
    fprintf('Country: %-15s   Δcost = %+6.2f%%   Incentive to defect? %s\n', ...
            name, pctChange, incentive);
end
%}
%% Plot over time by 2100

% Build full file paths
GHGhistoryFile = 'total-ghg-emissions_by_year.csv';
GHGhistoryFilePath = fullfile(dataFolder, GHGhistoryFile);

% Read data from the CSV file
GHGhistoryData = readtable(GHGhistoryFilePath);
% read columns
years_history = GHGhistoryData{:,3};
GHG_history = GHGhistoryData{:,4}/1e9;
%%
figure();
% plot history result first
plot(years_history, GHG_history, 'LineWidth', 2, 'HandleVisibility', 'off');
xlabel('Year');
ylabel('Global Total Emission (Gt/year)');
%title('Global GHG emission trend by scenarios');
grid on;
hold on;

% Define future years
years_future = 2023:2100;

% Define starting emission
GHG_2023 = GHG_history(end);

% Compute GHG_2100 to achieve average = 0.7544 * GHG_2023
target_avg_ratio = 0.7544;
GHG_2100 = (2 * target_avg_ratio - 1) * GHG_2023;

% Create linear transition from 2023 to 2100
GHG_future = linspace(GHG_2023, GHG_2100, length(years_future));

% Sensitivity bounds on average emission
% low case
params_low_adapt = params;
params_low_adapt.p2 = p2_low;
[s_eq_low, globalCost_low, globalEmission_low] = ...
    runToConvergence(s_initial, params_low_adapt, maxIter);
% high case
params_high_adapt = params;
params_high_adapt.p2 = p2_high;
[s_eq_high, globalCost_high, globalEmission_high] = ...
    runToConvergence(s_initial, params_high_adapt, maxIter);


avg_low = globalEmission_low;
avg_high = globalEmission_high;


% Compute GHG_2100 values to match the average constraints
GHG_2100_low = (2 * avg_low - 1) * GHG_2023;
GHG_2100_high = (2 * avg_high - 1) * GHG_2023;
% Generate the emission paths (linear)
GHG_low = linspace(GHG_2023, GHG_2100_low, length(years_future));
GHG_high = linspace(GHG_2023, GHG_2100_high, length(years_future));

% Shade the area between the two curves
fill([years_future, fliplr(years_future)], [GHG_low, fliplr(GHG_high)], ...
    [0.5, 0, 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% Plot NE future emissions line
plot(years_future, GHG_future, 'Color', [0.5, 0, 0], 'LineStyle', '--', 'LineWidth', 2,...
    'DisplayName', '3°C: Nash Equilibrium');  % dark red/brown

%
% Sensitivity bounds for good scenario
[final_Cost_Country_low, costFinal_low, emission_Final_low, ...
 s_eq_low, costHistory_low, emissionHistory_low] = ...
    runToConvergenceZealots(s_initial, params_low_adapt, maxIter, currentZealot);
[final_Cost_Country_high, costFinal_high, emission_Final_high, ...
 s_eq_high, costHistory_high, emissionHistory_high] = ...
    runToConvergenceZealots(s_initial, params_high_adapt, maxIter, currentZealot);

avg_good_low = emission_Final_low-0.01;
avg_good_high = emission_Final_high+0.01;

% Compute end emissions
GHG_2100_good_low = (2 * avg_good_low - 1) * GHG_2023;
GHG_2100_good_high = (2 * avg_good_high - 1) * GHG_2023;
% Generate emission paths
GHG_good_low = linspace(GHG_2023, GHG_2100_good_low, length(years_future));
GHG_good_high = linspace(GHG_2023, GHG_2100_good_high, length(years_future));
%{
 
Fill shaded area between good scenario bounds
fill([years_future, fliplr(years_future)], ...
     [GHG_good_low, fliplr(GHG_good_high)], ...
     [0, 0.5, 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
%}

% Define average ratio for good scenario
good_avg_ratio = 0.2597;
% Calculate the target end emission in 2100
GHG_2100_good = (2 * good_avg_ratio - 1) * GHG_2023;
% Generate linear emissions from 2023 to 2100
GHG_good = linspace(GHG_2023, GHG_2100_good, length(years_future));
% Plot the good scenario in dark green
%plot(years_future, GHG_good, 'Color', [0, 0.5, 0], 'LineStyle', '-', 'LineWidth', 1,...
%    'DisplayName', '2°C Negative: all pldges + network');
%}

good_avg_ratio = 0.2597;

ratios = [avg_good_low, good_avg_ratio, avg_good_high];
color_shade = [0, 0.5, 0];% [0.8, 0.4, 0];    % dark orange if needed
years_total = 2023:2100;
T = length(years_total);        % total duration = 78 years


% Compute net-zero years
net_zero_years = round(2 * T * ratios + 2022);
y_low = net_zero_years(1);
y_mid = net_zero_years(2);
y_high = net_zero_years(3);

% Construct trajectories
% Low
y1 = 2023:y_low;
y2 = (y_low+1):2100;
emiss_low = [linspace(GHG_2023, 0, length(y1)), zeros(1, length(y2))];

% High
y1 = 2023:y_high;
y2 = (y_high+1):2100;
emiss_high = [linspace(GHG_2023, 0, length(y1)), zeros(1, length(y2))];

% Mid
y1 = 2023:y_mid;
y2 = (y_mid+1):2100;
emiss_mid = [linspace(GHG_2023, 0, length(y1)), zeros(1, length(y2))];

% --- Shade between low and high ---
fill([years_total, fliplr(years_total)], ...
     [emiss_low, fliplr(emiss_high)], ...
     color_shade, 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
     'HandleVisibility', 'off');

% --- Plot central trajectory ---
plot(years_total, emiss_mid, 'Color', color_shade, ...
     'LineWidth', 1, 'DisplayName', '2°C Net-zero: all pledges + network');



legend('Location', 'best');
xlim([1990,2100]);
ylim([-5,60]);






%% plot Global total result under Zeolot options

% Compute statistics for costFinal_all and emissionFinal_all
meanCosts = mean(costFinal_all, 2);    % Mean cost across runs for each zeolot option
meanEmissions = mean(emissionFinal_all, 2);  % Mean emission across runs for each zeolot option

% New ordering for display:
new_order = [1, 2, 4, 3, 6, 5, 7, 8];
zeolotOptions_new = zeolotOptions(:,new_order);


% Reorder the computed means:
meanCosts_new = meanCosts(new_order);
meanEmissions_new = meanEmissions(new_order);

% Update the option names list:
optionNames = {'None', 'USA', 'EU', 'China', 'USA+EU', 'USA+China', 'China+EU', 'USA+China+EU'};

% Plot equilibrium total cost using the new order:
figure('Position', [100, 100, 1200, 600]);
subplot(1,2,1);
bar(meanCosts_new, 'FaceColor', 'flat');  
set(gca, 'XTick', 1:length(optionNames), 'XTickLabel', optionNames);
xlabel('Zealot Option');
ylabel('Global Equilibrium Total Cost');
title('Equilibrium Global Total Cost for Zealot Options');

% Plot equilibrium emission using the new order:
subplot(1,2,2);
bar(meanEmissions_new, 'FaceColor', 'flat');
set(gca, 'XTick', 1:length(optionNames), 'XTickLabel', optionNames);
xlabel('Zealot Option');
ylabel('Global Equilibrium Normalized Emission');
title('Equilibrium Normalized Global Emission for Zealot Options');


hold on;  % Retain the current plot for the line and text

% Draw a thicker horizontal line at y = 0.2
ax = gca;
xlims = get(ax, 'XLim');
line(xlims, [0.2 0.2], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);

% Place the label 'Zealot target' rotated vertically near the line.
% Adjust the x-position slightly left from the right end, and y-position slightly above
label_x = xlims(2) - 0.03 * diff(xlims);  % shift a bit left from the far right end
label_y = 0.2 - 0.02;                     % slightly above the line
text(label_x, label_y, 'Zealot target', 'Rotation', 90, ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'Color', 'r');

hold off;


%% Check zeolot incentive to hold



% Step 1: Compute the average (element-wise) cost vector for the no-zealot case.
nRuns = size(finalCostCountry_all, 2);
nCountries = length(finalCostCountry_all{1,1});  % assuming all cost vectors have the same length

baselineAvg = zeros(nCountries, 1);
for r = 1:nRuns
    baselineAvg = baselineAvg + finalCostCountry_all{1, r};
end
baselineAvg = baselineAvg / nRuns;

% Step 2: For each zealot option (rows 2 through 8), for each run,
%         compute the percentage of zealot countries whose cost is lower than
%         the baseline average at the corresponding entries.
nOptions = size(finalCostCountry_all, 1);  % should be 8; row 1 is the baseline.
percentageLower = zeros(nOptions - 1, 1);  % output: 7x1 vector

for opt = 2:nOptions
    runFractions = zeros(nRuns, 1);
    % Identify the indices of the countries that are flagged as zealots
    % for the current option.
    zealotIdx = find(zeolotOptions_new(:, opt));
    
    for r = 1:nRuns
        currentCost = finalCostCountry_all{opt, r};  % cost vector for current run & option
        % Count the number of zealot countries with cost lower than the baseline average
        nLower = sum(currentCost(zealotIdx) < baselineAvg(zealotIdx));
        % Compute the fraction for this run
        runFractions(r) = nLower / length(zealotIdx);
    end
    % Average the run fractions over all runs and convert to a percentage
    percentageLower(opt - 1) = mean(runFractions) * 100;
end

disp(percentageLower)


%% Waterfall chart for USA-EU-China zeolot case

% find two equilibrium strategies
s_eq_non_zeolot = s_space(finalEquiCountry_all{1,1});
s_eq_triple_zeolot = s_space(finalEquiCountry_all{8,1});

% non-zeolot total emission
emission_non_zeolot = emission_total - dot(s_eq_non_zeolot, country_CO2)/1e9;

% emission triple zeolot
emission_triple_zeolot = emission_total - dot(s_eq_triple_zeolot, country_CO2)/1e9;

% USA reduction
Decarb_USA = dot((s_eq_non_zeolot - s_eq_triple_zeolot)'.*country_CO2, zeolotOptions_new(:,2))/1e9;
% EU reduction
Decarb_EU = dot((s_eq_non_zeolot - s_eq_triple_zeolot)'.*country_CO2, zeolotOptions_new(:,3))/1e9;
% China reduction
Decarb_China = dot((s_eq_non_zeolot - s_eq_triple_zeolot)'.*country_CO2, zeolotOptions_new(:,4))/1e9;
% network reduction
Decarb_Network = emission_triple_zeolot - (emission_non_zeolot+Decarb_USA+Decarb_EU+Decarb_China);


y_changes = [emission_non_zeolot; Decarb_USA; Decarb_EU; Decarb_China; Decarb_Network];

figure;
wfall(gca, y_changes);
title('Waterfall Chart of Emission Reductions');
ylabel('Global Total Emission: Gt/year');
set(gca, 'XTick', 1:length(y_changes)+1, 'XTickLabel', ...
    {'Non-Zeolot','USA','EU','China','Network Reduction','Triple Zealot'});
ylim([0,40]);



%% Waterfall chart for net-zero pledges zeolot case

% find two equilibrium strategies
s_eq_non_zeolot = s_space(finalEquiCountry_all{1,1});
s_eq_pledge_zeolot = s_space(finalEquiCountry_real{1,1});

% non-zeolot total emission
emission_non_zeolot = emission_total - dot(s_eq_non_zeolot, country_CO2)/1e9;

% emission triple zeolot
emission_pledge_zeolot = emission_total - dot(s_eq_pledge_zeolot, country_CO2)/1e9;

% USA reduction
Decarb_USA = dot((s_eq_non_zeolot - s_eq_pledge_zeolot)'.*country_CO2, zeolotOptions_new(:,2))/1e9;
% EU reduction
Decarb_EU = dot((s_eq_non_zeolot - s_eq_pledge_zeolot)'.*country_CO2, zeolotOptions_new(:,3))/1e9;
% China reduction
Decarb_China = dot((s_eq_non_zeolot - s_eq_pledge_zeolot)'.*country_CO2, zeolotOptions_new(:,4))/1e9;
% Other net-zero reduction
Decarb_other = dot((s_eq_non_zeolot - s_eq_pledge_zeolot)'.*country_CO2, zeolot_other)/1e9;
% network reduction
Decarb_Network = emission_pledge_zeolot - (emission_non_zeolot+Decarb_USA+Decarb_EU+Decarb_China+Decarb_other);


y_changes = [emission_non_zeolot; Decarb_USA; Decarb_EU; Decarb_China; Decarb_other; Decarb_Network];

figure;
wfall(gca, y_changes);
%title('Waterfall Chart of Emission Reductions: Current Plegdes');
ylabel('Global Average Emission: Gt/year until 2100');
set(gca, 'XTick', 1:length(y_changes)+1, 'XTickLabel', ...
    {'Nash Equilibrium','USA','EU','China','Other Net-zero','Rest of World','Estimated Emission'});
ylim([0,40]);
yline(e_2deg*emission_total, '--', '2°C Target', ...
    'Color', [0, 0.5, 0], 'LabelHorizontalAlignment', 'center', ...
    'LabelVerticalAlignment', 'top', 'LineWidth', 1.5);
set(gca, 'FontSize', 13);  % Adjust the number as needed

%% Emission reduction by country plot

emission = country_CO2/1e9;
level1 = 1 - s_eq_pledge_zeolot;
level2 = s_eq_pledge_zeolot - s_eq_non_zeolot;
level3 = s_eq_non_zeolot;  % Each row sums to 1

% make sure level1 and zeolot_EU are column vectors:
lv1 = level1(:);
isEU = zeolot_EU(:);

% create a two‐column key: [primary sort by level1; secondary sort by -isEU]
% (negating isEU means that 1’s — EU countries — sort before 0’s)
key = [lv1, -isEU];

% sortrows returns the sorted rows and the index mapping
[~, idx] = sortrows(key, [1 2]);   % first column asc (level1), then second asc (-isEU)
% Reorder all vectors accordingly
emission_sorted = emission(idx);
level1 = level1(idx);
level2 = level2(idx);
level3 = level3(idx);

x_edges = [0, cumsum(emission_sorted)'];

colors = [ ...
    0.10 0.33 0.65;  % level3 (dark)
    0.43 0.69 0.84;  % level2
    0.78 0.87 0.97;  % level1 (light)
];

%{
colors = [0.8,0.4,0.4;  % level1 color
          0.4 0.8 0.4;  % level2 color
          0.2 0.6 0.8]; % level3 color
%}

figure; hold on;

fig = gcf;                % handle to current figure
fig.Units   = 'pixels';   % work in pixels
pos = fig.Position;       % pos = [left bottom width height]

% double the width
fig.Position = [pos(1), pos(2), pos(3)*2, pos(4)*1.2];

for i = 1:length(emission)
    x0 = x_edges(i);
    x1 = x_edges(i+1);

    % Compute lighter edge colors by mixing with white
    edge1 = colors(1, :) * 0.7 + 0.3;
    edge2 = colors(2, :) * 0.7 + 0.3;
    edge3 = colors(3, :) * 0.7 + 0.3;

    % Bottom level (level1)
    y0 = 0;
    y1 = level1(i);
    h1 = patch([x0 x1 x1 x0], [y0 y0 y1 y1], colors(1, :),...
        'EdgeColor', edge1);

    % Middle level (level2)
    y0 = y1;
    y1 = y0 + level2(i);
    h2 = patch([x0 x1 x1 x0], [y0 y0 y1 y1], colors(2, :),...
        'EdgeColor', edge2);

    % Top level (level3)
    y0 = y1;
    y1 = y0 + level3(i);
    h3 = patch([x0 x1 x1 x0], [y0 y0 y1 y1], colors(3, :),...
        'EdgeColor', edge3);

    % Only store handles once (from first country)
    if i == 1
        h(1) = h1;
        h(2) = h2;
        h(3) = h3;
    end
end

xlim([0 x_edges(end)]);
%ylim([0 1]);

set(gca, ...
    'YTick',    0:0.1:1, ...
    'YTickLabel', {'0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'}, ...
    'YLim',     [0 1]);

xlabel('Cumulative Current Emissions by Country (Gt/year)');
ylabel('Avg. Future Emissions (% of current Baseline)');
%title('Variable Width Stacked Bar with Three Levels');

box on;
legend(h([3 2 1]), {'Nash Equilibrium Reduction', 'Net-zero and Induced Reduction', 'Residual Emissions'},...
     'Location', 'northwest');  % Reversed order

% Define the y‐positions and their labels
yl     = [0.17, 0.24, 0.30];
labels = {'Net-Zero 2050 Commitment','Net-Zero 2060 Commitment','Net-Zero 2070 Commitment'};

for k = 1:numel(yl)
    % dashed black line
    yline(yl(k), '--k', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    % black text label at x=0
    text(0, yl(k), labels{k}, ...
         'VerticalAlignment',   'bottom', ...
         'HorizontalAlignment', 'left', ...
         'Color',               'k', ...
         'FontSize',            10);
end


% table documentation
% 1) Sort the names
countryNames_sorted = countryNames(idx);

% 2) Build the table
T_2C_sorted = table( ...
    countryNames_sorted, ...      % 1st column: sorted country names
    level1', ...                   % 2nd column: sorted level1 values
    emission_sorted, ...          % 3rd column: sorted emissions
    'VariableNames', {'Country','CommittedPct','Emissions_Gt'} ...
);

% sum the top 27 rows (EU countries)
eu_total_emissions = sum(emission_sorted(1:27));

% print it
fprintf('Total emissions of 27 EU countries: %.3f Gt/year\n', eu_total_emissions);


%% Print information about Network Influence and Reduction

% Step 1: GDP-weighted sensitivity vector
GDP_sens = country_GDP .* country_network_sens;  % 209x1
% Step 2: Compute total influence each country exerts
% Each country's influence = sum over all countries they influence: 
%   influence_to_country * GDP_sens_of_that_country

% Transpose InfluenceMatrix to access columns easily
InfluenceMatrix_T = network';  % Now rows are the influencers

% Multiply: each row (influencer) dot GDP_sens vector
influence_scores = InfluenceMatrix_T * GDP_sens;  % 209x1 vector

normalized_scores = influence_scores / sum(influence_scores);

% Total influence for each group (sum of normalized influence scores)
USA_influence   = sum(normalized_scores(zeolot_USA == 1));
China_influence = sum(normalized_scores(zeolot_China == 1));
EU_influence    = sum(normalized_scores(zeolot_EU == 1));

fprintf('Influence Scores:\n');
fprintf('USA:   %.4f\n', USA_influence);
fprintf('China: %.4f\n', China_influence);
fprintf('EU:    %.4f\n', EU_influence);

% Step 1: Define exclusion mask for USA, China, EU
exclude_mask = (zeolot_USA | zeolot_China | zeolot_EU);  % 209x1 logical

% Step 2: Get influence scores of non-major countries
non_major_scores = normalized_scores;
non_major_scores(exclude_mask) = -Inf;  % Exclude by setting to -Inf

% Step 3: Find top 3 influential non-major countries
[sorted_scores, idx] = maxk(non_major_scores, 10);  % Get top 3 indices and scores

% Step 4: Print their names and scores
fprintf('Top 10 influential countries (excluding USA, China, EU):\n');
for i = 1:10
    fprintf('%s: %.4f\n', countryNames{idx(i)}, sorted_scores(i));
end



% compute cost for each country at 3 degree C scenario

[s_eq_3C, globalCost, globalEmission] = runToConvergence(s_initial, params, maxIter);
[Cost_country_3C, Cost_mitigation_3C, Cost_economic_3C, Cost_network_3C] = computeCost(s_eq_3C, params);

s_initial = randi([1 99], length(country_CO2), 1);
% 2) For positions flagged as zealots, set the strategy to 83 for net-zero 2050.
s_initial(zeolot_2050 == 1) = 83;
% set 76 for net-zero 2060.
s_initial(zeolot_2060 == 1) = 76;
% set 70 for net-zero 2070.
s_initial(zeolot_2070 == 1) = 70;

% 3) Run the best-response convergence algorithm with zealot tracking.
%    This function returns:
%    final_Cost_Country (vector), costFianl (scalar),
%    emission_Final (scalar), and additional outputs.
[final_Cost_Country_2C, costFianl, emission_Final, s_equilibrium, costHistory, emissionHistory] = ...
    runToConvergenceZealots(s_initial, params, maxIter, currentZealot);

[Cost_country_2C, Cost_mitigation_2C, Cost_economic_2C, Cost_network_2C] = computeCost(s_equilibrium, params);


%Cost_country_2C = final_Cost_Country_2C;

% Step 1: Compute absolute cost reduction per country
Cost_reduction = Cost_country_3C - Cost_country_2C;

% Step 2: Define masks
mask_EU    = logical(zeolot_EU);
mask_China = logical(zeolot_China);
mask_USA   = logical(zeolot_USA);

mask_other2050 = logical(zeolot_other2050);
mask_other2060 = logical(zeolot_other2060);
mask_other2070 = logical(zeolot_other2070);

mask_netzero_others = mask_other2050 | mask_other2060 | mask_other2070;

% Step 3: Define rest of world mask (excluding all above)
mask_combined = mask_EU | mask_China | mask_USA | mask_netzero_others;
mask_rest = ~mask_combined;

% Step 4: Aggregate cost reductions and 3C baseline costs
total_3C_EU    = sum(Cost_country_3C(mask_EU));
total_red_EU   = sum(Cost_reduction(mask_EU));

total_3C_China = sum(Cost_country_3C(mask_China));
total_red_China = sum(Cost_reduction(mask_China));

total_3C_USA   = sum(Cost_country_3C(mask_USA));
total_red_USA  = sum(Cost_reduction(mask_USA));

total_3C_netzero = sum(Cost_country_3C(mask_netzero_others));
total_red_netzero = sum(Cost_reduction(mask_netzero_others));

total_3C_rest = sum(Cost_country_3C(mask_rest));
total_red_rest = sum(Cost_reduction(mask_rest));

% Step 5: Calculate percentage reductions
pct_red_EU      = 100 * total_red_EU / total_3C_EU;
pct_red_China   = 100 * total_red_China / total_3C_China;
pct_red_USA     = 100 * total_red_USA / total_3C_USA;
pct_red_netzero = 100 * total_red_netzero / total_3C_netzero;
pct_red_rest    = 100 * total_red_rest / total_3C_rest;

% Step 6: Print results
fprintf('Cost Reduction from 3C to 2C:\n');
fprintf('EU:            %.2f%% reduction\n', pct_red_EU);
fprintf('China:         %.2f%% reduction\n', pct_red_China);
fprintf('USA:           %.2f%% reduction\n', pct_red_USA);
fprintf('Other net-zero countries: %.2f%% reduction\n', pct_red_netzero);
fprintf('Rest of world: %.2f%% reduction\n', pct_red_rest);



%% stacked bar chart

groups = {'EU', 'USA', 'China', 'Other Net-Zero', 'Rest of World'};

% Define a helper to sum costs by group
sumByGroup = @(cost, mask) [
    sum(cost(mask_EU));
    sum(cost(mask_USA));
    sum(cost(mask_China));
    sum(cost(mask_netzero_others));
    sum(cost(mask_rest));
];

% 2°C scenario
mitigation_2C = sumByGroup(Cost_mitigation_2C, '2C');
economic_2C   = sumByGroup(Cost_economic_2C, '2C');
network_2C    = sumByGroup(Cost_network_2C, '2C');

% 3°C scenario
mitigation_3C = sumByGroup(Cost_mitigation_3C, '3C');
economic_3C   = sumByGroup(Cost_economic_3C, '3C');
network_3C    = sumByGroup(Cost_network_3C, '3C');

% Combine into 5x6 matrix: columns = [mit3C, econ3C, net3C, mit2C, econ2C, net2C]
cost_matrix = [mitigation_3C, economic_3C, network_3C, mitigation_2C, economic_2C, network_2C];

figure;
hold on;
bar_width = 0.35;
x = 1:length(groups);

% Custom colors (soft contrast)
bar_colors = [
    [114, 147, 203];   % Blue
    [225, 151, 76];    % Orange
    [132, 186, 91]     % Green
] / 255;

% 3°C bars
h3C = bar(x - bar_width/2, cost_matrix(:,1:3)/1e12, bar_width, 'stacked');
for i = 1:3
    h3C(i).FaceColor = bar_colors(i, :);
end

% 2°C bars
h2C = bar(x + bar_width/2, cost_matrix(:,4:6)/1e12, bar_width, 'stacked');
for i = 1:3
    h2C(i).FaceColor = bar_colors(i, :);
end

% Axis and labels
xticks(x);
xticklabels(groups);
ylabel('Climate Cost (trillion $/year)');
ylim([0 6.5]);
legend({'Mitigation Cost', 'Adaptation Cost', 'Misalignment Cost'}, 'Location', 'northeast');
%title('Cost Breakdown by Economic Group: 3°C vs 2°C');

%% function

function h = wfall(ax, y)
    %WFALL Customized waterfall plot for a given axis and change values.
    %   h = wfall(ax, y) creates a waterfall chart in the axis specified by
    %   ax using the vector y. If only one argument is provided, ax is assumed
    %   to be the vector y and the current axis (gca) is used.
    %
    %   The initial bar is drawn in red, the final bar in green, and 
    %   subsequent reduction bars are colored blue for increases and red for
    %   decreases. Cumulative totals are calculated using cumsum(y).
    %
    %   This version annotates each bar with its cumulative value at the top,
    %   and for the reduction bars, the individual change (with sign) is
    %   displayed within the bar.
    
    if nargin == 1
        y = ax;
        ax = gca;
    end
    if ~strcmp(ax.NextPlot, 'add')
        fprintf('hold on not set for current axes. Overriding.\n');
        hold(ax, 'on');
    end

    y = y(:); % ensure column vector
    n = length(y);
    cumy = cumsum(y);
    
    % Compute the range manually
    range_y = max(cumy) - min(cumy);
    
    % Adjust axes limits to provide space for annotations:
    set(ax, 'XLim', [0, n+1] + 0.5, ...
            'YLim', [min(min(cumy) - 0.05*range_y, 0), max(max(cumy) + 0.05*range_y, 0)]);

    % Define custom colors:
    % Initial bar: red, final bar: green,
    % For decreases: MATLAB red-ish, for increases: MATLAB blue-ish.
    col_initial  = [0.8500, 0.3250, 0.0980]; % red for initial bar
    col_last     = [0, 0.5, 0];               % green for final bar
    col_increase = [0.8500, 0.3250, 0.0980];   % red-ish for negative changes
    col_decrease = [0, 0.4470, 0.7410];         % blue-ish for positive changes
    % For intermediate total bars if any, we still use black.
    col_total    = [0, 0, 0];                   % black, not used for first/last

    % Duplicate the current axis for each bar:
    for i = 1:n
        ax(i+1) = copyobj(ax(1), ax(1).Parent);
    end
    % Make all additional axes invisible
    set(ax(2:end), 'Color', 'none', 'XColor', 'none', 'YColor', 'none');
    linkprop(ax, {'XLim', 'YLim', 'Position', 'DataAspectRatio'});
    
    h = [];
    % First bar: the starting total is drawn on the main axis using col_initial.
    h(1) = bar(ax(1), 1, cumy(1), 'FaceColor', col_initial, 'BaseValue', 0);
    % Annotate the first bar with its cumulative value.
    text(1, cumy(1), sprintf('%.2f', cumy(1)),...
         'HorizontalAlignment','center','VerticalAlignment','bottom','FontWeight','bold');
    
    % For bars 2 to n: draw each step bar with cumulative differences.
    % These correspond to the reduction bars.
    for i = 1:n-1
        base = cumy(i);
        change_val = y(i+1);
        % Choose color based on the sign of the change:
        if change_val > 0
            color_current = col_increase;
        else
            color_current = col_decrease;
        end
        % Draw bar on a separate axis copy.
        h(end+1) = bar(ax(i+1), i+1, cumy(i+1), 'FaceColor', color_current, ...
                        'BaseValue', base, 'ShowBaseLine', 'off');
         
        % Annotate the individual change within the bar.
        tcolor = 'w';
        if change_val > 0
            tcolor = 'k';
        end
        midY = base + (cumy(i+1) - base)/2;
        text(ax(i+1), i+1, midY, sprintf('(%+.2f)', change_val),...
            'HorizontalAlignment','center','VerticalAlignment','middle',...
            'Color', tcolor, 'FontWeight','bold');
    end
    
    % Last total bar: drawn on the main axis copy using col_last.
    h(end+1) = bar(ax(1), n+1, cumy(n), 'FaceColor', col_last, 'BaseValue', 0);
    text(n+1, cumy(n), sprintf('%.2f', cumy(n)),...
         'HorizontalAlignment','center','VerticalAlignment','bottom','FontWeight','bold');
    
    % Ensure all bars use flat face coloring.
    %set(h, 'FaceColor', 'flat')
end
