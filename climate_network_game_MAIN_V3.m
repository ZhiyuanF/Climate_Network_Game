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

% Plot results
figure(1)
plot(s_sample, C_mitigation)
xlabel('Mitigation Strategy')
ylabel('Cumulative Mitigation Cost')
title('Mitigation Cost vs. Strategy (Discrete Approximation)')



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

composite_score = emission_total./(countriesData{:,2}.*countriesData{:,4}); % emission/(GDP*sensitivity)

% fitting the cumulative CO2 emission with temperature rise function as
% quadratic function fitting

% Point 1: (0, 1.5) Net-zero cumulative emission, 1.5 degree C T-increase
% Point 2: (13.3/emission_total, 2.0) use up carbon budget, 2.0 degree C T-increase
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

% Plot the direct emission -> loss mapping
figure(4);
plot(x3Fit, gdpLoss, '-');
xlabel('Global Emission Change');
ylabel('Economic Loss of GDP');
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




%% Test run functions

% construct initial strategy
s_initial = 40 * ones(length(country_CO2),1);

[Cost_total, Cost_mitigation, Cost_economic, Cost_network] = computeCost(s_initial, params);

s_next = bestResponseUpdate(s_initial, params);

%
% convergence check

% construct initial strategy
s_initial = randi([1 99], length(country_CO2), 1);

maxIter = 365;
[s_eq, globalCost, globalEmission] = runToConvergence(s_initial, params, maxIter);

fprintf('Global Cost at equilibrium: %.4e\n', globalCost);
fprintf('Global Emission at equilibrium: %.4f\n', globalEmission);


%% Trajectory likelihood: random (updated with full cost components)

nRuns   = 200;
maxIter = 365;

costHists        = cell(nRuns,1);
mitigationHists  = cell(nRuns,1);
economicHists    = cell(nRuns,1);
networkHists     = cell(nRuns,1);
emissHists       = cell(nRuns,1);

tic;  % Start the timer

for r = 1 : nRuns
    % --- 1) Generate a random initial strategy
    s_initial = randi([1 99], length(country_CO2), 1);

    % --- 2) Run best-response convergence tracking with all cost components
    [~, costHist, mitHist, econHist, netHist, emissHist] = ...
        runAndTrackConvergence_allCosts(s_initial, params, maxIter);

    % --- 3) Store results
    costHists{r}       = costHist;  
    mitigationHists{r} = mitHist;
    economicHists{r}   = econHist;
    networkHists{r}    = netHist;
    emissHists{r}      = emissHist;

    % --- 4) Progress print
    if mod(r, 10) == 0
        fprintf('Completed run %d of %d\n', r, nRuns);
    end
end

elapsedTime = toc;  % End timer
fprintf('All %d runs finished in %.2f seconds.\n', nRuns, elapsedTime);

%% Plot Global Cost Trajectories
figure(9); hold on;
for r = 1 : nRuns
    thisCost = costHists{r};
    p = plot(1:length(thisCost), thisCost/1e12, 'LineWidth', 1.0);
    p.Color = [0, 0, 1, 0.2];  % blue, transparent
end
hold off;
xlabel('Iteration');
ylabel('Global Total Climate Cost (trillion $/year)');
ylim([0 3]*10);
%title('Global Cost for 200 Random Starts Iteration');

%% Eveloe plot: cost
maxIter = max(cellfun(@length, costHists));
costMat = nan(nRuns, maxIter);

for r = 1 : nRuns
    thisCost = costHists{r};
    costMat(r, 1:length(thisCost)) = thisCost / 1e12;  % convert to trillion
end

% Compute envelopes
qLow   = prctile(costMat, 0, 1);   % 0th percentile (outer lower)
qHigh  = prctile(costMat, 100, 1);  % 100th percentile (outer upper)
qMed   = prctile(costMat, 50, 1);    % median
q25    = prctile(costMat, 25, 1);    % 25th percentile (inner lower)
q75    = prctile(costMat, 75, 1);    % 75th percentile (inner upper)
% Smooth all percentile curves
qLow_smooth  = smoothdata(qLow,  'gaussian', 5);
qHigh_smooth = smoothdata(qHigh, 'gaussian', 5);
q25_smooth   = smoothdata(q25,   'gaussian', 5);
q75_smooth   = smoothdata(q75,   'gaussian', 5);
qMed_smooth  = smoothdata(qMed,  'gaussian', 5);

% Plot trajectories and envelopes
figure(); clf; hold on;


% Replace qLow, qHigh, etc. with their smoothed versions
fill([1:maxIter, fliplr(1:maxIter)], ...
     [qLow_smooth, fliplr(qHigh_smooth)], ...
     [0.6 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);

fill([1:maxIter, fliplr(1:maxIter)], ...
     [q25_smooth, fliplr(q75_smooth)], ...
     [0.2 0.4 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.4);

plot(1:maxIter, qMed_smooth, 'k-', 'LineWidth', 2);

xlabel('Iteration');
ylabel('Global Total Climate Cost (trillion $/year)');
ylim([0 3]*10);
legend({'Outer Envelope (100%)', 'Dense Region (50%)', 'Median'}, 'Location', 'northeast');
%title('Global Cost Trajectories with Envelopes');

hold off;

%% Plot Global Emission Trajectories
figure(10); hold on;
for r = 1 : nRuns
    thisEmiss = emissHists{r};
    p = plot(1:length(thisEmiss), thisEmiss, 'LineWidth', 1.0);
    p.Color = [0, 0, 1, 0.2];
end
hold off;
xlabel('Iteration');

ylim([0 1]);
%title('Global Emission for 200 Random Starts Iteration');
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

%% Eveloe plot: emission
% Aggregate all emission histories into a matrix
maxIter = max(cellfun(@length, emissHists));
emissMat = nan(nRuns, maxIter);

for r = 1 : nRuns
    thisEmiss = emissHists{r};
    emissMat(r, 1:length(thisEmiss)) = thisEmiss;
end

% Compute percentile envelopes
qLow   = prctile(emissMat, 0, 1);
qHigh  = prctile(emissMat, 100, 1);
qMed   = prctile(emissMat, 50, 1);
q25    = prctile(emissMat, 25, 1);
q75    = prctile(emissMat, 75, 1);

% Smooth all percentile curves
qLow_smooth  = smoothdata(qLow,  'gaussian', 5);
qHigh_smooth = smoothdata(qHigh, 'gaussian', 5);
q25_smooth   = smoothdata(q25,   'gaussian', 5);
q75_smooth   = smoothdata(q75,   'gaussian', 5);
qMed_smooth  = smoothdata(qMed,  'gaussian', 5);

% Plot trajectories and envelopes
figure(); clf; hold on;


% Replace qLow, qHigh, etc. with their smoothed versions
fill([1:maxIter, fliplr(1:maxIter)], ...
     [qLow_smooth, fliplr(qHigh_smooth)], ...
     [0.6 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);

fill([1:maxIter, fliplr(1:maxIter)], ...
     [q25_smooth, fliplr(q75_smooth)], ...
     [0.2 0.4 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.4);

plot(1:maxIter, qMed_smooth, 'k-', 'LineWidth', 2);

xlabel('Iteration');
ylabel('Avg. Global Emissions by 2100 (Gt/year)');
ylim([0 1]);

% Custom Y-axis ticks for emission values in Gt/year
tickVals = 0:10:50;  % Gt/year
yt = tickVals / emission_total;
set(gca, ...
    'YTick', yt, ...
    'YTickLabel', arrayfun(@(x)sprintf('%d',x), tickVals, 'UniformOutput',false) ...
);

legend({'Outer Envelope (100%)', 'Dense Region (50%)', 'Median'}, 'Location', 'northeast');
%title('Global Emission Trajectories with Envelopes');

hold off;

%% stacked bar
% Initialize tracking variables
minCost = inf;
maxCost = -inf;
minInfo = struct('run', 0, 'iter', 0);
maxInfo = struct('run', 0, 'iter', 0);

% Store corresponding cost components for plotting later
minComponents = [];
maxComponents = [];

for r = 1:nRuns
    totalCostVec = costHists{r};
    
    [thisMin, minIter] = min(totalCostVec);
    [thisMax, maxIter] = max(totalCostVec);

    if thisMin < minCost
        minCost = thisMin;
        minInfo.run = r;
        minInfo.iter = minIter;
        minComponents = [mitigationHists{r}(minIter), economicHists{r}(minIter), networkHists{r}(minIter)];
    end

    if thisMax > maxCost
        maxCost = thisMax;
        maxInfo.run = r;
        maxInfo.iter = maxIter;
        maxComponents = [mitigationHists{r}(maxIter), economicHists{r}(maxIter), networkHists{r}(maxIter)];
    end
end

fprintf('Lowest Cost: %.4e (Run %d, Iter %d)\n', minCost, minInfo.run, minInfo.iter);
fprintf('Highest Cost: %.4e (Run %d, Iter %d)\n', maxCost, maxInfo.run, maxInfo.iter);

% Use the first run, last iteration as equilibrium
equilibriumIter = length(costHists{1});
equilibriumComponents = [
    mitigationHists{1}(equilibriumIter), ...
    economicHists{1}(equilibriumIter), ...
    networkHists{1}(equilibriumIter)
];

% Cost component labels
componentLabels = {'Mitigation', 'Economic Loss', 'Network Externality'};

% Combine both cases for plotting side-by-side
barData = [minComponents; equilibriumComponents; maxComponents]/1e12;

% Open figure(11) and match size explicitly without affecting figure(9)
figure(11);
fig11 = gcf;                  % Store handle to figure(11)
fig9 = figure(9);             % Just to get size info
figPos = get(fig9, 'Position');  % Get position from figure(9)
set(fig11, 'Position', figPos);  % Apply size to figure(11)

% Now plot on figure(11)
bar(barData, 'stacked');
set(gca, 'XTickLabel', {'Lowest Cost', 'Nash Equilibrium', 'Highest Cost'});
ylabel('Cost (trillion $/year)');
ylim([0 3]*10);  % Match y-axis

legend({'Mitigation Cost', 'Adaptation Cost', 'Misalignment Cost'}, ...
       'Location', 'northwest');

%title('Cost Component Breakdown: Lowest, Equilibrium, and Highest Cost');
