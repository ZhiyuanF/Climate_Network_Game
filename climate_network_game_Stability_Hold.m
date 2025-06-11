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
plot(x3Fit, gdpLoss*100, '-','LineWidth', 2);
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


%% Test on effect of network sensitivity

% Define the range of scalers to test
scalers = 0:0.1:2;  % Scalar values from 0 to 2 with step size 0.1

% For the scalar outputs, use matrices.
costFinal_network_sense    = zeros(1, numel(scalers));
emissionFinal_network_sense = zeros(1, numel(scalers));

for s = 1:numel(scalers)

    params_sensitivity = params;
    % Update the country_network_sens with the current scalar value
    params_sensitivity.country_network_sens = country_network_sens * scalers(s);

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
    [~, costFianl, emission_Final, ~, ~, ~] = ...
        runToConvergenceZealots(s_initial, params_sensitivity, maxIter, currentZealot);

    % 4) Store the outputs for each scalar case
    costFinal_network_sense(1, s)       = costFianl;
    emissionFinal_network_sense(1, s)   = emission_Final;

end

%%
figure;
% Plot costFinal on the left y-axis and emissionFinal on the right y-axis
[ax, h1, h2] = plotyy(scalers, emissionFinal_network_sense*emission_total, scalers, costFinal_network_sense*1e-12);

% Set the labels and titles
xlabel('Sensitivity of global network connectivity');
ylabel(ax(1), 'Avg. Emission by 2100: Gt/year');  % Left y-axis label for cost
ylabel(ax(2), 'Global Cost (trillion $/year)');  % Right y-axis label for emissions
%title('Emission Final and Cost Final vs Scaler');

% Set the line styles for better distinction
set(h1, 'LineStyle', '-', 'Color', 'b', 'LineWidth', 2);  % Cost Final
set(h2, 'LineStyle', '-', 'Color', [1 0 0 0.5], 'LineWidth', 2);  % Emission Final

% Set the axis limits for left and right y-axes
axis(ax(1), [0 2 10 20]);  % Left y-axis: x from 0 to 2, y from 10 to 20
axis(ax(2), [0 2 12 16]);  % Right y-axis: x from 0 to 2, y from 12 to 16

% Set the number of ticks for the left and right y-axes
yticks(ax(1), linspace(10, 20, 6));  % 6 ticks for left y-axis (from 10 to 20)
yticks(ax(2), linspace(12, 16, 5));  % 5 ticks for right y-axis (from 12 to 16)

% Set the x-axis to display percentages (0% to 200%)
xticks(ax(1), linspace(0, 2, 5));  % 5 ticks for x-axis (0% to 200%)
xticklabels(ax(1), strcat(string(linspace(0, 200, 5)), '%'));  % Add '%' symbol to x-ticks labels

%% check if zeolot have incentive to defect

countryNames = countriesData{:,1};

% 0) Sensitivity‐boost hyperparameter
sensBoost = 1;    % 100% boost when testing boosted case

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


%% Introducing new zeolot and check their cost incentives

% 0) incremental commitment
s_increment = 0;

% 1) Baseline run to get equilibrium strategies
[finalCost_orig, costFinal_orig, emissionFinal_orig, s_equilibrium, costHistory, emissionHistory] = ...
    runToConvergenceZealots(s_initial+s_increment, params, maxIter, currentZealot);
% Identify which countries “committed” by looking at equilibrium s > 65
newZealotMask = (s_equilibrium > 65);
fprintf('Total fixed-zealots (s_eq > 65): %d countries\n', sum(newZealotMask));

% --- determine which countries to test (the “new zealots”) ---
liftIdx = find(newZealotMask & ~currentZealot);

% pre‐allocate
nLift = numel(liftIdx);
origS     = zeros(nLift,1);
liftedS   = zeros(nLift,1);
pctChange = zeros(nLift,1);

% loop over each candidate
for k = 1:nLift
    i = liftIdx(k);
    
    % original equilibrium strategy
    s0 = s_equilibrium(i);
    origS(k) = s0;
    
    % decide lift level
    if s0 > 76 && s0 <= 83
        s1 = 83;
    elseif s0 > 70 && s0 <= 76
        s1 = 76;
    elseif s0 > 65 && s0 <= 70
        s1 = 70;
    else
        % outside your lift‐range: skip
        s1 = s0;
    end
    liftedS(k) = s1;
    
    % prepare initial state and mask (only this country is “fixed”)
    s_init = s_equilibrium;
    s_init(i) = s1;
    mask    = currentZealot;
    mask(i) = 1;
    
    % rerun convergence
    [finalCost_lift, ~, ~, ~, ~, ~] = ...
        runToConvergenceZealots(s_init, params, maxIter, mask);
    
    % compute percentage cost change
    cost_before = finalCost_orig(i);
    cost_after  = finalCost_lift(i);
    pctChange(k) = (cost_after - cost_before) / cost_before * 100;
end

% build results table
Country        = countryNames(liftIdx);
Equilibrium    = origS;
LiftedStrategy = liftedS;
CostDeltaPct   = pctChange;

results = table(Country, Equilibrium, LiftedStrategy, CostDeltaPct);

% display
disp(results);
%{
% Create a new boxplot figure
figure;
boxchart(pctChange);
%xlabel({'Δ Cost (%)'});
ylabel('Percentage Change in Cost');
title('Distribution of Cost Change for Lifted Strategies');
grid on;

% compute the quartiles
q1 = prctile(pctChange, [10, 95]);

% print them
fprintf('10th percentile (lower stability bound): %.2f%%\n', q1(1));
fprintf('95th percentile (upper stability bound): %.2f%%\n', q1(2));

%}
%% check the incentive change for the original zeolots

origZealotIdx = find(currentZealot);
newZealotIdx  = find(newZealotMask & ~currentZealot);
nOrig = numel(origZealotIdx);

CostChangeNoNew        = zeros(nOrig,1);
CostChangeWithNewUnlift= zeros(nOrig,1);
CostChangeWithNewLift  = zeros(nOrig,1);

% Masks for the “with‐new” cases
maskNewBase = false(size(s_equilibrium));
maskNewBase(newZealotIdx) = true;        % new zealots
maskNewBase(origZealotIdx)  = true;      % plus all originals

% Pre‐build the two starting s‐vectors
sWithNewUnlift = s_equilibrium;          % new zealots stay at eq.
sWithNewLift   = s_equilibrium;
sWithNewLift(newZealotIdx) = liftedS;    % new zealots lifted

for k = 1:nOrig
    i = origZealotIdx(k);
    
    % ---- Case 1: no new zealots ----
    mask1 = currentZealot;
    mask1(i) = false;        % let i defect
    sInit1 = s_equilibrium;
    [cost1, ~, ~, ~, ~, ~] = runToConvergenceZealots(sInit1, params, maxIter, mask1);
    CostChangeNoNew(k) = (cost1(i) - finalCost_orig(i)) / finalCost_orig(i) * 100;
    
    % ---- Case 2: new zealots fixed at eq. ----
    mask2 = maskNewBase;
    mask2(i) = false;        % let i defect
    [cost2, ~, ~, ~, ~, ~] = runToConvergenceZealots(sWithNewUnlift, params, maxIter, mask2);
    CostChangeWithNewUnlift(k) = (cost2(i) - finalCost_orig(i)) / finalCost_orig(i) * 100;
    
    % ---- Case 3: new zealots fixed at lifted ----
    mask3 = maskNewBase;
    mask3(i) = false;        % let i defect
    [cost3, ~, ~, ~, ~, ~] = runToConvergenceZealots(sWithNewLift, params, maxIter, mask3);
    CostChangeWithNewLift(k) = (cost3(i) - finalCost_orig(i)) / finalCost_orig(i) * 100;
end

% Build and display the comparison table
Country                   = countryNames(origZealotIdx);
CostChange_NoNew          = CostChangeNoNew;
CostChange_NewUnlifted    = CostChangeWithNewUnlift;
CostChange_NewLifted      = CostChangeWithNewLift;

comparison = table( ...
    Country, ...
    CostChange_NoNew, ...
    CostChange_NewUnlifted, ...
    CostChange_NewLifted ...
);

disp(comparison);



%% Box plot

% your thresholds
threshold_defect    = prctile(CostChange_NewUnlifted, 25);
threshold_no_pledge = prctile(pctChange, 75);

figure; hold on;

% draw the shaded band
xlims = [0.5 2.5];
yl = [threshold_defect, threshold_no_pledge];
hPatch = patch(...
    [xlims(1) xlims(2) xlims(2) xlims(1)], ...
    [yl(1)    yl(1)    yl(2)    yl(2)   ], ...
    [0.8 0.8 0.8], ...              % light gray
    'FaceAlpha', 0.3, ...           % 30% opacity
    'EdgeColor','none' ...
);
uistack(hPatch,'bottom');          % push behind the boxcharts

% add a centered text label on the shaded area
xText = mean(xlims)+0.25;
yText = mean(yl)-3;
text( ...
  xText, yText, ...
  'Strategy stability range (covers 75% countries)', ...
  'HorizontalAlignment','center', ...
  'VerticalAlignment','middle', ...
  'FontSize',10, ...
  'FontWeight','bold', ...
  'BackgroundColor','none' ...
);

% now plot your two boxcharts
x1 = ones(size(CostChange_NewUnlifted));
boxchart(x1, CostChange_NewUnlifted, 'BoxWidth',0.5);

% remove outlier for the second group
[~, idxMax]       = max(pctChange);
pctFiltered       = pctChange;
pctFiltered(idxMax) = [];
x2 = 2*ones(size(pctFiltered));
boxchart(x2, pctFiltered, 'BoxWidth',0.5);

% finalize axes
xlim(xlims);
set(gca, ...
    'XTick',      [1 2], ...
    'XTickLabel', {'Existing pledges defection', 'New pledges commitment'});
ylabel('Cost Change in Percentage');
%title('Cost Change for Each Country Unilatarally Altering Existing Strategy');
grid on;
ytickformat('%g%%');

%% Stair plot

% ==== Group 1: Old Zealots ====
[oldCostSorted, oldSortIdx] = sort(CostChange_NewUnlifted);
old_zealot.idx  = origZealotIdx(oldSortIdx);
old_zealot.name = countryNames(old_zealot.idx);
old_zealot.CO2  = country_CO2(old_zealot.idx);
old_zealot.cost = oldCostSorted;

% ==== Group 2: New Zealots ====
[newCostSorted, newSortIdx] = sort(pctChange);
new_zealot.idx  = liftIdx(newSortIdx);
new_zealot.name = countryNames(new_zealot.idx);
new_zealot.CO2  = country_CO2(new_zealot.idx);
new_zealot.cost = newCostSorted;
% ==== Combine Old and New Zealots ====
all_cost = [old_zealot.cost; new_zealot.cost];
all_CO2  = [old_zealot.CO2;  new_zealot.CO2];
all_name = [old_zealot.name; new_zealot.name];
group_tag = [repmat("Old", length(old_zealot.cost), 1); ...
             repmat("New", length(new_zealot.cost), 1)];

% ==== Drop the last entry (outlier) ====
all_cost  = all_cost(1:end-1);
all_CO2   = all_CO2(1:end-1)*1e-9;
all_name  = all_name(1:end-1);
group_tag = group_tag(1:end-1);

% ==== Calculate Bar Positions ====
xStart = [0; cumsum(all_CO2(1:end-1))];  % Left edge of each bar

% ==== Plot ====
figure(10); hold on;

% ==== Compute horizontal span ====
xEnd = xStart(end) + all_CO2(end) + 1;

% ==== Plot Stability Range as Light Gray Band ====
fill([0, xEnd, xEnd, 0], ...
     [threshold_defect, threshold_defect, threshold_no_pledge, threshold_no_pledge], ...
     [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.8);  % light gray shaded band

% add a centered text label on the shaded area
xText = mean(xlims)+27;
yText = mean(yl)-3;
text( ...
  xText, yText, ...
  'Strategy stability range (covers 75% countries)', ...
  'HorizontalAlignment','center', ...
  'VerticalAlignment','middle', ...
  'FontSize',10, ...
  'FontWeight','bold', ...
  'BackgroundColor','none' ...
);

top_n_labels = 5;  % You can change this value as needed
[~, top_emitters_idx] = maxk(all_CO2, top_n_labels);

for i = 1:length(all_cost)
    % Define rectangle corners
    x = [xStart(i), xStart(i) + all_CO2(i), ...
         xStart(i) + all_CO2(i), xStart(i)];
    y = [0, 0, all_cost(i), all_cost(i)];
    
    % Set color based on group
    if group_tag(i) == "Old"
        barColor = [0 0.4470 0.7410];  % Blue
    else
        barColor = [0.8500 0.3250 0.0980];  % Red
    end
    
    % Draw bar
    patch(x, y, barColor, 'EdgeColor', 'k', 'FaceAlpha', 0.8);
    
    %{
    if ismember(i, top_emitters_idx)
    text(mean(x(1:2)), all_cost(i), all_name{i}, ...
         'Rotation', 90, 'FontSize', 8, ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom','Rotation', 45);
    end
    %}
end

% ==== Compute where the old group ends ====
nOld = length(old_zealot.cost);
x_split = xStart(nOld) + all_CO2(nOld);

% ==== Draw vertical line ====
yL = ylim();  % get current y-limits
line([x_split, x_split], yL, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.2);

% ==== Add vertically rotated text labels ====
text(x_split-2, yL(2)-13, 'Existing Net-Zero Pledges', ...
     'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
     'FontSize', 8, 'Rotation', 90);

text(x_split+0.5, yL(2)-23, 'Potential New Net-Zero Pledges', ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
     'FontSize', 8, 'Rotation', 90);



xlabel('Accumulated CO₂ Emissions by Countries (Gt/year)');
ylabel('Cost Change in Percentage');
%title('Old vs. New Zealots: Cost Change vs CO₂ Emissions');

%legend({'Old Zealot', 'New Zealot'}, 'Location', 'northwest');

hold off;

xEnd = xStart(end) + all_CO2(end);
xlim([0, 45]);
% Get current y-tick values
yt = yticks();
% Convert to string with percentage symbol
yticklabels(arrayfun(@(v) sprintf('%.0f%%', v), yt, 'UniformOutput', false));

fig = figure(10);
defaultPos = fig.Position;  % [left bottom width height]
fig.Position = [defaultPos(1), defaultPos(2), ...
                1.5 * defaultPos(3), defaultPos(4)];


%% check for Dominoes effect


% Pre‐allocate
droppedOrig = [];
droppedNew  = [];

% initial settings
maskWithNew = currentZealot | newZealotMask;
sWithNew    = s_equilibrium;
sWithNew(newZealotIdx) = liftedS;  % lift each new zealot

% 1) Test existing zealots for defection
for i = origZealotIdx(:)'
    % Build mask fixing all other original zealots
    mask1 = maskWithNew;
    mask1(i) = false;  % free i to defect

    % Rerun equilibrium
    [cost1, ~, ~, ~, ~, ~] = ...
        runToConvergenceZealots(s_equilibrium, params, maxIter, mask1);

    % compute % saving
    saving = (cost1(i) - finalCost_orig(i)) / finalCost_orig(i) * 100;

    if saving < threshold_defect
        droppedOrig(end+1) = i;  %#ok<AGROW>
    end
end

% 2) Test new zealots for commitment
%    (i.e. remove them from the “would‐commit” mask if penalty too small)
for i = newZealotIdx(:)'
    % Build mask fixing everyone except this new zealot
    mask2 = maskWithNew;
    mask2(i) = false;  % free i to stay uncommitted

    % Rerun equilibrium
    [cost2, ~, ~, ~, ~, ~] = ...
        runToConvergenceZealots(sWithNew, params, maxIter, mask2);

    % compute % penalty avoided by not committing
    % i.e. compare cost_at_lift vs. cost_when_unfixed
    cost_lift = cost2(i);
    penalty   = (cost_lift - finalCost_orig(i)) / finalCost_orig(i) * 100;

    if penalty > threshold_no_pledge
        droppedNew(end+1) = i;  %#ok<AGROW>
    end
end

% Print results
if isempty(droppedOrig)
    fprintf('No existing zealots dropped this round.\n');
else
    fprintf('Dropped existing zealots:\n');
    for i = droppedOrig
        fprintf('  – %s\n', countryNames{i});
    end
end

if isempty(droppedNew)
    fprintf('No new zealots dropped this round.\n');
else
    fprintf('Dropped potential zealots (won’t commit):\n');
    for i = droppedNew
        fprintf('  – %s\n', countryNames{i});
    end
end

%% Dominoes effect check

% Initialize
round = 0;
droppedLog = struct('round', {}, 'droppedOrigIdx', {}, 'droppedOrigNames', {}, ...
                            'droppedNewIdx', {}, 'droppedNewNames', {}, ...
                            'globalCost', {}, 'globalEmission', {});

% initial settings
zealotMask = currentZealot | newZealotMask;
sWithNew    = s_equilibrium;
sWithNew(newZealotIdx) = liftedS;  % lift each new zealot

while true
    round = round + 1;
    
    % Figure out which are “original” vs “new” in the current mask
    origIdx = find(zealotMask & currentZealot);
    newIdx  = find(zealotMask & newZealotMask & ~currentZealot);
    
    %{
    % print currently included original zealots
    fprintf('Currently included EXISTING zealots (%d):\n', numel(origIdx));
    for j = 1:numel(origIdx)
        fprintf('  – %s\n', countryNames{origIdx(j)});
    end
    % print currently included new zealots
    fprintf('Currently included NEW zealots (%d):\n', numel(newIdx));
    for j = 1:numel(newIdx)
        fprintf('  – %s\n', countryNames{newIdx(j)});
    end
    %}
    droppedOrig = [];
    droppedNew  = [];
    
    % --- 1) test existing zealots for defection saving ---
    for i = origIdx(:)'
        mask1 = zealotMask;
        mask1(i) = false;  % free i to defect
        
        % rerun equilibrium
        [cost1, ~, ~, ~, ~, ~] = ...
            runToConvergenceZealots(s_equilibrium, params, maxIter, mask1);
        
        saving = (cost1(i) - finalCost_orig(i)) / finalCost_orig(i) * 100;
        if saving < threshold_defect
            droppedOrig(end+1) = i; %#ok<AGROW>
        end
    end
    
    % --- 2) test new zealots for commitment penalty ---
    for i = newIdx(:)'
        mask2 = zealotMask;
        mask2(i) = false;  % free i to stay uncommitted
        
        % rerun equilibrium
        [cost2, ~, ~, ~, ~, ~] = ...
            runToConvergenceZealots(sWithNew, params, maxIter, mask2);
        
        cost_lift = cost2(i);
        cost_free = finalCost_orig(i);
        penalty   = (cost_lift - cost_free) / cost_free * 100;
        
        if penalty > threshold_no_pledge
            droppedNew(end+1) = i; %#ok<AGROW>
        end
    end
    
    % If nothing dropped, we’re done
    if isempty(droppedOrig) && isempty(droppedNew)
        fprintf('No more drops in round %d; cascade complete.\n', round);
        break
    end
    
    % Record this round’s drops
    droppedLog(round).round           = round;
    droppedLog(round).droppedOrigIdx  = droppedOrig;
    droppedLog(round).droppedOrigNames= countryNames(droppedOrig);
    droppedLog(round).droppedNewIdx   = droppedNew;
    droppedLog(round).droppedNewNames = countryNames(droppedNew);

    
    
    % Display summary
    fprintf('Round %d dropped:\n', round);
    if ~isempty(droppedOrig)
        fprintf('  Existing zealots:\n');
        fprintf('    %s\n', droppedLog(round).droppedOrigNames{:});
    end
    if ~isempty(droppedNew)
        fprintf('  New zealots:\n');
        fprintf('    %s\n', droppedLog(round).droppedNewNames{:});
    end
    
    % Remove them from the mask for next round
    zealotMask([droppedOrig droppedNew]) = false;

    [~, Dominoes_costFinal, Dominoes_emissionFinal, ~, ~, ~] = ...
            runToConvergenceZealots(sWithNew, params, maxIter, zealotMask);
    % Store the resutls after dropped nations at this round 
    droppedLog(round).globalCost = Dominoes_costFinal;
    droppedLog(round).globalEmission = Dominoes_emissionFinal;
    
    % If mask is empty, stop early
    if ~any(zealotMask)
        fprintf('All zealots dropped by end of round %d.\n', round);
        break
    end
end

% --- Finally, print remaining zealots ---
remainingIdx   = find(zealotMask);
remainingNames = countryNames(remainingIdx);

if isempty(remainingIdx)
    fprintf('No countries remain committed.\n');
else
    fprintf('\nCountries remaining committed after round %d (%d total):\n', ...
            round, numel(remainingIdx));
    fprintf('  %s\n', remainingNames{:});
end

%% Dominoes effect check: impact of USA, China, EU

% Define the scenarios to test (removing EU, USA, and China separately)
scenarios = {'EU', 'USA', 'China'};  % List of countries to remove in each scenario

for scenario_idx = 1:numel(scenarios)

    % Create an empty struct to store results for the current scenario
    scenarioResults = struct('round', {}, 'droppedOrigIdx', {}, 'droppedOrigNames', {}, ...
                             'droppedNewIdx', {}, 'droppedNewNames', {}, ...
                             'globalCost', {}, 'globalEmission', {});


    % Initialize zealotMask for the current scenario
    zealotMask = currentZealot | newZealotMask;
    
    % Exclude the country for the current scenario
    switch scenarios{scenario_idx}
        case 'EU'
            zealotMask = zealotMask - zeolot_EU;
        case 'USA'
            zealotMask = zealotMask - zeolot_USA;
        case 'China'
            zealotMask = zealotMask - zeolot_China;
    end

    % Reset round to 0 for each scenario
    round = 0;
    
    sWithNew    = s_equilibrium;
    sWithNew(newZealotIdx) = liftedS;  % lift each new zealot

    while true
        round = round + 1;

        % Figure out which are “original” vs “new” in the current mask
        origIdx = find(zealotMask & currentZealot);
        newIdx  = find(zealotMask & newZealotMask & ~currentZealot);

        droppedOrig = [];
        droppedNew  = [];

        % --- 1) test existing zealots for defection saving ---
        for i = origIdx(:)'
            mask1 = zealotMask;
            mask1(i) = false;  % free i to defect

            % rerun equilibrium
            [cost1, ~, ~, ~, ~, ~] = ...
                runToConvergenceZealots(s_equilibrium, params, maxIter, mask1);

            saving = (cost1(i) - finalCost_orig(i)) / finalCost_orig(i) * 100;
            if saving < threshold_defect
                droppedOrig(end+1) = i; %#ok<AGROW>
            end
        end

        % --- 2) test new zealots for commitment penalty ---
        for i = newIdx(:)'
            mask2 = zealotMask;
            mask2(i) = false;  % free i to stay uncommitted

            % rerun equilibrium
            [cost2, ~, ~, ~, ~, ~] = ...
                runToConvergenceZealots(sWithNew, params, maxIter, mask2);

            cost_lift = cost2(i);
            cost_free = finalCost_orig(i);
            penalty   = (cost_lift - cost_free) / cost_free * 100;

            if penalty > threshold_no_pledge
                droppedNew(end+1) = i; %#ok<AGROW>
            end
        end

        % If nothing dropped, we’re done
        if isempty(droppedOrig) && isempty(droppedNew)
            switch scenarios{scenario_idx}
                case 'EU'
                    results_EU = scenarioResults;
                case 'USA'
                    results_USA = scenarioResults;
                case 'China'
                    results_China = scenarioResults;
            end

            fprintf('No more drops in round %d; cascade complete.\n', round);
            break
        end

        % Record this round’s drops for the current scenario
        scenarioResults(round).round = round;
        scenarioResults(round).droppedOrigIdx  = droppedOrig;
        scenarioResults(round).droppedOrigNames = countryNames(droppedOrig);
        scenarioResults(round).droppedNewIdx   = droppedNew;
        scenarioResults(round).droppedNewNames = countryNames(droppedNew);

        % Display summary
        fprintf('Round %d dropped in scenario %s:\n', round, scenarios{scenario_idx});
        if ~isempty(droppedOrig)
            fprintf('  Existing zealots:\n');
            fprintf('    %s\n', scenarioResults(round).droppedOrigNames{:});
        end
        if ~isempty(droppedNew)
            fprintf('  New zealots:\n');
            fprintf('    %s\n', scenarioResults(round).droppedNewNames{:});
        end

        % Remove them from the mask for next round
        zealotMask([droppedOrig droppedNew]) = false;

        [~, Dominoes_costFinal, Dominoes_emissionFinal, ~, ~, ~] = ...
            runToConvergenceZealots(sWithNew, params, maxIter, zealotMask);

        % Store the results after dropped nations at this round
        scenarioResults(round).globalCost = Dominoes_costFinal;
        scenarioResults(round).globalEmission = Dominoes_emissionFinal;

        % If mask is empty, stop early
        if ~any(zealotMask)

            switch scenarios{scenario_idx}
                case 'EU'
                    results_EU = scenarioResults;
                case 'USA'
                    results_USA = scenarioResults;
                case 'China'
                    results_China = scenarioResults;
            end

            fprintf('All zealots dropped by end of round %d.\n', round);
            break
        end
    end
end


%% Dominoes effect: weakened network ties

% Define a weakening factor for the connections between USA, China, and the EU
weaken_ratio = 0.5;  % Change this value between 0 and 1

% Create a weakened version of the network matrix G (209x209)
G_weak = network;  % Start with the original matrix
% define the connections to be weakened
zeolot_weaken = zeolot_USA + zeolot_EU + zeolot_China;
% find indices
weaken_indices = find(zeolot_weaken);

% Weaken the connections between the identified countries
for i = 1:numel(weaken_indices)
    for j = i+1:numel(weaken_indices)
        % Get the current pair of indices
        idx1 = weaken_indices(i);  % First index
        idx2 = weaken_indices(j);  % Second index

        % Weaken the connection between these two indices
        G_weak(idx1, idx2) = G_weak(idx1, idx2) * weaken_ratio;  % Impact of idx1 on idx2
        G_weak(idx2, idx1) = G_weak(idx2, idx1) * weaken_ratio;  % Impact of idx2 on idx1 (symmetric relationship)
    end
end

% Re-normalize the matrix to ensure each row sums to 1
for i = 1:size(G_weak, 1)
    row_sum = sum(G_weak(i, :));  % Calculate the sum of the row
    G_weak(i, :) = G_weak(i, :) / row_sum;  % Normalize the row
end

% construct new params
params_network_weaken = params;
params_network_weaken.network = G_weak;


% Define the scenarios to test (removing EU, USA, and China separately)
scenarios = {'EU', 'USA', 'China'};  % List of countries to remove in each scenario

for scenario_idx = 1:numel(scenarios)

    % Create an empty struct to store results for the current scenario
    scenarioResults = struct('round', {}, 'droppedOrigIdx', {}, 'droppedOrigNames', {}, ...
                             'droppedNewIdx', {}, 'droppedNewNames', {}, ...
                             'globalCost', {}, 'globalEmission', {});


    % Initialize zealotMask for the current scenario
    zealotMask = currentZealot | newZealotMask;
    
    % Exclude the country for the current scenario
    switch scenarios{scenario_idx}
        case 'EU'
            zealotMask = zealotMask - zeolot_EU;
        case 'USA'
            zealotMask = zealotMask - zeolot_USA;
        case 'China'
            zealotMask = zealotMask - zeolot_China;
    end

    % Reset round to 0 for each scenario
    round = 0;
    
    sWithNew    = s_equilibrium;
    sWithNew(newZealotIdx) = liftedS;  % lift each new zealot

    while true
        round = round + 1;

        % Figure out which are “original” vs “new” in the current mask
        origIdx = find(zealotMask & currentZealot);
        newIdx  = find(zealotMask & newZealotMask & ~currentZealot);

        droppedOrig = [];
        droppedNew  = [];

        % --- 1) test existing zealots for defection saving ---
        for i = origIdx(:)'
            mask1 = zealotMask;
            mask1(i) = false;  % free i to defect

            % rerun equilibrium
            [cost1, ~, ~, ~, ~, ~] = ...
                runToConvergenceZealots(s_equilibrium, params_network_weaken, maxIter, mask1);

            saving = (cost1(i) - finalCost_orig(i)) / finalCost_orig(i) * 100;
            if saving < threshold_defect
                droppedOrig(end+1) = i; %#ok<AGROW>
            end
        end

        % --- 2) test new zealots for commitment penalty ---
        for i = newIdx(:)'
            mask2 = zealotMask;
            mask2(i) = false;  % free i to stay uncommitted

            % rerun equilibrium
            [cost2, ~, ~, ~, ~, ~] = ...
                runToConvergenceZealots(sWithNew, params_network_weaken, maxIter, mask2);

            cost_lift = cost2(i);
            cost_free = finalCost_orig(i);
            penalty   = (cost_lift - cost_free) / cost_free * 100;

            if penalty > threshold_no_pledge
                droppedNew(end+1) = i; %#ok<AGROW>
            end
        end

        % Instead of repeating the same result assignment after each break, store it once
        if isempty(droppedOrig) && isempty(droppedNew)
            switch scenarios{scenario_idx}
                case 'EU'
                    results_EU_weaken = scenarioResults;
                case 'USA'
                    results_USA_weaken = scenarioResults;
                case 'China'
                    results_China_weaken = scenarioResults;
            end

            fprintf('No more drops in round %d; cascade complete.\n', round);
            break
        end

        % Record this round’s drops for the current scenario
        scenarioResults(round).round = round;
        scenarioResults(round).droppedOrigIdx  = droppedOrig;
        scenarioResults(round).droppedOrigNames = countryNames(droppedOrig);
        scenarioResults(round).droppedNewIdx   = droppedNew;
        scenarioResults(round).droppedNewNames = countryNames(droppedNew);

        % Display summary
        fprintf('Round %d dropped in scenario %s:\n', round, scenarios{scenario_idx});
        if ~isempty(droppedOrig)
            fprintf('  Existing zealots:\n');
            fprintf('    %s\n', scenarioResults(round).droppedOrigNames{:});
        end
        if ~isempty(droppedNew)
            fprintf('  New zealots:\n');
            fprintf('    %s\n', scenarioResults(round).droppedNewNames{:});
        end

        % Remove them from the mask for next round
        zealotMask([droppedOrig droppedNew]) = false;

        [~, Dominoes_costFinal, Dominoes_emissionFinal, ~, ~, ~] = ...
            runToConvergenceZealots(sWithNew, params_network_weaken, maxIter, zealotMask);

        % Store the results after dropped nations at this round
        scenarioResults(round).globalCost = Dominoes_costFinal;
        scenarioResults(round).globalEmission = Dominoes_emissionFinal;

        % If mask is empty, stop early
        if ~any(zealotMask)

            switch scenarios{scenario_idx}
                case 'EU'
                    results_EU_weaken = scenarioResults;
                case 'USA'
                    results_USA_weaken = scenarioResults;
                case 'China'
                    results_China_weaken = scenarioResults;
            end

            fprintf('All zealots dropped by end of round %d.\n', round);
            break
        end
    end
end

%%

% Define a weakening factor range for the connections between USA, China, and the EU
weaken_ratios = 0:0.25:1;  % From 0 to 1 with 0.1 spacing

% Define the scenarios to test (removing EU, USA, and China separately)
scenarios = {'EU', 'USA', 'China'};  % List of countries to remove in each scenario

% Initialize a matrix to store the results (globalCost and globalEmission for each scenario and weaken_ratio)
results_globalCost = zeros(numel(weaken_ratios), numel(scenarios));  % For storing global cost
results_globalEmission = zeros(numel(weaken_ratios), numel(scenarios));  % For storing global emission



% Start the timer
tic;
for r = 1:numel(weaken_ratios)
    weaken_ratio = weaken_ratios(r);  % Current weaken ratio

    % Create a weakened version of the network matrix G (209x209)
    G_weak = network;  % Start with the original matrix

    % Define the connections to be weakened
    zeolot_weaken = zeolot_USA + zeolot_EU + zeolot_China;

    % Find the indices of the zealot countries
    weaken_indices = find(zeolot_weaken);

    % Weaken the connections between the identified countries
    for i = 1:numel(weaken_indices)
        for j = i+1:numel(weaken_indices)
            % Get the current pair of indices
            idx1 = weaken_indices(i);  % First index
            idx2 = weaken_indices(j);  % Second index

            % Weaken the connection between these two indices
            G_weak(idx1, idx2) = G_weak(idx1, idx2) * weaken_ratio;  % Impact of idx1 on idx2
            G_weak(idx2, idx1) = G_weak(idx2, idx1) * weaken_ratio;  % Impact of idx2 on idx1 (symmetric relationship)
        end
    end

    % Re-normalize the matrix to ensure each row sums to 1
    for i = 1:size(G_weak, 1)
        row_sum = sum(G_weak(i, :));  % Calculate the sum of the row
        G_weak(i, :) = G_weak(i, :) / row_sum;  % Normalize the row
    end

    % Construct new params with the weakened network
    params_network_weaken = params;
    params_network_weaken.network = G_weak;
    params_network_weaken.country_network_sens = country_network_sens - ...
                                                 country_network_sens.*zeolot_weaken*(1-weaken_ratio);

    % Iterate through each scenario
    for scenario_idx = 1:numel(scenarios)

        % Create an empty struct to store results for the current scenario
        scenarioResults = struct('round', {}, 'droppedOrigIdx', {}, 'droppedOrigNames', {}, ...
                                 'droppedNewIdx', {}, 'droppedNewNames', {}, ...
                                 'globalCost', {}, 'globalEmission', {});

        % Initialize zealotMask for the current scenario
        zealotMask = currentZealot | newZealotMask;

        % Exclude the country for the current scenario
        switch scenarios{scenario_idx}
            case 'EU'
                zealotMask = zealotMask - zeolot_EU;
            case 'USA'
                zealotMask = zealotMask - zeolot_USA;
            case 'China'
                zealotMask = zealotMask - zeolot_China;
        end

        % Reset round to 0 for each scenario
        round = 0;

        sWithNew = s_equilibrium;
        sWithNew(newZealotIdx) = liftedS;  % lift each new zealot

        while true
            round = round + 1;

            % Figure out which are “original” vs “new” in the current mask
            origIdx = find(zealotMask & currentZealot);
            newIdx = find(zealotMask & newZealotMask & ~currentZealot);

            droppedOrig = [];
            droppedNew = [];

            % --- 1) test existing zealots for defection saving ---
            for i = origIdx(:)'
                mask1 = zealotMask;
                mask1(i) = false;  % free i to defect

                % rerun equilibrium
                [cost1, ~, ~, ~, ~, ~] = ...
                    runToConvergenceZealots(s_equilibrium, params_network_weaken, maxIter, mask1);

                saving = (cost1(i) - finalCost_orig(i)) / finalCost_orig(i) * 100;
                if saving < threshold_defect
                    droppedOrig(end+1) = i; %#ok<AGROW>
                end
            end

            % --- 2) test new zealots for commitment penalty ---
            for i = newIdx(:)'
                mask2 = zealotMask;
                mask2(i) = false;  % free i to stay uncommitted

                % rerun equilibrium
                [cost2, ~, ~, ~, ~, ~] = ...
                    runToConvergenceZealots(sWithNew, params_network_weaken, maxIter, mask2);

                cost_lift = cost2(i);
                cost_free = finalCost_orig(i);
                penalty = (cost_lift - cost_free) / cost_free * 100;

                if penalty > threshold_no_pledge
                    droppedNew(end+1) = i; %#ok<AGROW>
                end
            end

            % If nothing dropped, we’re done
            if isempty(droppedOrig) && isempty(droppedNew)
                % Store the results for the current scenario and weaken_ratio (only the last round's global cost and emission)
                results_globalCost(r, scenario_idx) = scenarioResults(round-1).globalCost;
                results_globalEmission(r, scenario_idx) = scenarioResults(round-1).globalEmission;
                break
            end

            % Record this round’s drops for the current scenario
            scenarioResults(round).round = round;
            scenarioResults(round).droppedOrigIdx = droppedOrig;
            scenarioResults(round).droppedOrigNames = countryNames(droppedOrig);
            scenarioResults(round).droppedNewIdx = droppedNew;
            scenarioResults(round).droppedNewNames = countryNames(droppedNew);
            
            %{
            % Display summary
            fprintf('Round %d dropped in scenario %s with weaken_ratio %.1f:\n', round, scenarios{scenario_idx}, weaken_ratio);
            if ~isempty(droppedOrig)
                fprintf('  Existing zealots:\n');
                fprintf('    %s\n', scenarioResults(round).droppedOrigNames{:});
            end
            if ~isempty(droppedNew)
                fprintf('  New zealots:\n');
                fprintf('    %s\n', scenarioResults(round).droppedNewNames{:});
            end
            %}

            % Remove them from the mask for next round
            zealotMask([droppedOrig droppedNew]) = false;

            [~, Dominoes_costFinal, Dominoes_emissionFinal, ~, ~, ~] = ...
                runToConvergenceZealots(sWithNew, params_network_weaken, maxIter, zealotMask);

            % Store the results after dropped nations at this round
            scenarioResults(round).globalCost = Dominoes_costFinal;
            scenarioResults(round).globalEmission = Dominoes_emissionFinal;

            % If mask is empty, stop early
            if ~any(zealotMask)
                % Store the results for the current scenario and weaken_ratio (only the last round's global cost and emission)
                results_globalCost(r, scenario_idx) = scenarioResults(round).globalCost;
                results_globalEmission(r, scenario_idx) = scenarioResults(round).globalEmission;
                break
            end
        end

        
    end

    % Print progress after finishing each weaken_ratio and elapsed time for that ratio
    elapsed_time = toc;  % Measure time after finishing each weaken_ratio
    fprintf('Finished weaken_ratio %.1f. Time elapsed: %.2f seconds.\n', weaken_ratio, elapsed_time);
    
    % Restart the timer for the next weaken_ratio
    tic;
end

% Display the stored results for global cost and emission
disp('Global Cost Results:');
disp(results_globalCost);
disp('Global Emission Results:');
disp(results_globalEmission);

%% Create a figure
figure;

% Plot the results for each scenario (EU, USA, China) on the same graph
plot(weaken_ratios, results_globalEmission(:, 1)*emission_total, '-o', 'LineWidth', 2, 'MarkerSize', 6);  % EU scenario
hold on;
plot(weaken_ratios, results_globalEmission(:, 2)*emission_total, '-s', 'LineWidth', 2, 'MarkerSize', 6);  % USA scenario
plot(weaken_ratios, results_globalEmission(:, 3)*emission_total, '-^', 'LineWidth', 2, 'MarkerSize', 6);  % China scenario

% Add labels and title
xlabel('Degree of EU-USA-China Network Connectivity');
ylabel('Avg. Global Emission by 2100: Gt/year');
%title('Global Emission vs. Weaken Ratio for Different Scenarios');

% Add a legend
legend({'EU Defect', 'USA Defect', 'China Defect'}, 'Location', 'Best');

% Set the x-axis to display percentages (0% to 200%)
% Set x-axis ticks to display percentages (0% to 100%)
xticks(weaken_ratios);  % Set the x-ticks to be from 0 to 1 with 0.1 spacing
xticklabels(cellfun(@(x) sprintf('%.0f%%', x*100), num2cell(weaken_ratios), 'UniformOutput', false));  % Convert to percentage format
% Add grid for better readability
grid on;

% Display the plot
hold off;

%% High stability of current network connectivity

% Define the stability_bar data (assuming the values are already assigned)
stability_bar = [emission_Final, droppedLog(end).globalEmission, results_globalEmission(end,:)]*emission_total;

% 2 degreee C line
e_2deg = 0.2597*emission_total;
% defect degree C scenario
defect_emission = 17;
e_defect = polyval(p1, defect_emission/emission_total);

% Create the bar chart
figure(9);

% Create the bar chart
fig = figure(9);
defaultPos = fig.Position;
fig.Position = [defaultPos(1), defaultPos(2), 0.6 * defaultPos(3), defaultPos(4)];

bar(stability_bar);
yline(e_2deg, '--b', '2°C Threshold', 'LabelHorizontalAlignment', 'left', 'LineWidth', 2);
yline(defect_emission, '--r', '2.15°C Threshold', 'LabelHorizontalAlignment', 'left', 'LineWidth', 2);

% Set the x-axis labels
set(gca, 'XTickLabel', {'No defect', 'Incentivized defect', 'EU defect', 'USA defect', 'China defect'});

% Add title and labels
%title('Global Emission for Different Defection Scenarios');
ylabel('Avg. Global Emission by 2100: Gt/year');



