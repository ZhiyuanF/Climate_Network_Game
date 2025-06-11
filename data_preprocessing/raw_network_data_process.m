%% Climate network games raw data processing: Network Structure

% raw data from: https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.STA:ITG(4.0.0)

% downloaded data file:
% dataset_2025-04-06T20_54_26.960301847Z_DEFAULT_INTEGRATION_IMF.STA_IMTS_1.0.0.csv

clc;
clearvars;

%% data reading and check

% read CSV file into a table
raw_dataTable = readtable('dataset_2025-04-06T20_54_26.960301847Z_DEFAULT_INTEGRATION_IMF.STA_IMTS_1.0.0.csv');

% Extract the 4th column. This returns either a string array or cell array
countryColumn = raw_dataTable{2:end,4}; % 2:end skip the header

% Use the unique function to get unique country names
uniqueCountries = unique(countryColumn);

% move "World" to the first entry in the list.
% uniqueCountries is a cell array, use strcmp:
idxWorld = strcmp(uniqueCountries, 'World');

if any(idxWorld)
    % Reorder so "World" is at the top
    uniqueCountries = [uniqueCountries(idxWorld); uniqueCountries(~idxWorld)];
end


%% Raw data Processing and Network matrix construction

% define the dimension of network matrix to be N*N, where N is number of
% unique countries.
N = length(uniqueCountries);

% Create an N x N matrix (initialized to zeros):
Network_matrix = zeros(N, N);

% Create an N x N matrix (initialized to zeros):
duplicate_count_matrix = zeros(N, N);


% Number of rows in your original dataset
numRows = size(raw_dataTable, 1);

for r = 2:numRows  % starting from 2nd row to skip the header (if not already handled)
    % 1) Extract the 'from' country (column 4)
    fromCountry = raw_dataTable{r, 4};
    
    % 2) Extract the 'to' country (column 6)
    toCountry   = raw_dataTable{r, 6};
    
    % 3) Extract the numeric value in the last column
    valueToAdd = raw_dataTable{r, end};
    
    % 4) If valueToAdd is NaN, skip this row
    if isnan(valueToAdd)
        continue;  % go to the next iteration of the loop without updating anything
    end
    
    % 5) Find the indices (i, j) in uniqueCountries for 'from' and 'to'
    i = find(strcmp(uniqueCountries, fromCountry), 1);
    j = find(strcmp(uniqueCountries, toCountry),   1);
    
    % (Optional) If i or j is empty, it means we didn't find that country in uniqueCountries
    % you can handle that situation. For simplicity, we'll assume all are valid.
    
    % 6) Update your matrix entry
    Network_matrix(i, j) = Network_matrix(i, j) + valueToAdd;
    
    % 7) Increment the count to keep track of how many times we’ve updated (i,j)
    duplicate_count_matrix(i, j) = duplicate_count_matrix(i, j) + 1;
end

% After finishing the loop that populates Network_matrix and duplicate_count_matrix
% Throw an error if any entry in "duplicate_count_matrix" is bigger than 2 ---
if any(duplicate_count_matrix(:) > 2)
    error('Some entries in duplicate_count_matrix exceed 2!');
end


% Data explanation: the original dataset does include some potential
% dulicate entries but assigning different values. For example, coutnry A
% export to B, country B import from A, both could exist and have distinct
% values, which is the nature of the dataset. Without worrying about it, we
% simple take the sum of all raw entries.

% we freeze this version of network matrix as raw_matrix
raw_Network_matrix = Network_matrix;

%% Applicable network matrix reconstruction

% sum of all entries will make the matrix symmetric
% to incoporate all entries, make the network matrix plus it transpose

full_Network_matrix = Network_matrix + Network_matrix'; % unit: million USD

% Drop the first row and first column, first row/column is "World" which should be removed.
reducedMatrix = full_Network_matrix(2:end, 2:end);

% Compute row sums of the reduced matrix
rowSums_values = sum(reducedMatrix, 2);  % this gives a column vector, unit Million $USD

% Divide each entry in a row by that row's sum
% the possibility of zero row sums to avoid NaNs or Inf.
% replace zero sums with 1 for safe division (assuming that row is effectively all zeros):
zeroMask = (rowSums_values == 0);
rowSums_values(zeroMask) = 1;  % so we don't divide by zero

% Create the weighted matrix
influence_Matrix = reducedMatrix ./ rowSums_values;
% !!! the influence_Matrix in no longer symmetric
% for each (i,j) in influence_Matrix, it means country i recives 
% influence_Matrix(i,j) value of influence from country j, and all
% influence that country i recives sum to 1. 

% at this point, lots of the influence value in the matrix is very small.
% we want to make those values to 0, re-weight the matrix to make the
% network and calculation simpler.

% 1) Calculate and display sparsity BEFORE thresholding
numZerosBefore = nnz(influence_Matrix == 0);     % Count how many elements are exactly zero
totalEntries   = numel(influence_Matrix);        % Total number of elements in the matrix
sparsityBefore = numZerosBefore / totalEntries;

fprintf('Sparsity BEFORE thresholding: %.2f%%\n', 100 * sparsityBefore);

% Threshold values below 0.001 to zero:
threshold = 0.001;
influence_Matrix(influence_Matrix < threshold) = 0;

% Recalculate each row's sum:
rowSums = sum(influence_Matrix, 2);
% Create the weighted matrix again
influence_Matrix = influence_Matrix ./ rowSums;


% 2) Calculate and display sparsity AFTER thresholding
numZerosAfter = nnz(influence_Matrix == 0);
sparsityAfter = numZerosAfter / totalEntries;

fprintf('Sparsity AFTER thresholding: %.2f%%\n', 100 * sparsityAfter);
fprintf('Total influence lost: %.2f%%\n', 100-(100 * sum(rowSums)/length(rowSums)));

% adjust the uniqueCountries as well
uniqueCountries = uniqueCountries(2:end);

% store the result
save('influence_Matrix.mat', 'influence_Matrix');
save('influence_Matrix_TradeValue.mat', 'reducedMatrix');

%% GDP, influence sensitivity, and other input data construction

% read the GDP dataset containing raw data
raw_GDP_Table = readtable('dataset_2025-04-07T19_31_28.539266684Z_DEFAULT_INTEGRATION_IMF.RES_WEO_6.0.0.csv');

% Get all unique indicators from column 5
uniqueIndicators = unique(raw_GDP_Table{2:end,5});

% create a structure to hold each sub-table
splitTables = struct();
indicator_mapping = {'GDP_per_capita';'GDP_billion';'Net_debt';'Net_balance'};

% Loop over each unique indicator
for i = 1:length(uniqueIndicators)
    
    % Extract the indicator we’re interested in
    thisIndicator = uniqueIndicators(i);
    
    % If the column is a cell array of strings or a string array,
    % you may need to use strcmp instead of == (depending on data type).
    % For numeric indicators: use raw_GDP_Table{:,5} == thisIndicator.
    % For string indicators: use strcmp.
    %
    % For example, if it’s a string/cellstr column, do:
    rowSelector = strcmp(raw_GDP_Table{:,5}, thisIndicator);
    
    % Step 3: Extract all rows that match this indicator
    thisSubTable = raw_GDP_Table(rowSelector,:);
    
    % Step 4 (optional): store in a struct for easy access
    indicatorName = indicator_mapping{i};  % convert string to char if needed
    indicatorName = matlab.lang.makeValidName(indicatorName);  % make it a valid field name
    splitTables.(indicatorName) = thisSubTable;
end

% column 2
% fill the GDP data
% Further processing of GDP_billion table
GDP_billion = splitTables.GDP_billion;

% initialize an expanded cell array, which is 209x2
expandedCountries = [uniqueCountries, cell(size(uniqueCountries))];
% loop through each country in uniqueCountries
for i = 1:size(uniqueCountries,1)
    thisCountry = uniqueCountries{i};                  % Extract the country name from the cell
    rowMask     = strcmp(GDP_billion{:,4}, thisCountry);  % Find matching rows in GDO_billion
    
    if any(rowMask)
        % If there's at least one match, take the value in column 10
        % If there are multiple matches, this picks them all in a cell array
        % (you might decide to take the first match only: GDO_billion{find(rowMask,1),10})
        expandedCountries{i,2} = GDP_billion{rowMask,10};
    else
        % If there's no match, you can store NaN, 0, or keep it empty
        expandedCountries{i,2} = NaN;
    end
end

% column 3
% fill the trading data
% expandedCountries is assumed 209×2; this line makes it 209×3
%converted_values = rowSums_values / 1000;
expandedCountries(:,3) = num2cell(rowSums_values / 1000);

% column 4
% fill the calculated influence value: baseline
% fill NaN with lower 5% quantile since they generally are less sensitive
% to tradings

% Convert columns 2 and 3 from cell -> numeric arrays
col2 = cell2mat(expandedCountries(:,2));   % e.g. "GDO_billion" values
col3 = cell2mat(expandedCountries(:,3));   % "rowsum_values/1000" values

% Compute the ratio: col3 / col2
% Wherever col2 is NaN, the ratio will become NaN automatically.
ratioValues = col3 ./ col2;  % e.g. if col2 is NaN, ratio is NaN

% freeze this matrix for raw data checking
expandedCountries(:,4) = num2cell(ratioValues);
expandedCountries_freeze = expandedCountries;

% compute the 25% quantile (Q1) of these ratios, excluding NaNs
validMask = ~isnan(ratioValues);
ratioQ1   = quantile(ratioValues(validMask), 0.05);

% Fill NaNs in the ratio array with this Q1 value
ratioValues(isnan(ratioValues)) = ratioQ1;

% Put the final ratio array back into expandedCountries, as a new column (column #4)
expandedCountries(:,4) = num2cell(ratioValues);

% back-filling the NaN in column 2
col4 = cell2mat(expandedCountries(:,4));  % ratio (col3 / col2), with some fill
% Find rows where Column #2 is NaN
naMask = isnan(col2);
% Backfill those NaNs with col3 / col4
col2(naMask) = col3(naMask) ./ col4(naMask);

% Assign the updated numeric array back into expandedCountries as a cell array
expandedCountries(:,2) = num2cell(col2);

%% GHG emission data processing and construction

% read the GHG emission dataset containing the emission and emission per
% capaita data
raw_GHG_Table = readtable('List_of_countries_by_greenhouse_gas_emissions_2.csv');


% country name cleaning and matching
% Extract the raw country names:
gdpCountriesRaw = raw_GDP_Table{:,4};  % cell array of strings
ghgCountriesRaw = raw_GHG_Table{:,2};  % cell array of strings

% utilizing the text matchin function
GHG_countries_matching = matchCountriesByLongestSubstring(ghgCountriesRaw, uniqueCountries);

% examing the results manually is necessary!
GHG_countries_matching_Table = cell2table(GHG_countries_matching, ...
    'VariableNames', {'Original','BestMatch','Confidence'});
writetable(GHG_countries_matching_Table, 'bestMatches.csv', 'QuoteStrings', true);

writecell(uniqueCountries, 'preferred_country_index.csv');

%% Using the manually reviewed file, construct matching

finalMatches = readcell('bestMatches_reviewed.csv'); 


% Expand expandedCountries to hold two more columns (will become 209×6)
numRows = size(expandedCountries, 1);
expandedCountries(:,5:6) = {[]};  % initialize columns 5 & 6 as empty


for i = 1:numRows

    cName = expandedCountries{i,1};  % "listA" country name from column #2

    rowIdx = find(strcmp(finalMatches(:,2), cName), 1)-1; % index correction consider the header row

    if ~isempty(rowIdx)

        % Extract column #1 and #3 from raw_GHG_Table for that rowIdx
        valCol1 = raw_GHG_Table{rowIdx,1};  % e.g. some numeric or text data
        valCol3 = raw_GHG_Table{rowIdx,3};  % e.g. GHG figures, etc.

        % Store those in columns #5 and #6 of finalMatches
        expandedCountries{i,5} = valCol1;
        expandedCountries{i,6} = valCol3;
    else
        % If no match, leave them empty or insert a placeholder, e.g. 'NO MATCH'
        expandedCountries{i,5} = [];
        expandedCountries{i,6} = [];
    end

end

%% Filling the missing entries
% total emission 1000, per capita emission 2
% Identify rows where column #5 is empty
empty5 = cellfun(@isempty, expandedCountries(:,5));
% Fill these rows with {1000}
expandedCountries(empty5,5) = {1000};

% Identify rows where column #6 is empty
empty6 = cellfun(@isempty, expandedCountries(:,6));
% Fill these rows with {2}
expandedCountries(empty6,6) = {2};

% store the final output: expandedCountries, as .csv
% Define the header row (as a 1×6 cell)
headers = {"country", "GDP_billion", "trade_billion", ...
           "influence_sensitivity", "GHG_ktCO2", "tCO2_per_capita"};

% Concatenate the header row on top of your expandedCountries (an n×6 cell)
expandedCountries_with_header = [headers; expandedCountries];

% Write to CSV
writecell(expandedCountries_with_header, 'Countries_input_data.csv');

%% Text matching functions
function bestMatches = matchCountriesByLongestSubstring(listA, listB)
    % matchCountriesByLongestSubstring finds for each string in listA 
    % the best match in listB using the length of the longest common substring.
    %
    % Inputs:
    %   listA: cell array of country names (e.g. {"China","Germany",...})
    %   listB: cell array of country names (e.g. {"People's Republic of China","Germany (DE)",...})
    %
    % Output:
    %   bestMatches: an N×3 cell array where N = length(listA).
    %     - Column 1: the original entry from listA
    %     - Column 2: the best match in listB
    %     - Column 3: notes about confidence / manual verification
    
    % Ensure inputs are cell arrays of strings
    if ~iscell(listA) || ~iscell(listB)
        error('Both inputs must be cell arrays of strings.');
    end
    
    nA = numel(listA);
    bestMatches = cell(nA,3);
    
    for i = 1:nA
        strA = listA{i};
        
        bestLCSLength = 0;
        bestMatchInB = '';
        
        % Compare strA with every entry in listB to find the best LCS
        for j = 1:numel(listB)
            strB = listB{j};
            currentLCSLength = longestCommonSubstring(strA, strB);
            
            if currentLCSLength > bestLCSLength
                bestLCSLength = currentLCSLength;
                bestMatchInB = strB;
            end
        end
        
        % Store the results
        bestMatches{i,1} = strA;        % Original string
        bestMatches{i,2} = bestMatchInB;% Best match in listB
        
        % If the best LCS length is 2 or less, mark as uncertain
        if bestLCSLength <= 2
            bestMatches{i,3} = 'LCS <= 2 → needs manual check';
        else
            % Also check coverage (e.g., ratio of LCS length to length of strA)
            coverageRatio = bestLCSLength / length(strA);
            if coverageRatio == 1.0
                bestMatches{i,3} = 'Perfect match (100% coverage)';
            else
                bestMatches{i,3} = sprintf('Coverage: %.1f%%', coverageRatio*100);
            end
        end
    end
end

function L = longestCommonSubstring(str1, str2)
    % longestCommonSubstring returns the length of the longest common substring
    % between two input strings str1 and str2.
    
    if isempty(str1) || isempty(str2)
        L = 0;
        return
    end
    
    len1 = length(str1);
    len2 = length(str2);
    dp = zeros(len1, len2);
    
    L = 0;  % Keep track of the maximum length found
    
    for i = 1:len1
        for j = 1:len2
            if str1(i) == str2(j)
                if i == 1 || j == 1
                    dp(i, j) = 1;
                else
                    dp(i, j) = dp(i-1, j-1) + 1;
                end
                
                if dp(i, j) > L
                    L = dp(i, j);
                end
            else
                dp(i, j) = 0;
            end
        end
    end
end




