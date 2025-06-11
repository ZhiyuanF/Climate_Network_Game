%% Plotting the global trading network using global map
% 
% Clear workspace and command window
clear all;   % remove all variables
clc;         % clear command window

%% data reading and check

% read CSV file into a table
dataTable_country = readtable('Countries_input_data.csv');
dataTable_coordinates = readtable('Countries_input_data_mapping_tool.csv');

trade_network = load('influence_Matrix_TradeValue.mat').reducedMatrix;

%%
% Extract data by columns, names and coordinates
country_ID = dataTable_country{:,1}; % 
country_name = dataTable_coordinates{:,2}; % 
country_long = dataTable_coordinates{:,5}; % 
country_lat = dataTable_coordinates{:,4}; % 

% other data
country_GDP = dataTable_country{:,2}; % 

%% Processing

% Normalize GDP for marker size
GDP_sqrt = sqrt(country_GDP);
GDP_minSize = 10;
GDP_maxSize = 300;
GDP_norm = (GDP_sqrt - min(GDP_sqrt)) ./ (max(GDP_sqrt) - min(GDP_sqrt));
marker_sizes = GDP_minSize + GDP_norm * (GDP_maxSize - GDP_minSize);

% Trade links to be plotted
% Assume trade_network is a 209x209 symmetric matrix
n = size(trade_network, 1);

% Step 1: Mask upper triangle (excluding diagonal)
upperTriMask = triu(true(n), 1);  % 1 above diagonal

% Step 2: Extract values and their row/col indices
[row_idx, col_idx] = find(upperTriMask);
values = trade_network(upperTriMask);

% Step 3: Sort values and get indices of top k
[sorted_vals, sort_idx] = sort(values, 'descend');
top_k = 200;
top_idx = sort_idx(1:top_k);

% Step 4: Extract top 200 entries
top_rows = row_idx(top_idx);
top_cols = col_idx(top_idx);
top_values = sorted_vals(1:top_k);

% Step 5: Combine into a matrix
top_k_matrix = [top_rows, top_cols, top_values];

% Top N links to pick per country
top_n_per_country = 4;     

per_row_links = [];
for i = 1:n
    row_vals = trade_network(i, :);
    row_vals(i) = -Inf;  % exclude self

    [sorted_row_vals, row_sort_idx] = sort(row_vals, 'descend');
    valid_top_idx = row_sort_idx(1:min(top_n_per_country, n-1));  % handle edge case

    for j = 1:length(valid_top_idx)
        a = i;
        b = valid_top_idx(j);
        if a > b
            [a, b] = deal(b, a);  % ensure unique undirected pair
        end
        per_row_links = [per_row_links; a, b, trade_network(i, b)];
    end
end

% Step 3: Combine and deduplicate
combined = [top_k_matrix; per_row_links];
pair_strs = strcat(string(combined(:,1)), '_', string(combined(:,2)));
[~, unique_idx] = unique(pair_strs, 'stable');
top_links_final = combined(unique_idx, :);

top_links_final = top_links_final(top_links_final(:,3) > 0, :);


n_links = size(top_links_final, 1);

% Normalize trade values for linewidths
% Step 1: Take square root of trade values
sqrt_vals = sqrt(top_links_final(:,3));

% Step 2: Normalize to [0, 1]
norm_vals = (sqrt_vals - min(sqrt_vals)) ./ (max(sqrt_vals) - min(sqrt_vals));

% Step 3: Scale to desired line width range
minWidth = 0.1;
maxWidth = 5;
line_widths = minWidth + norm_vals * (maxWidth - minWidth);

% Map plot
% === World‑map Plot Using gcwaypts ===

figure('Color','w');
worldmap('World');                        % projected axes

% Draw filled coast
%load coastlines
%geoshow(coastlat, coastlon, ...
%    'DisplayType','polygon', 'FaceColor',[0.9 0.9 0.9], 'EdgeColor','none');

% Load and plot country borders using built-in shapefile
land = shaperead('landareas.shp', 'UseGeoCoords', true);

% Draw country borders
geoshow(land, 'DisplayType', 'polygon', ...
    'FaceColor', 'none', 'EdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.5);  % Thin dark borders

countries = shaperead('world-administrative-boundaries.shp', 'UseGeoCoords', true);
geoshow(countries, 'DisplayType', 'polygon', ...
    'FaceColor', 'none', 'EdgeColor', 'k', 'LineWidth', 0.5);

% Draw the scatter plot (returns a group)
scatterm(country_lat, country_long, marker_sizes, ...
    [0 0 0.6], 'filled');

%{
for k = 1:n_links
    i = top_k_matrix(k, 1);
    j = top_k_matrix(k, 2);

    lat1 = country_lat(i);  lon1 = country_long(i);
    lat2 = country_lat(j);  lon2 = country_long(j);

    [latc, lonc] = gcwaypts(lat1, lon1, lat2, lon2, 100);

    

    plotm(latc, lonc, '-', 'LineWidth', line_widths(k), 'Color', [1 0 0 0.6]);  % Red with some transparency
end
%}

for k = 1:n_links
    i = top_links_final(k, 1);
    j = top_links_final(k, 2);

    lat1 = country_lat(i);  lon1 = country_long(i);
    lat2 = country_lat(j);  lon2 = country_long(j);

    % Convert lat/lon to figure coordinates (x-y)
    [x1, y1] = mfwdtran(lat1, lon1);
    [x2, y2] = mfwdtran(lat2, lon2);

    % Draw a straight line on figure, no projection issues
    line([x1 x2], [y1 y2], ...
        'LineWidth', line_widths(k), ...
        'Color', [0.4 0.8 1 0.5]);  % Red with transparency

end


