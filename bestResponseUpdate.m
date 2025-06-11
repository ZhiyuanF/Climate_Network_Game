function s_next = bestResponseUpdate(s_initial, params)
% bestResponseUpdate performs one best-response iteration for all countries.
%
%   s_initial : 209×1 vector of current strategies (indices in [1..120])
%   params    : struct of fixed model hyperparameters
%
%   s_next    : 209×1 vector of updated strategies after one iteration

    nCountries = length(s_initial);   % should be 209
    s_next = s_initial;              % start with the current strategies

    % Loop over each country
    for i = 1 : nCountries
        % --- 1) Collect possible strategies for this country 'i' ---
        currentStrat = s_next(i);                     % e.g. 40
        candidates = currentStrat;                    % always keep current strategy

        if currentStrat > 1
            candidates = [candidates, currentStrat - 1];
        end
        if currentStrat < length(params.s_space)
            candidates = [candidates, currentStrat + 1];
        end
        
        % --- 2) Evaluate each candidate's cost for country i only ---
        costValues = zeros(size(candidates));
        for c = 1 : length(candidates)
            s_temp = s_next;        % copy the current strategy profile
            s_temp(i) = candidates(c); 
            
            % Compute cost for this full profile
            [Cost_total, ~, ~, ~] = computeCost(s_temp, params);
            
            % Only country i's cost matters for i's best response
            costValues(c) = Cost_total(i);
        end

        % --- 3) Pick the strategy index that yields the lowest cost for i ---
        [~, idxMin] = min(costValues);
        bestStrat = candidates(idxMin);
        
        % --- 4) Update i's strategy in s_next ---
        s_next(i) = bestStrat;
    end
end
