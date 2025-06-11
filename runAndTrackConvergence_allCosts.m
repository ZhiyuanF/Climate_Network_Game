function [s_equilibrium, costHistory, mitigationCostHistory, economicCostHistory, networkCostHistory, emissionHistory] = runAndTrackConvergence_allCosts(s_initial, params, maxIter)
% runAndTrackConvergence_allCosts  
% Repeatedly apply best-response updates, tracking all cost components and 
% emissions after each iteration, until convergence or maxIter.
%
%   INPUTS:
%     s_initial   : 209×1 initial strategy vector
%     params      : struct of all fixed inputs (s_space, network, etc.)
%     maxIter     : maximum number of iterations
%
%   OUTPUTS:
%     s_equilibrium       : 209×1 final strategy vector
%     costHistory         : Iteration-by-iteration total global cost
%     mitigationCostHistory: Iteration-by-iteration mitigation cost
%     economicCostHistory : Iteration-by-iteration economic loss cost
%     networkCostHistory  : Iteration-by-iteration network externality cost
%     emissionHistory     : Iteration-by-iteration global emissions (normalized)

    % Initialize arrays to store cost/emission each iteration
    costHistory           = zeros(maxIter, 1);
    mitigationCostHistory = zeros(maxIter, 1);
    economicCostHistory   = zeros(maxIter, 1);
    networkCostHistory    = zeros(maxIter, 1);
    emissionHistory       = zeros(maxIter, 1);

    s_old = s_initial;

    for iter = 1:maxIter
        % 1) Perform one best-response sweep
        s_new = bestResponseUpdate(s_old, params);

        % 2) Compute all cost components
        [Cost_total, Cost_mitigation, Cost_economic, Cost_network] = computeCost(s_new, params);
        costHistory(iter)           = sum(Cost_total);
        mitigationCostHistory(iter) = sum(Cost_mitigation);
        economicCostHistory(iter)   = sum(Cost_economic);
        networkCostHistory(iter)    = sum(Cost_network);

        % 3) Compute normalized emissions
        emissionIntensity = 1 - params.s_space(s_new);
        emissionHistory(iter) = ...
            (dot(emissionIntensity, params.country_CO2) / 1e9) / params.emission_total;

        % 4) Check convergence
        if all(s_new == s_old)
            s_equilibrium = s_new;

            % Truncate histories to actual length
            costHistory           = costHistory(1:iter);
            mitigationCostHistory = mitigationCostHistory(1:iter);
            economicCostHistory   = economicCostHistory(1:iter);
            networkCostHistory    = networkCostHistory(1:iter);
            emissionHistory       = emissionHistory(1:iter);
            return;
        end

        s_old = s_new;

        if iter == maxIter
            fprintf('Reached maximum iteration %d without convergence.\n', maxIter);
            s_equilibrium = s_new;
            return;
        end
    end
end
