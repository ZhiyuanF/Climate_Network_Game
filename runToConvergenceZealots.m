function [final_Cost_Country, costFianl, emission_Final, s_equilibrium, costHistory, emissionHistory] = runToConvergenceZealots(s_initial, params, maxIter, zealotIndicator)
% runToConvergenceZealots Repeatedly applies best-response updates while
% holding designated agents (“zealots”) fixed at their initial strategies,
% and tracks convergence metrics including cost and emission.
%
%   INPUTS:
%     s_initial      : 209×1 initial strategy vector.
%     params         : Structure containing all necessary fixed inputs (e.g., s_space, network parameters, country_CO2, emission_total).
%     maxIter        : Maximum number of iterations to run the update process.
%     zealotIndicator: 209×1 binary vector indicating zealot agents (1 = keep initial strategy, 0 = update normally).
%
%   OUTPUTS:
%     final_Cost_Country: Final cost per country computed using the last strategy profile.
%     costFianl         : Total cost (sum across all countries) from the last iteration.
%     emission_Final    : Total emission value computed in the final iteration.
%     s_equilibrium     : Final strategy vector (209×1), with zealot entries fixed at their initial values.
%     costHistory       : Vector tracking the iteration-by-iteration total cost until convergence or maxIter.
%     emissionHistory   : Vector tracking the iteration-by-iteration total emission until convergence or maxIter.
%
%   Note:
%     The function updates all agent strategies using a best-response dynamic at
%     each iteration, but for agents indicated as zealots (zealotIndicator == 1),
%     their strategies remain equal to their initial setting. Convergence is
%     determined by checking if the strategies of non-zealot agents no longer change.   Note: For zealots (where zealotIndicator == 1), their strategy remains fixed as in s_initial.

    % Validate that the zealotIndicator is of the same size as s_initial.
    if numel(s_initial) ~= numel(zealotIndicator)
        error('The zealotIndicator must be of the same size as s_initial.');
    end

    % Initialize arrays to store cost/emission each iteration.
    costHistory = zeros(maxIter, 1);
    emissionHistory = zeros(maxIter, 1);

    % Initialize the strategy profile with s_initial.
    s_old = s_initial;
    
    for iter = 1:maxIter
        
        % 1) Perform one best-response sweep on ALL agents.
        s_new = bestResponseUpdate(s_old, params);
        
        % 2) Override the strategies of the zealots so that they retain s_initial.
        s_new(zealotIndicator == 1) = s_initial(zealotIndicator == 1);
        
        % 3) Compute cost and emission with the *new* strategy profile.
        [Cost_total, ~, ~, ~] = computeCost(s_new, params);
        costHistory(iter) = sum(Cost_total);  % Total cost across all countries
        
        emissionIntensity = 1 - params.s_space(s_new);
        emissionHistory(iter) = (dot(emissionIntensity, params.country_CO2) / 1e9) / params.emission_total;
        
        % 4) Check convergence (only updating the non-zealot positions).
        % We consider convergence reached if the normal agents' strategies do not change.
        if all(s_new(zealotIndicator == 0) == s_old(zealotIndicator == 0))
            s_equilibrium = s_new;
            costHistory = costHistory(1:iter);
            emissionHistory = emissionHistory(1:iter);

            % output the final total cost as well
            final_Cost_Country = Cost_total;
            % output the final total cost as well
            costFianl = costHistory(end);
            emission_Final = emissionHistory(end);
            return;
        end
        
        % Prepare for next iteration.
        s_old = s_new;
        
        % If we reach maxIter without convergence, output the latest results.
        if iter == maxIter
            fprintf('Reached maximum iteration %d without convergence.\n', maxIter);
            s_equilibrium = s_new;

            % output the final total cost as well
            final_Cost_Country = Cost_total;
            % output the final total cost as well
            costFianl = costHistory(end);
            emission_Final = emissionHistory(end);
            return;
        end
    end
end