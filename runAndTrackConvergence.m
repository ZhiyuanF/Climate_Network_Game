function [s_equilibrium, costHistory, emissionHistory] = runAndTrackConvergence(s_initial, params, maxIter)
% runAndTrackConvergence  Repeatedly apply best-response updates, tracking
% the global cost and emission after each iteration, until convergence or maxIter.
%
%   INPUTS:
%     s_initial   : 209×1 initial strategy vector
%     params      : struct of all fixed inputs (s_space, network, etc.)
%     maxIter     : maximum number of iterations
%
%   OUTPUTS:
%     s_equilibrium : 209×1 final strategy vector
%     costHistory   : Iteration-by-iteration global cost (up to convergence)
%     emissionHistory : Iteration-by-iteration global emission (up to convergence)

    % Initialize arrays to store cost/emission each iteration
    costHistory = zeros(maxIter, 1);
    emissionHistory = zeros(maxIter, 1);

    s_old = s_initial;
    
    % For iteration 1 to maxIter
    for iter = 1 : maxIter
        
        % 1) Perform one best-response sweep
        s_new = bestResponseUpdate(s_old, params);

        % 2) Compute cost and emission with the *new* strategy profile
        [Cost_total, ~, ~, ~] = computeCost(s_new, params);
        costHistory(iter) = sum(Cost_total);  % sum of cost_total for all countries
        
        emissionIntensity = 1 - params.s_space(s_new);
        emissionHistory(iter) = ...
            (dot(emissionIntensity, params.country_CO2) / 1e9) / params.emission_total;

        % 3) Check if we have converged
        if all(s_new == s_old)
            % If converged, store s_new as final
            %fprintf('Converged at iteration %d.\n', iter);
            s_equilibrium = s_new;
            
            % Truncate the history arrays to the actual iteration count
            costHistory = costHistory(1:iter);
            emissionHistory = emissionHistory(1:iter);
            return;
        end

        % Prepare for next iteration
        s_old = s_new;

        % If we never break, then we didn't converge within maxIter
        if iter == maxIter
            fprintf('Reached maximum iteration %d without convergence.\n', maxIter);
            s_equilibrium = s_new;
            
            % Keep the full cost/emission history
            return;
        end
    end
end