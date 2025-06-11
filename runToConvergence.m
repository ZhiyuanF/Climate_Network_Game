function [s_equilibrium, globalCost, globalEmission] = runToConvergence(s_initial, params, maxIter)
% runToConvergence  Repeatedly apply best-response updates to a strategy profile
% until convergence or maxIter is reached. Then compute global metrics.
%
%   INPUTS:
%     s_initial : 209×1 initial strategy vector
%     params    : struct of all fixed inputs (s_space, network, etc.)
%     maxIter   : maximum number of iterations to run
%
%   OUTPUTS:
%     s_equilibrium : 209×1 final strategy vector
%     globalCost    : sum of all countries' total costs at s_equilibrium
%     globalEmission: normalized global emission at s_equilibrium

    s_old = s_initial;

    for iter = 1 : maxIter
        s_new = bestResponseUpdate(s_old, params);

        % Check if no changes occurred (convergence)
        if all(s_new == s_old)
            fprintf('Converged at iteration %d.\n', iter);
            s_equilibrium = s_new;
            break;
        end

        s_old = s_new;

        % If we never break, we need a final assignment after the loop
        if iter == maxIter
            fprintf('Reached maximum iteration %d without convergence.\n', maxIter);
            s_equilibrium = s_new;
        end
    end

    % --- Compute global cost and global emissions at equilibrium ---
    [Cost_total, ~, ~, ~] = computeCost(s_equilibrium, params);
    globalCost = sum(Cost_total);

    % The emissionIntensity for each country is (1 - s_space(s_equilibrium)).
    % Then normalized global emissions is:
    emissionIntensity = 1 - params.s_space(s_equilibrium);
    globalEmission = (dot(emissionIntensity, params.country_CO2) / 1e9) ...
                     / params.emission_total;
end