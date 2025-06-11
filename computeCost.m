function [Cost_total, Cost_mitigation, Cost_economic, Cost_network] = ...
    computeCost(s_initial, params)
% computeCost  Compute total and component costs for each country's strategy.
%
%   INPUTS:
%     s_initial: 209×1 vector of strategy indices (e.g., each entry in [1..120])
%     params:    Struct holding:
%        - s_space               (1×120)
%        - C_mitigation          (function handle or array)
%        - network               (209×209)
%        - country_CO2           (209×1)
%        - country_GDP           (209×1)
%        - country_network_sens  (209×1)
%        - emission_total        (scalar)
%        - p1, p2                (polynomial coeff. vectors)
%
%   OUTPUTS:
%     Cost_total:      209×1 vector of total cost for each country
%     Cost_mitigation: 209×1 vector of mitigation costs
%     Cost_economic:   209×1 vector of economic costs
%     Cost_network:    209×1 vector of network costs
%

    % Unpack from params for easier reference
    s_space              = params.s_space;
    C_mitigation         = params.C_mitigation;
    network              = params.network;
    country_CO2          = params.country_CO2;
    country_GDP          = params.country_GDP;
    country_network_sens = params.country_network_sens;
    emission_total       = params.emission_total;
    p1                   = params.p1;
    p2                   = params.p2;

    % Part 1, mitigation cost
    % Each country's chosen mitigation level:
    chosenMitigation = C_mitigation(s_initial);  % yields a 209×1 vector
    % First component of the cost: mitigation cost
    Cost_mitigation = chosenMitigation' .* country_CO2;
    
    
    % Part 2, economic cost
    emissionIntensity = 1 - s_space(s_initial);  % yields a 209×1 vector
    % Global total emissions
    emission_global = dot(emissionIntensity',country_CO2)/1000000000/emission_total;
    % economic cost for each country
    Cost_economic = polyval(p2, polyval(p1, emission_global))*country_GDP;
    
    
    % Part 3, network cost
    effectiveNetworkStrategy = network * s_space(s_initial)';
    % network influence for each country
    influence_network = round((s_space(s_initial)' - effectiveNetworkStrategy),4).^2;
    % network cost for each country
    Cost_network = influence_network .* country_network_sens .* country_GDP;

    % Total Cost
    Cost_total = Cost_mitigation + Cost_economic + Cost_network;  % 209×1

end

