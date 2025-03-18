function [x, y] = positionEstimator(test_data, modelParameters)
    spikes = test_data.spikes;
    
    % Compute mean firing rates as features
    avg_firing_rates = mean(spikes, 2)';  % (1 x num_neurons)
    
    % Predict Hand Position using Linear Model
    x = avg_firing_rates * modelParameters.Wx;
    y = avg_firing_rates * modelParameters.Wy;
end