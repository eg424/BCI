function [x, y] = positionEstimator(test_data, modelParameters)
    spikes = test_data.spikes;
    
    % Compute cum spike counts as features
    cum_spikes = sum(spikes, 2)';

    % Predict velocity
    xdot = cum_spikes * modelParameters.Wx;
    ydot = cum_spikes * modelParameters.Wy;

    % Euler's integration to predict position
    start_pos = test_data.startHandPos;
    duration = size(spikes, 2);

    x = start_pos(1) + xdot * duration;
    y = start_pos(2) + ydot * duration;
end