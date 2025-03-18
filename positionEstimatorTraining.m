function modelParameters = positionEstimatorTraining(training_data)
    num_neurons = size(training_data(1,1).spikes, 1);
    num_trials = size(training_data, 1);
    num_angles = size(training_data, 2);

    % Collect spike counts & corresponding hand positions
    X_data = [];
    xdot_data = [];
    ydot_data = [];

    for trial = 1:num_trials
        for angle = 1:num_angles
            spikes = training_data(trial, angle).spikes;
            time_bins = size(spikes, 2);
            
            % Cumulative spike count - instead of mean firing rate
            cum_spikes = sum(spikes, 2)';
            
            % Target x,y velocities - instead of position
            start_pos = training_data(trial, angle).handPos(1:2, 1);
            final_pos = training_data(trial, angle).handPos(1:2, end);
            duration = size(training_data(trial, angle).handPos, 2);

            v = (final_pos - start_pos) / duration;

            X_data = [X_data; cum_spikes];
            xdot_data = [xdot_data; v(1)];
            ydot_data = [ydot_data; v(2)];
        end
    end
    
    % Training
    modelParameters.Wx = pinv(X_data) * xdot_data; 
    modelParameters.Wy = pinv(X_data) * ydot_data;

end