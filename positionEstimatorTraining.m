function modelParameters = positionEstimatorTraining(training_data)
    num_neurons = size(training_data(1,1).spikes, 1);
    num_trials = size(training_data, 1);
    num_angles = size(training_data, 2);

    % Collect spike counts & corresponding hand positions
    X_data = [];
    Y_data = [];

    for trial = 1:num_trials
        for angle = 1:num_angles
            spikes = training_data(trial, angle).spikes;
            time_bins = size(spikes, 2);
            
            % Average firing rate per neuron in trial
            avg_firing_rates = mean(spikes, 2)';
            
            % Target x,y hand position
            final_pos = training_data(trial, angle).handPos(1:2, end)';

            X_data = [X_data; avg_firing_rates]; % Inputs
            Y_data = [Y_data; final_pos];  % Outputs
        end

    % Training
    modelParameters.Wx = pinv(X_data) * Y_data(:, 1); 
    modelParameters.Wy = pinv(X_data) * Y_data(:, 2);
    end

end