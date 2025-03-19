function [modelParameters] = positionEstimatorTraining(training_data)
    % Arguments: training_data contains trialID, spikes, and hand positions for each trial.
    
    % Prepare Data for PCA + LDA
    n_trials = length(training_data);
    n_neurons = size(training_data(1).spikes, 1); % Assuming spikes size is [98 x time_steps]
    n_angles = 8; % 8 reaching angles

    % Find the minimum number of time steps across all trials and angles
    min_time_steps = inf; % Initialize to a large number
    for trial = 1:n_trials
        for angle = 1:n_angles
            spikes = training_data(trial, angle).spikes;
            handPos = training_data(trial, angle).handPos(1:2, :);
            
            % Find the minimum time steps for each trial and angle
            min_time_steps = min(min_time_steps, min(size(spikes, 2), size(handPos, 2)));
        end
    end
    
    % Collect all spikes and positions for PCA + LDA
    all_spikes = [];
    all_positions = [];
    labels = [];
    
    % Loop again and truncate the data to the minimum time steps across all trials and angles
    for trial = 1:n_trials
        for angle = 1:n_angles
            % Get spikes and handPos
            spikes = training_data(trial, angle).spikes; % Neural spike data for this trial and angle
            handPos = training_data(trial, angle).handPos(1:2, :); % Only x, y positions
            
            % Debug: Check the dimensions of spikes and handPos
            disp(['Trial: ', num2str(trial), ', Angle: ', num2str(angle)]);
            disp(['Spikes dimensions: ', num2str(size(spikes))]);
            disp(['Hand positions dimensions: ', num2str(size(handPos))]);
            
            % Truncate to the minimum time steps found across all trials and angles
            spikes = spikes(:, 1:min_time_steps);
            handPos = handPos(:, 1:min_time_steps);
            
            % Debug: Check the new dimensions after truncation
            disp(['New Spikes dimensions: ', num2str(size(spikes))]);
            disp(['New Hand positions dimensions: ', num2str(size(handPos))]);

            % Ensure we collect the same number of rows in both matrices
            time_steps = min_time_steps; % Use the same time steps for both
            all_spikes = [all_spikes; spikes(:, 1:time_steps)']; % [time_steps x 98]
            all_positions = [all_positions; handPos(:, 1:time_steps)']; % [time_steps x 2]
            labels = [labels; repmat(angle, time_steps, 1)]; % Angle as label

        end
    end
    
    % Debug: Check the dimensions of the collected data
    disp(['All spikes size: ', num2str(size(all_spikes))]);
    disp(['All positions size: ', num2str(size(all_positions))]);

    % Check if dimensions of all_spikes and all_positions match for concatenation
    if size(all_spikes, 1) ~= size(all_positions, 1)
        error('Mismatch in the number of time steps between spikes and positions.');
    end

    % Apply PCA to the spike data (dimensionality reduction)
    [coeff, score, ~, ~, explained] = pca(all_spikes);
    
    % Debug: Check PCA output
    disp('PCA Completed');
    disp(['PCA components: ', num2str(size(coeff))]);
    disp(['Explained variance: ', num2str(explained')]);

    % Reduce spikes data using the first few principal components
    explained_variance_threshold = 95; % Retain components explaining 95% variance
    cumulative_variance = cumsum(explained);
    n_components = find(cumulative_variance >= explained_variance_threshold, 1);
    reduced_spikes = score(:, 1:n_components); % Reduced spike data
    
    % Apply LDA for position classification based on reduced spikes
    % LDA works with class labels, so we use the reaching angles (labels)
    lda_model = fitcdiscr(reduced_spikes, labels);
    
    % Store PCA and LDA model parameters
    modelParameters.pca_coeff = coeff(:, 1:n_components); % PCA coefficients (principal components)
    modelParameters.lda_model = lda_model; % LDA model
    
end
