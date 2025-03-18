function angle_predicted = classifyAngle(spikes, Model)
    % Extract features from spike data (same as during training)
    numNeurons = size(spikes, 1);
    ISI_mean = zeros(1, numNeurons);
    ISI_var = zeros(1, numNeurons);

    for neuron = 1:numNeurons
        spikeIndices = find(spikes(neuron, :) > 0); % Find spike times
        if length(spikeIndices) > 1
            ISI = diff(spikeIndices); % Compute inter-spike intervals
            ISI_mean(neuron) = mean(ISI);
            ISI_var(neuron) = var(ISI);
        else
            ISI_mean(neuron) = 0; % If no spikes or only one, ISI set to 0
            ISI_var(neuron) = 0;
        end
    end

    % Compute other features for each neuron
    meanFiring = mean(spikes, 2)';
    varFiring = var(spikes, 0, 2)';
    fanoFactor = varFiring ./ (meanFiring + 1e-6); % Avoid division by zero
    tot_spikes = sum(spikes, 2)';

    % Concatenate all features into a single feature vector
    featureVector = [meanFiring, varFiring, fanoFactor, tot_spikes, ISI_mean, ISI_var];

    % Normalize features (same as during training)
    featureVector = (featureVector - Model.meanFeature) ./ sqrt(Model.varFeature);
    
    % Classify angle using the trained Naive Bayes model
    prob = zeros(1, length(Model.angles));
    
    % Loop over each possible angle class
    for i = 1:length(Model.angles)
        % Compute the log-likelihood for each feature
        log_likelihood = 0;
        for j = 1:length(featureVector)
            % Calculate the log of the normal PDF for each feature
            log_likelihood = log_likelihood + log(normpdf(featureVector(j), Model.meanFeature(i,j), sqrt(Model.varFeature(i,j))));
        end
        
        % Add log-prior for the angle class
        prob(i) = log_likelihood + log(Model.priors(i));
    end

    % Predicted angle is the one with the highest probability
    [~, angle_predicted] = max(prob);
end