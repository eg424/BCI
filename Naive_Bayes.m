function Model = Naive_Bayes(trainingData)
    % Extract the number of trials, angles, and neurons
    numTrials = size(trainingData, 1);
    numAngles = size(trainingData, 2);
    numNeurons = size(trainingData(1,1).spikes, 1);
    numFeatures = 6; % Mean, Variance, Fano Factor, Total Spike Count, ISI Mean, ISI Variance

    % Initialize feature and angle matrices
    features = zeros(numTrials * numAngles, numNeurons * numFeatures);
    angles = zeros(numTrials * numAngles, 1);
    
    sampleIdx = 1; % Index for feature matrix
    
    for trial = 1:numTrials
        for angle = 1:numAngles
            % Extract spike data
            spikes = trainingData(trial, angle).spikes; % (numNeurons x timeBins)
            spikeTimes = cell(numNeurons, 1);
            ISI_mean = zeros(1, numNeurons);
            ISI_var = zeros(1, numNeurons);
            
            % Compute ISI for each neuron
            for neuron = 1:numNeurons
                spikeIndices = find(spikes(neuron, :) > 0); % Find spike times
                if length(spikeIndices) > 1
                    ISI = diff(spikeIndices); % Compute inter-spike intervals
                    ISI_mean(neuron) = mean(ISI);
                    ISI_var(neuron) = var(ISI);
                else
                    ISI_mean(neuron) = 0; % If no spikes or one spike, set ISI to zero
                    ISI_var(neuron) = 0;
                end
            end
            
            % Compute other features per neuron
            meanFiring = mean(spikes, 2)'; % (1 x numNeurons)
            varFiring = var(spikes, 0, 2)'; % (1 x numNeurons)
            fanoFactor = varFiring ./ (meanFiring + 1e-6); % Avoid division by zero
            tot_spikes = sum(spikes, 2)'; % Total spike count
            
            % Concatenate all features into a single row vector
            featureVector = [meanFiring, varFiring, fanoFactor, tot_spikes, ISI_mean, ISI_var];

            % Store features and corresponding angle
            features(sampleIdx, :) = featureVector;
            angles(sampleIdx) = angle;
            sampleIdx = sampleIdx + 1;
        end
    end

    % Normalize Features (Standardization)
    features = (features - mean(features)) ./ (std(features) + 1e-6);

    % Train Naive Bayes Classifier
    uniqueAngles = unique(angles);
    numClasses = length(uniqueAngles);

    meanFeature = zeros(numClasses, size(features, 2));
    varFeature = zeros(numClasses, size(features, 2));
    priors = zeros(numClasses, 1);

    for i = 1:numClasses
        idx = (angles == uniqueAngles(i));
        classFeatures = features(idx, :);
        
        meanFeature(i, :) = mean(classFeatures, 1);
        varFeature(i, :) = var(classFeatures, 0, 1) + 1e-6; % Add small value to avoid zero variance
        priors(i) = sum(idx) / length(angles);  % Compute class prior probabilities
    end

    % Store model parameters
    Model.meanFeature = meanFeature;
    Model.varFeature = varFeature;
    Model.priors = priors;
    Model.angles = uniqueAngles;
end