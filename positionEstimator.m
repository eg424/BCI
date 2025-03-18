function [x, y] = positionEstimator(testData, modelParameters)

    % Initial state estimation
    spikes = testData.spikes;
    numNeurons = size(spikes, 1);
    T = 30;
    binSize = 395;

    % Load the classifier 
    classifier = modelParameters.classifier;
    
    % Extract test features
    numTrials = size(testData, 1);
    numAngles = size(testData, 2);
    numNeurons = size(testData(1,1).spikes, 1);
    numFeatures = 6; % Mean, Variance, Fano Factor, Total Spike Count, ISI Mean, ISI Variance
    
    testFeatures = zeros(numTrials * numAngles, numNeurons * numFeatures);
    trueAngles = zeros(numTrials * numAngles, 1);
    
    sampleIdx = 1;
    
    for trial = 1:numTrials
        for angle = 1:numAngles
            spikes = testData(trial, angle).spikes; % (numNeurons x timeBins)
            spikeTimes = cell(numNeurons, 1);
            ISI_mean = zeros(1, numNeurons);
            ISI_var = zeros(1, numNeurons);
            
            % Compute ISI for each neuron
            for neuron = 1:numNeurons
                spikeIndices = find(spikes(neuron, :) > 0); 
                if length(spikeIndices) > 1
                    ISI = diff(spikeIndices);
                    ISI_mean(neuron) = mean(ISI);
                    ISI_var(neuron) = var(ISI);
                else
                    ISI_mean(neuron) = 0; 
                    ISI_var(neuron) = 0;
                end
            end
    
            % Compute other features per neuron
            meanFiring = mean(spikes, 2)'; 
            varFiring = var(spikes, 0, 2)'; 
            fanoFactor = varFiring ./ (meanFiring + 1e-6); % Avoid division by zero
            tot_spikes = sum(spikes, 2)'; 
            
            % Concatenate all features into a single row vector
            featureVector = [meanFiring, varFiring, fanoFactor, tot_spikes, ISI_mean, ISI_var];
    
            % Store features and corresponding angle
            testFeatures(sampleIdx, :) = featureVector;
            trueAngles(sampleIdx) = angle;
            sampleIdx = sampleIdx + 1;
        end
    end
    
    % Normalize Features 
    testFeatures = (testFeatures - mean(testFeatures)) ./ (std(testFeatures) + 1e-6);
    
    % Predict angles using Naive Bayes classifier
    predictedAngles = zeros(size(trueAngles));
    
    for i = 1:size(testFeatures, 1)
        probabilities = zeros(8, 1);
        
        for j = 1:length(classifier.angles)
            % Compute likelihood using Gaussian Naive Bayes formula
            % likelihood = exp(-0.5 * ((testFeatures(i, :) - classifier.meanFeature(j, :)).^2 ./ classifier.varFeature(j, :))) ...
            %              ./ sqrt(2 * pi * classifier.varFeature(j, :));
            % 
            % probabilities(j) = classifier.priors(j) * prod(likelihood)

            logLikelihood = -0.5 * sum(((testFeatures(i, :) - classifier.meanFeature(j, :)).^2 ./ classifier.varFeature(j, :))) ...
                - 0.5 * sum(log(2 * pi * classifier.varFeature(j, :)));

            logPrior = log(classifier.priors(j)); % Convert prior to log-space
            
            logProbabilities(j) = logPrior + logLikelihood; % Sum log probabilities

        end
        [~, bestClass] = max(logProbabilities);
        predictedAngles(i) = classifier.angles(bestClass)

        % [~, bestClass] = max(probabilities);
        % predictedAngles(i) = classifier.angles(bestClass);
    end
    
    % Predict the angle using the classifier
    angle_predicted = classifyAngle(spikes, classifier); 

    % Load model parameters based on predicted angle
    A = modelParameters(angle_predicted).A;
    W = modelParameters(angle_predicted).W;
    H = modelParameters(angle_predicted).H;
    Q = modelParameters(angle_predicted).Q;
    avgVelocity = modelParameters(angle_predicted).avgVelocity;

    % Initial estimates
    x_est = [testData.startHandPos(1); testData.startHandPos(2); avgVelocity];
    P_est = eye(4);
    
    % Compute smoothed firing rates
    cumsumSpikes = cumsum(spikes, 2);
    firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
    firingRates = [zeros(numNeurons, binSize-1), firingRates];
    firingRateChanges = diff(firingRates, 1, 2); % First derivative of firing rates
    firingRateChanges = [zeros(numNeurons, 1), firingRateChanges]; % Pad with zeros
    meanFiringRate = mean(firingRates, 2);
    meanFiringRateChange = mean(firingRateChanges, 2);

    for neuron = 1:numNeurons
        spikeIndices = find(spikes(neuron, :) > 0); 
        if length(spikeIndices) > 1
            ISI = diff(spikeIndices); 
            ISI_mean(neuron) = mean(ISI);
            ISI_var(neuron) = var(ISI);
        else
            ISI_mean(neuron) = 0; % If no spikes or only one, ISI set to 0
            ISI_var(neuron) = 0;
        end
    end

    meanISI = mean(ISI_mean);
    varISI = mean(ISI_var);

    % Classify angle using Naive Bayes model
    angle_predicted = classifyAngle(spikes, classifier); 

    Z = [firingRates; repmat(meanFiringRate, 1, size(firingRates,2)); ...
            repmat(meanISI, 1, size(firingRates,2)); repmat(varISI, 1, size(firingRates,2))];
    
    % Compute firing rates
    cumsumSpikes = cumsum(spikes, 2);
    firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
    firingRates = [zeros(numNeurons, binSize-1), firingRates];
   
    % Batch Kalman filter update
    X_pred = A * x_est;
    P_pred = A * P_est * A' + W;
    lambda = 1e-6;  % regularisation constant
    S = H * P_pred * H' + Q + lambda * eye(size(Q));
    K = (P_pred * H') / S;
    X_est = X_pred + K * (Z - H * X_pred);
    P_est = P_pred - K * H * P_pred;

    % Output estimated position
    x = X_est(1, end);
    y = X_est(2, end);
    
    % for t = 1:T:size(spikes, 2)
    %     % Prediction step
    %     X_pred = A * x_est;
    %     P_pred = A * P_est * A' + W;
    % 
    %     % Compute Kalman Gain
    %     lambda = 1e-6;  % Regularization constant
    %     S = H * P_pred * H' + Q + lambda * eye(size(Q));
    %     K = (P_pred * H') / S;
    % 
    %     % Update step
    %     x_est = X_pred + K * (Z(:, t) - H * X_pred);
    %     P_est = P_pred - K * H * P_pred;
    % end
    % 
    % % Output estimated position
    % x = x_est(1);
    % y = x_est(2);
end