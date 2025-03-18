% Load dataset
data = load("monkeydata_training.mat");
TT_data = data.trial;
numTrials = size(TT_data, 1);
numAngles = size(TT_data, 2);

% Set parameters for feature extraction
numNeurons = size(TT_data(1,1).spikes, 1);
numFeatures = 6; % Mean, Variance, Fano Factor, Total Spike Count, ISI Mean, ISI Variance

% Split data into 70% training and 30% testing
trainFraction = 0.8;
randIdx = randperm(numTrials);
numTrainTrials = round(trainFraction * numTrials);
trainIndices = randIdx(1:numTrainTrials);
testIndices = randIdx(numTrainTrials+1:end);

trainingData = TT_data(trainIndices, :);
testingData = TT_data(testIndices, :);

% Train the Naive Bayes classifier
Model = Naive_Bayes(trainingData);

% Extract test features from testing data
numTestTrials = size(testingData, 1);
testFeatures = zeros(numTestTrials * numAngles, numNeurons * numFeatures);
trueAngles = zeros(numTestTrials * numAngles, 1);
sampleIdx = 1;

for trial = 1:numTestTrials
    for angle = 1:numAngles
        spikes = testingData(trial, angle).spikes; % (numNeurons x timeBins)
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
    probabilities = zeros(length(Model.angles), 1);
    
    for j = 1:length(Model.angles)
        % Compute likelihood using Gaussian Naive Bayes formula
        likelihood = exp(-0.5 * ((testFeatures(i, :) - Model.meanFeature(j, :)).^2 ./ Model.varFeature(j, :))) ...
                     ./ sqrt(2 * pi * Model.varFeature(j, :));
        probabilities(j) = Model.priors(j) * prod(likelihood);
    end
    
    [~, bestClass] = max(probabilities);
    predictedAngles(i) = Model.angles(bestClass);
end

% Compute accuracy
correctPredictions = sum(predictedAngles == trueAngles);
accuracy = (correctPredictions / length(trueAngles)) * 100;
errorRate = 100 - accuracy;

fprintf('Naive Bayes Classifier Accuracy: %.2f%%\n', accuracy);
fprintf('Error Rate: %.2f%%\n', errorRate);

% Display confusion matrix
figure;
confusionMat = confusionmat(trueAngles, predictedAngles);
confusionchart(confusionMat);
title('Confusion Matrix for Naive Bayes Classification');
xlabel('Predicted Angle');
ylabel('True Angle');