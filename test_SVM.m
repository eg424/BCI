% Load dataset
data = load("monkeydata_training.mat");
TT_data = data.trial;
numTrials = size(TT_data, 1);
numAngles = size(TT_data, 2);

% Set parameters for feature extraction
numNeurons = size(TT_data(1,1).spikes, 1);
numFeatures = 6;           % mean, var, fano, spike count, ISI mean, ISI var
numTemporalWindows = 3;    % now includes the 301–350 ms window
%totalFeatures = numNeurons * (numFeatures + numTemporalWindows);
accuracies = zeros(numRounds, 1);





roundWindows = [300, 320, 340, 360, 380];
numRounds = length(roundWindows);







% Split data into 80% training and 20% testing
trainFraction = 0.8;
randIdx = randperm(numTrials);
numTrainTrials = round(trainFraction * numTrials);
trainIndices = randIdx(1:numTrainTrials);
testIndices = randIdx(numTrainTrials+1:end);

trainingData = TT_data(trainIndices, :);
testingData = TT_data(testIndices, :);

% Train the SVM classifier
Model = SVM_Classifier(trainingData);

% Extract test features from testing data
numTestTrials = size(testingData, 1);

% Test each round separately
for r = 1:numRounds
    T = roundWindows(r);
    %fprintf("\n=== Evaluating Round %d (time window = %d ms) ===\n", r, T);

    % === Determine feature length dynamically ===
    exampleSpikes = testingData(1, 1).spikes(:, 1:T);
    exampleFeature = extractSVMFeatures(exampleSpikes, r);
    featureLength = length(exampleFeature);

    % Allocate feature + labels arrays
    numTestSamples = numel(testingData);
    testFeatures = zeros(numTestTrials * numAngles, length(exampleFeature));

    trueAngles = zeros(numTestSamples, 1);

    % Fill features
    sampleIdx = 1;
    for trial = 1:size(testingData, 1)
        for angle = 1:numAngles
            spikes = testingData(trial, angle).spikes(:, 1:T);
            feat = extractSVMFeatures(spikes, r);
            testFeatures(sampleIdx, :) = feat;
            trueAngles(sampleIdx) = angle;
            sampleIdx = sampleIdx + 1;
        end
    end
    
    %fprintf("Round %d\n", r);
    %fprintf("testFeatures: %d cols\n", size(testFeatures, 2));
    %fprintf("Model.featureMeans{%d}: %d cols\n", r, length(Model.featureMeans{r}));
    %fprintf("Model.featureStds{%d}: %d cols\n", r, length(Model.featureStds{r}));
    % Normalize
    testFeatures = (testFeatures - Model.featureMeans{r}) ./ Model.featureStds{r};

    % Predict
    predictedAngles = predict(Model.svms{r}, testFeatures);

    % Evaluate
    correct = sum(predictedAngles == trueAngles);
    accuracy = (correct / length(trueAngles)) * 100;
    %fprintf("Accuracy at %d ms: %.2f%%\n", T, accuracy);
    accuracies(r) = accuracy;

    % Optional: Confusion matrix
    figure;
    %confusionchart(confusionmat(trueAngles, predictedAngles));
    %title(sprintf("Confusion Matrix - SVM @ %d ms", T));
end

accuracy = (sum(accuracies)/numRounds);
errorRate = 100 - accuracy;

fprintf('SVM Classifier Accuracy: %.2f%%\n', accuracy);
fprintf('Error Rate: %.2f%%\n', errorRate);