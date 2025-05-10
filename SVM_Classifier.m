function Model = SVM_Classifier(trainingData)


    numTrials = size(trainingData, 1);
    numAngles = size(trainingData, 2);
    numRounds = 5; % For 300 to 380 in 20ms steps

    % Preallocate storage
    Model.svms = cell(1, numRounds);
    Model.featureMeans = cell(1, numRounds);
    Model.featureStds = cell(1, numRounds);

    for round = 1:numRounds
        sampleIdx = 1;

        % Extract one feature vector to get dimensionality
        exampleSpikes = trainingData(1,1).spikes(:, 1:(300 + 20*(round - 1)));
        exampleFeature = extractSVMFeatures(exampleSpikes, round);
        numFeatures = length(exampleFeature);
        rawFeatures = zeros(numTrials * numAngles, numFeatures);
        angles = zeros(numTrials * numAngles, 1);

        % Build feature matrix
        for trial = 1:numTrials
            for angle = 1:numAngles
                spikes = trainingData(trial, angle).spikes(:, 1:(300 + 20*(round - 1)));
                featureVector = extractSVMFeatures(spikes, round); % ✅ use shared extractor

                rawFeatures(sampleIdx, :) = featureVector;
                angles(sampleIdx) = angle;
                sampleIdx = sampleIdx + 1;

            end
        end

        % Normalize
        featureMean = mean(rawFeatures, 1);
        featureStd = std(rawFeatures, 0, 1) + 1e-6;
        features = (rawFeatures - featureMean) ./ featureStd;

        % Train SVM
        svm = fitcecoc(features, angles, ...
            'Coding', 'onevsall', ...
            'Learners', 'svm', ...
            'ClassNames', 1:8, ...
            'Prior', 'uniform');

        % Store round-specific model
        Model.svms{round} = svm;
        Model.featureMeans{round} = featureMean;
        Model.featureStds{round} = featureStd;

    end
end
