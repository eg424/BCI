 function predictedAngle = classifyAngle_SVM(spikes, Model)

    % Determine round based on spike length
    timeLen = size(spikes, 2);
    timeWindows = [300, 320, 340, 360, 380];

    % Find the correct round index based on current spike window
    round = find(timeLen <= timeWindows, 1, 'first');
    if isempty(round)
        round = numel(timeWindows);  % use the last round if exceeds all
    end

    % Extract feature vector from spikes
    featureVector = extractSVMFeatures(spikes, round);

    % Normalize using training mean/std for that round
    featureMean = Model.featureMeans{round};
    featureStd = Model.featureStds{round};
    featureVector = (featureVector - featureMean) ./ featureStd;

    % Predict using corresponding SVM
    svmModel = Model.svms{round};
    predictedAngle = predict(svmModel, featureVector);
end



