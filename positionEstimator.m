function [x, y] = positionEstimator(testData, modelParameters)
    % Extract trained network
    net = modelParameters.net;
    
    % Preprocess test spikes
    spikes = testData.spikes;
    numNeurons = size(spikes, 1);
    binSize = 50;
    timeHistory = 10;

    % Compute binned firing rates
    numBins = floor(size(spikes, 2) / binSize);
    binnedSpikes = zeros(numNeurons, numBins);
    for t = 1:numBins
        binStart = (t-1) * binSize + 1;
        binEnd = min(t * binSize, size(spikes, 2));
        binnedSpikes(:, t) = sum(spikes(:, binStart:binEnd), 2) / (binEnd - binStart);
    end
    
    % Normalize using training statistics (assumes training mean/std are known)
    binnedSpikes = (binnedSpikes - mean(binnedSpikes, 2)) ./ (std(binnedSpikes, 0, 2) + 1e-6);

    % Use last few bins as input to LSTM
    if size(binnedSpikes, 2) < timeHistory
    XTest = padarray(binnedSpikes, [0, timeHistory - size(binnedSpikes, 2)], 'pre', 'replicate');
    else
        XTest = binnedSpikes(:, end-timeHistory+1:end);
    end

    % Predict hand position (x, y)
    YPred = predict(net, XTest);
    
    x = YPred(1);
    y = YPred(2);
end