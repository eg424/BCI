
function modelParameters = positionEstimatorTraining(trainingData)
    % Prepare data for LSTM training
    [XTrain, YTrain] = preprocessLSTM(trainingData);

    % Define LSTM network architecture
    numFeatures = size(XTrain{1}, 1); % Number of input neurons
    numResponses = size(YTrain{1}, 1); % Number of output variables (x, y)
    numHiddenUnits = 100; % Hidden units in LSTM

    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
        fullyConnectedLayer(numResponses)
        regressionLayer];

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', 10, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'training-progress');

    % Train the LSTM model
    net = trainNetwork(XTrain, YTrain, layers, options);

    % Store trained network in model parameters
    modelParameters.net = net;
end

function [XTrain, YTrain] = preprocessLSTM(trainingData)
    % Extract and format data for LSTM training
    numTrials = size(trainingData, 1);
    numAngles = size(trainingData, 2);
    
    XTrain = {}; % Input (spike rates)
    YTrain = {}; % Output (hand positions)
    
    binSize = 50; % Time window for binning spikes
    timeHistory = 10; % Number of past bins to use as input
    
    for trial = 1:numTrials
        for angle = 1:numAngles
            spikes = trainingData(trial, angle).spikes;
            handPos = trainingData(trial, angle).handPos(1:2, :); % Use only x, y
            
            % Compute firing rates in bins
            numBins = floor(size(spikes, 2) / binSize);
            binnedSpikes = zeros(size(spikes, 1), numBins);
            for t = 1:numBins
                binStart = (t-1) * binSize + 1;
                binEnd = min(t * binSize, size(spikes, 2));
                binnedSpikes(:, t) = sum(spikes(:, binStart:binEnd), 2) / (binEnd - binStart);
            end

            % Normalize firing rates (z-score)
            binnedSpikes = (binnedSpikes - mean(binnedSpikes, 2)) ./ (std(binnedSpikes, 0, 2) + 1e-6);
            
            % Format as sequences
            for t = timeHistory:numBins
                XTrain{end+1} = binnedSpikes(:, t-timeHistory+1:t); % Input sequence
                YTrain{end+1} = handPos(:, t-timeHistory+1:t); % Make output same length as input
            end

        end
    end
end
