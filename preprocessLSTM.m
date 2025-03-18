function [XTrain, YTrain] = preprocessLSTM(trainingData)
    % Extract and format data for LSTM training
    numTrials = size(trainingData, 1);
    numAngles = size(trainingData, 2);
    
    XTrain = {};
    YTrain = {};
    
    timeWindow = 20;  % Adjust based on time history you need
    
    for trial = 1:size(trainingData, 1)
        for angle = 1:size(trainingData, 2)
            
            spikes = trainingData(trial, angle).spikes;
            handPos = trainingData(trial, angle).handPos;
            
            % Ensure consistent time binning
            numBins = floor(size(spikes, 2) / timeWindow);
            
            for bin = 1:numBins
                startIdx = (bin - 1) * timeWindow + 1;
                endIdx = bin * timeWindow;
                
                % Extract spike counts for this window
                spikeCounts = sum(spikes(:, startIdx:endIdx), 2);
                
                % Store input features (spike history)
                XTrain{end+1} = spikeCounts;
                
                % Store target outputs (hand position)
                YTrain{end+1} = handPos(1:2, endIdx);  % X and Y only
        end
    end
end

% Convert to arrays for deep learning
XTrain = cat(2, XTrain{:});  % Convert to numeric array
YTrain = cat(2, YTrain{:});  