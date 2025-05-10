function [decodedPosX, decodedPosY, predictedAngle] = positionEstimator(test_data, modelParameters)
    %==========================================================================
    % Predicts 2D hand positions and movement angle from spikes test data.
    %
    % Inputs:
    %  - test_data.spikes: struct containing spikes to be decoded.
    %
    %  - modelParameters - Trained LSTM to decode outputs.
    %
    % Outputs:
    %   decodedPosX    - Vector of decoded X hand positions, [1 x T].
    %   decodedPosY    - Vector of decoded Y hand positions, [1 x T].
    %   predictedAngle - Vector of predicted movement angles (rad), [1 x T-1].
    %==========================================================================
    %% Preprocessing
    % Extract spikes; expected size is [98 x T]
    spikes = test_data.spikes;
    % disp(['Original spikes size: ', mat2str(size(spikes))]);
        
    % Reshape for CTB (98 x T x 1)
    T = size(spikes,2);
    spikes = reshape(spikes, [98, T, 1]);
    % disp(['Spikes size after reshaping: ', mat2str(size(spikes))]);
    
    dlSpikes = dlarray(spikes, 'CTB'); % (Channels, Time, Batch)
    % disp(['dlSpikes size: ', mat2str(size(dlSpikes))]);
    
    estimatedHandPos = predict(modelParameters, dlSpikes);
    % disp(['Predicted output size: ', mat2str(size(estimatedHandPos))]);
    
    % Convert to numeric array and squeeze dimensions
    estimatedHandPos = extractdata(estimatedHandPos);
    estimatedHandPos = squeeze(estimatedHandPos);
    % disp(['Final estimatedHandPos size after squeezing: ', mat2str(size(estimatedHandPos))]);
    
    %% Predictions
    decodedPosX = estimatedHandPos(1,end);
    decodedPosY = estimatedHandPos(2,end);
    predictedAngle = estimatedHandPos(3,end);
end