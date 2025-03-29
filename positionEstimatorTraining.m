function modelParameters = positionEstimatorTraining2(training_data)
    %======================================================================
    % Trains LSTM model to predict hand positions and angles from training 
    % data (spikes and hand positions). The model is trained using Adam
    % optimiser with custom loss function
    %
    % Inputs:
    % - training_data(i,j).spikes: matrix of spikes for training, [98 x T]
    % - training_data(i,j).handPos: matrix of hand position, [2 x T]
    %
    % Outputs:
    % - modelParameters: trained model parameters (LSTM).
    %
    % Notes:
    % - Followed guide in https://uk.mathworks.com/help/deeplearning/ug/long-short-term-memory-networks.html
    % - The function preprocesses the training data, computes the angle, 
    %   and sorts the data by sequence length to minimise padding.
    % - The LSTM architecture consists of two LSTM layers with dropout
    %   (regularisation) and a fully connected layer for prediction
    % - File includes trained model already (positionEstimatorTrained.mat),
    %   so no need to train. However if anyone wants to play around with it,
    %   current training done using GPU. Made comments to run on CPU.
    %======================================================================

    % Extract training data from the training_data struct
    numTrials = size(training_data, 1);
    numAngles = size(training_data, 2);
    XTrain = cell(1, numTrials);
    YTrain = cell(1, numTrials);
    
    counter = 1;
    for i = 1:numTrials
        for j = 1:numAngles
            spikes = training_data(i,j).spikes;
            handPos = training_data(i,j).handPos(1:2, :);
            
            % Compute movement angle
            deltaPos = diff(handPos, 1, 2);
            theta = atan2(deltaPos(2, :), deltaPos(1, :)); 
            theta = [theta, theta(end)]; % Match length with handPos
            
            XTrain{counter} = spikes;
            YTrain{counter} = [handPos; theta];
            counter = counter + 1;
        end
    end
    
    % Sort sequences by length to minimise padding
    sequenceLengths = cellfun(@(X) size(X, 2), XTrain);
    
    [~, idx] = sort(sequenceLengths);
    XTrain = XTrain(idx);
    YTrain = YTrain(idx);

    numFeatures = 98; % Neurons
    numHidden1 = 200; % Play around
    numHidden2 = 200; % Play around
    numClasses = 3; % x, y, predAngle
   
    % LSTM architecture
    layers = [
        sequenceInputLayer(numFeatures, 'Normalization', 'zerocenter')
        lstmLayer(numHidden1, 'OutputMode', 'sequence')
        dropoutLayer(0.2)
        lstmLayer(numHidden2, 'OutputMode', 'sequence')
        dropoutLayer(0.2)
        fullyConnectedLayer(numClasses)
        ];

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 150, ...
        'MiniBatchSize', 16, ...
        'Shuffle', 'every-epoch', ...
        'InputDataFormats', 'CTB', ...
        'GradientThreshold', 1, ...
        'SequenceLength', 'longest', ...
        'SequencePaddingDirection', 'right', ...
        'Verbose', false, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'gpu'); % GPU Computation (~20') - Delete this line for CPU

    % Custom loss function with weighted MSE
    % lossFunction = @(Y, T) mean((Y(1:2,:) - T(1:2,:)).^2, 'all') + 0.1 * mean((Y(3,:) - T(3,:)).^2, 'all');
    lossFunction = 'mse';

    % Move data to GPU - Comment out if on CPU
    XTrain = cellfun(@(x) gpuArray(x), XTrain, 'UniformOutput', false);
    YTrain = cellfun(@(y) gpuArray(y), YTrain, 'UniformOutput', false);

    % Training
    modelParameters = trainnet(XTrain, YTrain, layers, lossFunction, options);
   
    % Uncomment to save new model
    % save('positionEstimatorTrained.mat', 'modelParameters'); 
end
