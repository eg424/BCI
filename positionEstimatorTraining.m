function [modelParameters] = positionEstimatorTraining(training_data)
    [X_train, Y_train] = preprocessData(training_data);
    
    fprintf('Training data size: %d sequences\n', length(X_train));
    fprintf('Training labels size: %d sequences\n', length(Y_train));
    
    layers = [ 
        sequenceInputLayer(size(X_train{1},2), 'Name', 'input')
        lstmLayer(50, 'OutputMode', 'sequence', 'Name', 'lstm')
        fullyConnectedLayer(20, 'Name', 'fc1')
        reluLayer('Name', 'relu')
        fullyConnectedLayer(2, 'Name', 'output') 
    ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', 10, ...
        'InitialLearnRate', 0.005, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'InputDataFormats', 'TCB');

    lossFcn = 'mse';

    % Debugging (verify dataset consistency before training)
    assert(length(X_train) == length(Y_train), 'Mismatch in training data size');

    modelParameters.net = trainnet(X_train, Y_train, layers, lossFcn, options);
end