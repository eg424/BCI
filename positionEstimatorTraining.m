function [modelParameters] = positionEstimatorTraining(training_data)
    % Preprocess data
    [XTrain, YTrain_speed, YTrain_direction] = preprocessData(training_data);

    fprintf('Training data size: %d sequences\n', length(XTrain));
    fprintf('Training speed labels size: %d sequences\n', length(YTrain_speed));
    fprintf('Training direction labels size: %d sequences\n', length(YTrain_direction));

    % Ensure consistency in sample count
    minSamples = min([length(XTrain), length(YTrain_speed), length(YTrain_direction)]);
    XTrain = XTrain(1:minSamples);
    YTrain_speed = YTrain_speed(1:minSamples);
    YTrain_direction = YTrain_direction(1:minSamples);

    % Convert labels to matrices and ensure correct dimensions:
    % For speed, we want a 1xN matrix; for direction, 2xN.
    if iscell(YTrain_speed)
        YTrain_speed = cell2mat(YTrain_speed(:))';  % Now [1 x numSamples]
    else
        YTrain_speed = YTrain_speed';
    end

    if iscell(YTrain_direction)
        YTrain_direction = cell2mat(YTrain_direction(:))';  % Now [2 x numSamples]
    else
        YTrain_direction = YTrain_direction';
    end

    % Display sizes
    disp('Final XTrain size: variable-length sequences in cell array');
    disp(['Final YTrain_speed size: ', mat2str(size(YTrain_speed))]);      % Expected: [1, numSamples]
    disp(['Final YTrain_direction size: ', mat2str(size(YTrain_direction))]);  % Expected: [2, numSamples]

    % --- Reshape XTrain Sequences ---
    % Each cell in XTrain is currently [T x C] (e.g., [3 x 98]).
    % We reshape it to [T x 1 x C] so that when using "TBC" (Time, Batch, Channel),
    % a mini-batch will have dimensions [T x miniBatchSize x C].
    for i = 1:length(XTrain)
        seq = XTrain{i};  % size: [T x 98]
        XTrain{i} = reshape(seq, size(seq,1), 1, size(seq,2)); % becomes [T x 1 x 98]
    end

    % --- Define the Networks ---
    % Speed Prediction Branch
    layersSpeed = [
        sequenceInputLayer(98, 'Name', 'input', 'Normalization', 'none')
        lstmLayer(50, 'OutputMode', 'sequence', 'Name', 'lstm')
        fullyConnectedLayer(50, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(10, 'Name', 'fc_speed_1')
        reluLayer('Name', 'relu_speed')
        fullyConnectedLayer(1, 'Name', 'speed_output')
    ];

    % Direction Prediction Branch (cos and sin)
    layersDirection = [
        sequenceInputLayer(98, 'Name', 'input', 'Normalization', 'none')
        lstmLayer(50, 'OutputMode', 'last', 'Name', 'lstm')
        fullyConnectedLayer(50, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(10, 'Name', 'fc_dir_1')
        reluLayer('Name', 'relu_dir')
        fullyConnectedLayer(2, 'Name', 'direction_output')
    ];

    % --- Training Options ---
    % For InputDataFormats "TBC": T = time steps, B = batch, C = channel.
    % For TargetDataFormats "CB": C = channel, B = batch.
    options = trainingOptions('adam', ...
        'MaxEpochs', 1, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.005, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'InputDataFormats', 'TBC', ...  % Now each mini-batch is [T x B x C]
        'TargetDataFormats', 'CB');       % Targets: [C x B]

    % Training
    lossFcn = 'mse';

    modelParameters.netSpeed = trainnet(XTrain, YTrain_speed, layersSpeed, lossFcn, options);
    modelParameters.netDirection = trainnet(XTrain, YTrain_direction, layersDirection, lossFcn, options);

    % Fake networks that return random outputs for testing
    % modelParameters.netSpeed = @(X) rand(size(X,1), 1) * 20;  % Random speed between 0 and 20
    % modelParameters.netDirection = @(X) [cos(rand(size(X,1),1)*2*pi), sin(rand(size(X,1),1)*2*pi)];  % Random direction

end