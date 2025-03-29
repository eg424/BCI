function [modelParameters] = positionEstimatorTraining(training_data, binSize)

    if nargin < 2
        binSize = 20;  % Default bin size if not specified
    end

    % Preprocess data
    [X, Y_speed, Y_direction, Y_angle] = preprocessData(training_data, binSize);

    %fprintf('Training data size: %d sequences\n', length(X));
    %fprintf('Training speed labels size: %d sequences\n', length(Y_speed));
    %fprintf('Training direction labels size: %d sequences\n', length(Y_direction));

    % Ensure consistency in sample count
    minSamples = min([length(X), length(Y_speed), length(Y_direction)]);
    X = X(1:minSamples);
    Y_speed = Y_speed(1:minSamples);
    Y_direction = Y_direction(1:minSamples);
    Y_angle = Y_angle(1:minSamples);

    % Convert labels to matrices and ensure correct dimensions:

    % For speed, we want a 1xN matrix; for direction, 2xN.
    if iscell(Y_speed)
        Y_speed = cell2mat(Y_speed(:))';  % Now [1 x numSamples]
    else
        Y_speed = Y_speed';
    end

    if iscell(Y_direction)
        Y_direction = cell2mat(Y_direction(:))';  % Now [2 x numSamples]
    else
        Y_direction = Y_direction';
    end

    if iscell(Y_angle)
        Y_angle = cell2mat(Y_angle(:))';  % Now [1 x numSamples]
    else
        Y_angle = Y_angle';
    end

    % --- Reshape X Sequences ---
    % Each cell in X is currently [T x C] (e.g., [3 x 98]).
    % We reshape it to [T x 1 x C] so that when using "TBC" (Time, Batch, Channel),
    % a mini-batch will have dimensions [T x miniBatchSize x C].
    for i = 1:length(X)
        seq = X{i};  % size: [T x 98]
        % Append the sine and cosine of the angle to the spike data
        angle = Y_angle(i);  % Regular indexing
        angle_sin = sin(angle);
        angle_cos = cos(angle);

        % Assuming angle_sin and angle_cos are scalars, repeat them for each time step
        angle_sin = repmat(angle_sin, size(seq, 1), 1);  % Repeat for each time step (632 times)
        angle_cos = repmat(angle_cos, size(seq, 1), 1);  % Repeat for each time step (632 times)
        
        % Verify dimensions
        % disp(size(seq));
        %disp(size(angle_sin));
        % disp(size(angle_cos));
        
        % Now concatenate them
        X{i} = cat(2, seq, angle_sin, angle_cos);  % Concatenate along the second dimension
        
        X{i} = reshape(X{i}, size(X{i}, 1), 1, size(X{i}, 2)); % becomes [T x 1 x (98 + 2)]
    end

    % Speed Prediction Branch
    layersSpeed = [
        sequenceInputLayer(100, 'Name', 'input', 'Normalization', 'zscore')   % 98 spikes + 2 (cos and sin of angle)
        lstmLayer(100, 'OutputMode', 'last', 'Name', 'lstm')
        dropoutLayer(0.3)
        fullyConnectedLayer(50, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(20, 'Name', 'fc_speed_1')
        reluLayer('Name', 'relu_speed')
        fullyConnectedLayer(1, 'Name', 'speed_output')
    ];

    % Direction Prediction Branch (cos and sin)
    layersDirection = [
        sequenceInputLayer(100, 'Name', 'input', 'Normalization', 'zscore')   % 98 spikes + 2 (cos and sin of angle)
        lstmLayer(100, 'OutputMode', 'last', 'Name', 'lstm')
        dropoutLayer(0.3)
        fullyConnectedLayer(50, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(20, 'Name', 'fc_speed_1')
        reluLayer('Name', 'relu_speed')
        fullyConnectedLayer(2, 'Name', 'direction_output')
    ];

    % Angle Prediction Branch (for angle as a scalar)
    layersAngle = [
        sequenceInputLayer(100, 'Name', 'input', 'Normalization', 'zscore')
        lstmLayer(100, 'OutputMode', 'last', 'Name', 'lstm')
        dropoutLayer(0.3)
        fullyConnectedLayer(50, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(20, 'Name', 'fc_angle_1')
        reluLayer('Name', 'relu_angle')
        fullyConnectedLayer(1, 'Name', 'angle_output')
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

    % Display sizes for debugging
    %disp(['Final X size: ', mat2str(size(X))]);
    %disp(['Final Y_speed size: ', mat2str(size(Y_speed))]);      % Expected: [1, numSamples]
    %disp(['Final Y_direction size: ', mat2str(size(Y_direction))]);  % Expected: [2, numSamples]
    %disp(['Final Y_angle size: ', mat2str(size(Y_angle))]);  % Expected: [1, numSamples]
    
    %disp('Y_speed class:');
    %disp(class(Y_speed));  % Should display 'double'
    %disp('Y_speed size:');
    %disp(size(Y_speed));   % Should be [1, numSamples]

    %disp('Y_direction class:');
    %disp(class(Y_direction));  % Should display 'double'
    %disp('Y_direction size:');
    %disp(size(Y_direction));   % Should be [1, numSamples]

    %disp('Y_angle class:');
    %disp(class(Y_angle));  % Should display 'double'
    %disp('Y_angle size:');
    %disp(size(Y_angle));   % Should be [1, numSamples]
    
    % Training
    lossFcn = 'mse';

    % Training for Speed
    modelParameters.netSpeed = trainnet(X, Y_speed, layersSpeed, lossFcn, options);
    
    % Training for Direction
    modelParameters.netDirection = trainnet(X, Y_direction, layersDirection, lossFcn, options);
    
    % Training for Angle
    modelParameters.netAngle = trainnet(X, Y_angle, layersAngle, lossFcn, options);

    % Save the trained model for future use
    % save('positionEstimatorTrained.mat', 'net');
end
