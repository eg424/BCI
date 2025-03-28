function [x, y] = positionEstimator(test_data, modelParameters)
    % Extract features
    spikes = test_data.spikes; % 98 x T
    X_test = spikes'; % (T x 98)
    
    fprintf('Extracted test features size: %s\n', mat2str(size(X_test)));
        
    % Predict speed and direction separately
    predictedSpeed = predict(modelParameters.netSpeed, X_test);
    predictedDirection = predict(modelParameters.netDirection, X_test);

    % predictedPos = predict(modelParameters.net, X_test);

    % Testing
    % predictedSpeed = modelParameters.netSpeed(X_test);
    % predictedDirection = modelParameters.netDirection(X_test);
    
    % Ensure output shapes are correct
    predictedDirection = reshape(predictedDirection, 2, []); % Ensure it's [2 x N]
    
    % Convert direction to velocity
    disp("Size of predictedSpeed: ");
    disp(size(predictedSpeed));
    
    disp("Size of predictedDirection: ");
    disp(size(predictedDirection));

    velocity = predictedSpeed' .* predictedDirection; % Now velocity is [2 × 320]

    % Integrate velocity to estimate position
    estimatedPositions = cumsum(velocity, 2);  % Cumulative sum across time

    % Extract final estimated position
    x = estimatedPositions(1, end);
    y = estimatedPositions(2, end);

    fprintf('Predicted final position: x = %.2f, y = %.2f\n', x, y);
end
