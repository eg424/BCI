function [x, y, predictedAngle] = positionEstimator(test_data, modelParameters)
    % Extract features
    spikes = test_data.spikes; % 98 x T
    X_test = spikes'; % (T x 98)

    % If the model expects 100 neurons as input, pad the data to match
    if size(X_test, 2) < 100
        X_test = padarray(X_test, [0, 100 - size(X_test, 2)], 'post');
    elseif size(X_test, 2) > 100
        X_test = X_test(:, 1:100); % If the input size exceeds 100, truncate it
    end
    
    fprintf('Extracted test features size: %s\n', mat2str(size(X_test)));
        
    % Predict speed, angle and direction separately
    predictedSpeed = predict(modelParameters.netSpeed, X_test);
    predictedDirection = predict(modelParameters.netDirection, X_test);
    predictedAngle = predict(modelParameters.netAngle, X_test);

    % Testing
    % predictedSpeed = modelParameters.netSpeed(X_test);
    % predictedDirection = modelParameters.netDirection(X_test);
    
    % Ensure output shapes are correct
    % predictedDirection = reshape(predictedDirection, 2, []); % Ensure it's [2 x N]
    
    % Ensure predictedDirection has the correct shape [2 x N]
    if size(predictedDirection, 1) == 1
        predictedDirection = repmat(predictedDirection, 2, 1);  % Repeat the values in rows
    end

    %disp("Size of predictedSpeed: ");
    %disp(size(predictedSpeed));
    
    %disp("Size of predictedDirection: ");
    %disp(size(predictedDirection));

    %disp("Size of predictedAngle: ");
    %disp(size(predictedAngle));

    % Convert direction to velocity
    % velocity = predictedSpeed' .* predictedDirection; % Now velocity is [2 × 320]

    % Alternatively, if you want to use the angle directly:
    velocity = predictedSpeed .* [cos(predictedAngle), sin(predictedAngle)];
    velocity = velocity';

    % Integrate velocity to estimate position (cumulative sum over time)
    estimatedPositions = cumsum(velocity, 2);  

    % Extract final estimated position
    x = estimatedPositions(1, end);
    y = estimatedPositions(2, end);

    %fprintf('Predicted final position: x = %.2f, y = %.2f\n', x, y);
    
    % Optionally print predicted angle (in radians)
    %fprintf('Predicted final angle: %.2f radians\n', predictedAngle(end));
end
