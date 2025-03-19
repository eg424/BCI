function [x, y, newModelParameters] = positionEstimator(testData, modelParameters)
    % testData is a structure representing the current trial's data.
    % It contains fields: trialId, spikes, decodedHandPos, and startHandPos.
    
    % Get the spikes for the current trial (spikes is a [neurons x time] matrix)
    spikes = testData.spikes;
    
    % Apply PCA transformation to reduce dimensionality
    % (spikes' is [time x neurons], so reduced_spikes becomes [time x n_components])
    reduced_spikes = spikes' * modelParameters.pca_coeff;
    
    % Use the LDA model to predict the reaching angle for each time step,
    % then take the prediction at the current (latest) time step.
    predicted_angles = predict(modelParameters.lda_model, reduced_spikes);
    current_angle = predicted_angles(end);  % Use the most recent prediction
    
    % Map the discrete predicted angle (1-8) to a continuous angle (in radians).
    % Assuming: 1 -> 0°, 2 -> 45°, 3 -> 90°, ... 8 -> 315°.
    theta = (current_angle - 1) * (2*pi/8);
    
    % Determine the current position:
    % If no decoded positions exist, use the starting hand position.
    if isempty(testData.decodedHandPos)
        currentPos = testData.startHandPos;
    else
        currentPos = testData.decodedHandPos(:, end);
    end
    
    % Define a step size (you can incorporate features of the neural activity here)
    % You can set modelParameters.step_size during training or choose a constant.
    if isfield(modelParameters, 'step_size')
        step = modelParameters.step_size;
    else
        step = 10;  % Default step size; adjust as needed.
    end
    
    % Update the hand position by taking a step in the direction of theta.
    newPos = currentPos + step * [cos(theta); sin(theta)];
    
    % Return the new predicted hand position.
    x = newPos(1);
    y = newPos(2);
    
    % Optionally, update the model parameters if desired (here, we leave them unchanged).
    newModelParameters = modelParameters;
end
