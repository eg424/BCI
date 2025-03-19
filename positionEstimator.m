function [x, y] = positionEstimator(testData, modelParameters)
    % Load model parameters
    A = modelParameters.A;
    W = modelParameters.W;
    H = modelParameters.H;
    Q = modelParameters.Q;
    avgVelocity = modelParameters.avgVelocity;

    % Hidden state dimension
    numHiddenStates = 20;  % Assuming H maps to [firingRates, firingRateChanges]
    
    % Initial state estimation
    spikes = testData.spikes;
    numNeurons = size(spikes, 1);
    binSize = 395;
    
    % Initial estimates
    x_est = [testData.startHandPos(1); testData.startHandPos(2); avgVelocity; 0; 0; zeros(numHiddenStates, 1)];
    P_est = eye(6 + numHiddenStates);
    
    % Compute smoothed firing rates
    cumsumSpikes = cumsum(spikes, 2);
    firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
    firingRates = [zeros(numNeurons, binSize-1), firingRates];
    %Z = firingRates(:, 1:T);
    firingRateChanges = diff(firingRates, 1, 2); % First derivative of firing rates
    firingRateChanges = [zeros(numNeurons, 1), firingRateChanges]; % Pad with zeros
    meanFiringRate = mean(firingRates, 2);
    % % Compute variance and mean of firing rates over time
    % spikeCountVariance = var(spikes, 0, 2);
    % spikeCountMean = mean(spikes, 2);
    % fanoFactor = spikeCountVariance ./ (spikeCountMean + 1e-6); 
    meanFiringRateChange = mean(firingRateChanges, 2);


    Z = [firingRates; firingRateChanges; repmat(meanFiringRate, 1, size(firingRates, 2))];  
   
    size_a = size(A)
    size_xest = size(x_est)
    % Batch Kalman filter update
    X_pred = A * x_est;
    P_pred = A * P_est * A' + W;
    lambda = 1e-6;  % regularisation constant
    S = H * P_pred * H' + Q + lambda * eye(size(Q));
    K = (P_pred * H') / S;
    X_est = X_pred + K * (Z - H * X_pred);
    P_est = P_pred - K * H * P_pred;

    % Output estimated position
    x = X_est(1, end);
    y = X_est(2, end);
    
    % for t = 1:T:size(spikes, 2)
    %     % Prediction step
    %     X_pred = A * x_est;
    %     P_pred = A * P_est * A' + W;
    % 
    %     % Compute Kalman Gain
    %     lambda = 1e-6;  % Regularization constant
    %     S = H * P_pred * H' + Q + lambda * eye(size(Q));
    %     K = (P_pred * H') / S;
    % 
    %     % Update step
    %     x_est = X_pred + K * (Z(:, t) - H * X_pred);
    %     P_est = P_pred - K * H * P_pred;
    % end
    % 
    % % Output estimated position
    % x = x_est(1);
    % y = x_est(2);
end