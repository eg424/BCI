function [x, y] = positionEstimator2(testData, modelParameters)
    % Load model parameters
    A = modelParameters.A;
    W = modelParameters.W;
    H = modelParameters.H;
    Q = modelParameters.Q;
    avgVelocity = modelParameters.avgVelocity;
    classifier = modelParameters.classifier;
    
    % Load testing data
    spikes = testData.spikes;
    numNeurons = size(spikes, 1);
    binSize = 395;
    
    % Initial estimates
    x_est = [testData.startHandPos(1); testData.startHandPos(2); avgVelocity];
    P_est = eye(4);
    
    % Compute spikes train's features for Z
    cumsumSpikes = cumsum(spikes, 2);
    firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
    firingRates = [zeros(numNeurons, binSize-1), firingRates];
    firingRateChanges = diff(firingRates, 1, 2);
    firingRateChanges = [zeros(numNeurons, 1), firingRateChanges];
    meanFiringRate = mean(firingRates, 2);

    % Predict the angle using SVM classifier
    angle_predicted = classifyAngle_SVM2(spikes, classifier);

    % Populate Z feature matrix
    Z = [firingRates; firingRateChanges; repmat(meanFiringRate, 1, size(firingRates, 2)); repmat(angle_predicted, 1, size(firingRates, 2))];  
   
    % Batch Kalman filter update
    X_pred = A * x_est;
    P_pred = A * P_est * A' + W;
    lambda = 1e-6; 
    S = H * P_pred * H' + Q + lambda * eye(size(Q));
    K = (P_pred * H') / S;
    X_est = X_pred + K * (Z - H * X_pred);
    P_est = P_pred - K * H * P_pred;

    % Output estimated position
    x = X_est(1, end);
    y = X_est(2, end);
end