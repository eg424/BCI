function [x, y, modelParameters] = positionEstimator(testData, modelParameters)

    % SVM + per-angle Kalman filter decoder with progressive reclassification

    spikes = testData.spikes;
    numNeurons = size(spikes, 1);
    binSize = 395;
    classifier = modelParameters.classifier;

    % Progressive reclassification window lengths
    reclassWindows = [300, 320, 340, 360, 380];
    maxT = size(spikes, 2);

    % Try reclassifying progressively until we get consistent prediction
    for round = 1:length(reclassWindows)
        T = min(reclassWindows(round), maxT);
        spikeSegment = spikes(:, 1:T);
        predictedAngle = classifyAngle_SVM(spikeSegment, classifier);

        if round == length(reclassWindows)
            angle_predicted = predictedAngle;
        end
    end

    % Select angle-specific model
    model = modelParameters.angle_models(angle_predicted);
    A = model.A;
    W = model.W;
    H = model.H;
    Q = model.Q;

    % Estimate initial velocity using early spike activity
    earlyWindow = min(300, size(spikes, 2));
    cumsumSpikes = cumsum(spikes(:, 1:earlyWindow), 2);
    earlyFiringRates = cumsumSpikes(:, end) / earlyWindow;
    B = modelParameters.velocityMap;
    initVelocity = B * earlyFiringRates;

    % Initial state vector
    x_est = [testData.startHandPos(1)-20; testData.startHandPos(2)-4; initVelocity];
    P_est = eye(4);

    % Compute observation matrix Z
    cumsumSpikes = cumsum(spikes, 2);
    firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
    firingRates = [zeros(numNeurons, binSize-1), firingRates];
    firingRateChanges = diff(firingRates, 1, 2);
    firingRateChanges = [zeros(numNeurons, 1), firingRateChanges];
    meanFiringRate = mean(firingRates, 2);

    Z = [firingRates; firingRateChanges; repmat(meanFiringRate, 1, size(firingRates, 2))];

    % Apply Kalman filter
    X_pred = A * x_est;
    P_pred = A * P_est * A' + W;
    lambda = 1e-6;
    S = H * P_pred * H' + Q + lambda * eye(size(Q));
    K = (P_pred * H') / S;
    X_est = X_pred + K * (Z - H * X_pred);
    P_est = P_pred - K * H * P_pred;

    % Output estimated position
    x = X_est(1, end)-5;
    y = X_est(2, end);

end
