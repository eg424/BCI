function modelParameters = positionEstimatorTraining(trainingData)
    % Trains per-angle Kalman filters and a shared SVM classifier with velocity map
    numTrials = size(trainingData, 1);
    numAngles = size(trainingData, 2);
    numNeurons = size(trainingData(1,1).spikes, 1);
    binSize = 395;

    angle_models = struct();
    allVelocities = [];
    allFiringRates = [];

    for angle = 1:numAngles
        X = [];
        Z = [];
        trainingVelocities = [];

        for trial = 1:numTrials
            spikes = trainingData(trial, angle).spikes;
            handPos = trainingData(trial, angle).handPos;

            velocities = [];
            for dim = 1:2
                cumsumPos = cumsum(handPos(dim, :), 2);
                velocity = (cumsumPos(:, binSize+1:end) - cumsumPos(:, 1:end-binSize)) / binSize;
                velocity = [zeros(1, binSize), velocity];
                velocities = [velocities; velocity];
            end
            trainingVelocities = [trainingVelocities, velocities];

            cumsumSpikes = cumsum(spikes, 2);
            firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
            firingRates = [zeros(numNeurons, binSize-1), firingRates];

            firingRateChanges = diff(firingRates, 1, 2);
            firingRateChanges = [zeros(numNeurons, 1), firingRateChanges];

            meanFiringRate = mean(firingRates, 2);

            for t = 1:size(handPos,2)-1
                X = [X, [handPos(1,t); handPos(2,t); velocities(:,t)]];
                Z = [Z, [firingRates(:,t); firingRateChanges(:,t); meanFiringRate]];
            end

            allVelocities = [allVelocities, velocities];
            allFiringRates = [allFiringRates, firingRates];
        end

        A = (X(:,2:end) * X(:,1:end-1)') / (X(:,1:end-1) * X(:,1:end-1)');
        A = A * 0.3;
        W = cov(X(:,2:end)' - (A * X(:,1:end-1))');
        W = W + 1e-3 * eye(size(W));
        W = W * 0.5;
        lambda = 1e-3;
        H = (Z * X') / (X * X' + lambda * eye(size(X,1)));
        Q = cov(Z' - (H * X)');
        Q = Q * 0.08;

        angle_models(angle).A = A;
        angle_models(angle).W = W;
        angle_models(angle).H = H;
        angle_models(angle).Q = Q;
    end

    % Fit linear velocity model for initial velocity estimation
    B = allVelocities * pinv(allFiringRates);

    % Train updated SVM classifier with progressive rounds
    modelParameters.angle_models = angle_models;
    modelParameters.classifier = SVM_Classifier(trainingData);
    modelParameters.velocityMap = B;

end



