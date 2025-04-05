function modelParameters = positionEstimatorTraining2(trainingData)
    % Extract number of trials and neurons
    numTrials = length(trainingData);
    numNeurons = size(trainingData(1,1).spikes, 1);

    SVMModel = SVM_Classifier2(trainingData);
    
    % Define matrices for Kalman filter
    X = [];
    Z = [];
    trainingVelocities = [];
    
    binSize = 390;
    for trial = 1:numTrials
        for angle = 1:size(trainingData, 2)
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

            % Predict angle using SVM Classifier
            angle_predicted = classifyAngle_SVM2(spikes, SVMModel);

            for t = 1:size(handPos,2)-1
                X = [X, [handPos(1,t); handPos(2,t); velocities(:,t)]]; 
                Z = [Z, [firingRates(:,t); firingRateChanges(:,t); meanFiringRate; angle_predicted]];
            end
        end
    end
    
    % Estimate Kalman parameters
    A = (X(:,2:end) * X(:,1:end-1)') / (X(:,1:end-1) * X(:,1:end-1)');
    A = A * 0.3;
    W = cov(X(:,2:end)' - (A * X(:,1:end-1))');
    W = W + 1e-3 * eye(size(W));
    lambda = 1e-3;
    H = (Z * X') / (X * X' + lambda * eye(size(X,1)));
    Q = cov(Z' - (H * X)');
    Q = Q * 0.7; 
 
    % Store parameters
    modelParameters.A = A;
    modelParameters.W = W;
    modelParameters.H = H;
    modelParameters.Q = Q;
    modelParameters.avgVelocity = mean(trainingVelocities, 2);
    modelParameters.classifier = SVMModel;
end