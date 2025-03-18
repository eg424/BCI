function modelParameters = positionEstimatorTraining(trainingData)
    
    % Extract number of trials and neurons
    numTrials = length(trainingData);
    numNeurons = size(trainingData(1,1).spikes, 1);
    numAngles = size(trainingData, 2);
    
    % Load Naive Bayes classifier
    NBModel = Naive_Bayes(trainingData); 
    
    % Initialize struct to store angle-specific Kalman parameters
    modelParameters = struct();
    
    % Define bin size for feature extraction
    binSize = 390;

    for angle = 1:numAngles
        X = [];  % State matrix (position, velocity)
        Z = [];  % Observation matrix (spike rates)
        trainingVelocities = [];

        for trial = 1:numTrials

            spikes = trainingData(trial, angle).spikes; % (numNeurons x timeBins)
            handPos = trainingData(trial, angle).handPos; % (3 x timeBins)
            
            % Compute velocities using cumulative sum smoothing
            velocities = [];
            for dim = 1:2  
                cumsumPos = cumsum(handPos(dim, :), 2);
                velocity = (cumsumPos(:, binSize+1:end) - cumsumPos(:, 1:end-binSize)) / binSize;
                velocity = [zeros(1, binSize), velocity]; 
                velocities = [velocities; velocity]; 
            end
            trainingVelocities = [trainingVelocities, velocities];

            % Compute acceleration
            %acceleration = diff(velocities, 1, 2);
            %acceleration = [zeros(2, 1), acceleration];

            % Compute firing rates
            cumsumSpikes = cumsum(spikes, 2);
            firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
            firingRates = [zeros(numNeurons, binSize-1), firingRates];

            % Compute firing rate changes (first derivative)
            firingRateChanges = diff(firingRates, 1, 2);
            firingRateChanges = [zeros(numNeurons, 1), firingRateChanges];

            % Compute ISI features for each neuron
            ISI_mean = zeros(1, numNeurons);
            ISI_var = zeros(1, numNeurons);

            for neuron = 1:numNeurons
                spikeIndices = find(spikes(neuron, :) > 0); % Find spike times
                if length(spikeIndices) > 1
                    ISI = diff(spikeIndices); % Compute inter-spike intervals
                    ISI_mean(neuron) = mean(ISI);
                    ISI_var(neuron) = var(ISI);
                else
                    ISI_mean(neuron) = 0; % If no spikes or only one, ISI set to 0
                    ISI_var(neuron) = 0;
                end
            end

            % Compute mean values for each feature
            %meanVelocity = mean(velocities, 2); 
            meanFiringRate = mean(firingRates, 2);
            %meanAcceleration = mean(acceleration, 2);
            %meanFiringRateChange = mean(firingRateChanges, 2);
            meanISI = mean(ISI_mean);
            varISI = mean(ISI_var); 

            % Store feature matrices
            for t = 1:size(handPos,2)-1
                X = [X, [handPos(1,t); handPos(2,t); velocities(:,t)]];
                Z = [Z, [firingRates(:,t); meanFiringRate; meanISI; varISI]];
            end
        end

        % Estimate Kalman filter parameters
        A = (X(:,2:end) * X(:,1:end-1)') / (X(:,1:end-1) * X(:,1:end-1)');
        A = A * 0.3;
        W = cov(X(:,2:end)' - (A * X(:,1:end-1))');
        W = W + 1e-3 * eye(size(W));
    
        % Compute observation matrix H with regularization
        lambda = 1e-3; % Regularization parameter
        H = (Z * X') / (X * X' + lambda * eye(size(X,1)));
    
        % Compute observation noise covariance matrix
        Q = cov(Z' - (H * X)');
        Q = Q * 0.7;
    
        % Store model parameters
        modelParameters(angle).A = A;
        modelParameters(angle).W = W;
        modelParameters(angle).H = H;
        modelParameters(angle).Q = Q;
        modelParameters(angle).avgVelocity = mean(trainingVelocities, 2);
    end

    % Store the classifier
    for i = 1:numAngles
        modelParameters(i).classifier = NBModel;
    end
end