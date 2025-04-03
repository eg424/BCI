function RMSE = test_Prediction(teamName)
%==========================================================================
% Tests model performance on 4 random trials and angles.
% 
% Notes:
%   - Already uses pre-trained model (`positionEstimatorTrained.mat`) if 
%   available. If it doesn't exist, it trains a new one (Need to delete 
%   previous trained model file 'positionEstimatorTrained.mat' to test a new one).
%   - Selects 4 random trials and directions for testing. One direction is 
%   always set to 1 because I wanted to compare vs other angles when I was
%   looping for trials only.
%==========================================================================

    load monkeydata_training.mat

    % Set random number generator
    rng('shuffle'); % Changed to select random trial
    ix = randperm(length(trial));

    % Select training and testing data
    trainingData = trial(ix(1:50),:);
    testData = trial(ix(51:end),:);

    fprintf('Testing the continuous position estimator...\n')

    meanSqError = 0;
    n_predictions = 0;  
    totalTime = 0;
    
    % Initialize RMSE storage for each trial
    trialRMSE = [];

    % modelParameters = positionEstimatorTraining(trainingData);

    % Instead of training every time, check if the trained model exists.
    % Can comment this out (Ctrl + R) and uncomment above if test new model.
    if exist('positionEstimatorTrained.mat', 'file')
        load('positionEstimatorTrained.mat', 'modelParameters');
    else
        modelParameters = positionEstimatorTraining(trainingData);
    end
    
    tic;
     
    % Select random trials and directions for testing
    selectedTrials = randi([1, size(testData, 1)], 1, 4);   % Random trial indices
    selectedDirections = [1, randi([1, size(testData, 2)], 1, 3)]; % One direction is always 1, other 3 are random
    % selectedTrials = [50 15 32 39];
    % selectedDirections = [5 1 4 2];

    % Shuffle to randomise order
    selectedDirections = selectedDirections(randperm(length(selectedDirections)));

    figure;
    for subplotIdx = 1:4
        trialIdx = selectedTrials(subplotIdx);
        directionIdx = selectedDirections(subplotIdx);
        
        % Initialise for storing predictions
        decodedHandPos = [];
        decodedAngles = [];

        % Test selected trial and direction
        times = 320:20:size(testData(trialIdx, directionIdx).spikes, 2);

        trialMeanSqError = 0;

        for t = times
            past_current_trial.trialId = testData(trialIdx, directionIdx).trialId;
            past_current_trial.spikes = testData(trialIdx, directionIdx).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;
            past_current_trial.startHandPos = testData(trialIdx, directionIdx).handPos(1:2, 1); 
            
            % Call the positionEstimator function
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, predictedAngle] = positionEstimator(past_current_trial, modelParameters);
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end

            % Append decoded position and angle
            decodedPos = [decodedPosX; decodedPosY];
            decodedAngles = [decodedAngles predictedAngle];
            decodedHandPos = [decodedHandPos decodedPos];

            % Get the actual position and calculate the actual angle
            actualPos = testData(trialIdx, directionIdx).handPos(1:2, t);
            actualAngle = atan2(actualPos(2) - past_current_trial.startHandPos(2), ...
                                actualPos(1) - past_current_trial.startHandPos(1));

            % Print predicted vs actual at each time step for comparison
            %fprintf('Time: %d | Decoded: (%.2f, %.2f) | Actual: (%.2f, %.2f) | Predicted Angle: %.2f | Actual Angle: %.2f\n', ...
                    %t, decodedPosX, decodedPosY, actualPos(1), actualPos(2), predictedAngle, actualAngle);

            % Compute Mean Squared Error
            trialMeanSqError = trialMeanSqError + norm(actualPos - decodedPos)^2;
        end
        n_predictions = n_predictions+length(times);
        meanSqError = meanSqError + trialMeanSqError;

        % Plot decoded and actual positions for this direction
        subplot(2, 2, subplotIdx);
        plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r');
        hold on;
        plot(testData(trialIdx, directionIdx).handPos(1, times), ...
             testData(trialIdx, directionIdx).handPos(2, times), 'b');
        title(['Trial ', num2str(trialIdx), ' Direction ', num2str(directionIdx)]);
        xlabel('X Position');
        ylabel('Y Position');
        legend('Decoded Position', 'Actual Position');
        axis equal;
        grid on;

        % Store RMSE for current trial
        rmse_trial = sqrt(trialMeanSqError / length(times));
        trialRMSE = [trialRMSE rmse_trial];

        fprintf('Selected Trial: %d, Direction: %d, RMSE: %.4f\n', trialIdx, directionIdx, rmse_trial);
    end

    elapsedTime = toc;
    totalTime = totalTime + elapsedTime;

    % Plot RMSE for each trial
    % figure;
    % bar(trialRMSE);
    % title('RMSE for Each Trial');
    % xlabel('Trial');
    % ylabel('RMSE');
    % grid on;

    RMSE = sqrt(meanSqError/n_predictions) 
    fprintf('Total time taken: %.4f seconds\n', totalTime);
    fprintf('Average time per prediction: %.4f seconds\n', totalTime / n_predictions);
    Weighted_rank = 0.9 * RMSE + 0.1 * totalTime

end