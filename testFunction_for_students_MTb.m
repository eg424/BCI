% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

function RMSE = testFunction_for_students_MTb(teamName)

load monkeydata_training.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

% addpath('Banana-Certified Interfaces');

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...\n')

meanSqError = 0;
n_predictions = 0;  
totalTime = 0;

figure
hold on
axis square
grid

% modelParameters = positionEstimatorTraining(trainingData);

% Instead of training every time, check if the trained model exists.
% Can comment this out and uncomment above if want to test new model.
if exist('positionEstimatorTrained.mat', 'file')
    load('positionEstimatorTrained.mat', 'modelParameters');
else
    modelParameters = positionEstimatorTraining(trainingData);
end

tic;
for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];
        decodedAngles = [];  % Store predicted angles

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;
            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, predictedAngle] = positionEstimator(past_current_trial, modelParameters);
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end

            
            % Append decoded position and angle
            decodedPos = [decodedPosX; decodedPosY];
            decodedAngles = [decodedAngles predictedAngle];  % Store predicted angle
            decodedHandPos = [decodedHandPos decodedPos];
            actualPos = testData(tr,direc).handPos(1:2,t);
            actualAngle = atan2(actualPos(2) - past_current_trial.startHandPos(2), ...
                                actualPos(1) - past_current_trial.startHandPos(1));  % Compute actual angle

            % Use the final predicted angle for error computation and printing:
            finalPredAngle = predictedAngle(end);

            % Print the time, predicted and actual positions and angles
            %fprintf('Time: %d | Decoded: (%.2f, %.2f) | Actual: (%.2f, %.2f) | Predicted Angle: %.2f | Actual Angle: %.2f\n', ...
                %t, decodedPosX(end), decodedPosY(end), actualPos(1), actualPos(2), finalPredAngle, actualAngle);

            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            
        end
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b');
    end
end

elapsedTime = toc;
totalTime = totalTime + elapsedTime;

legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions) 
fprintf('Total time taken: %.4f seconds\n', totalTime);
fprintf('Average time per prediction: %.4f seconds\n', totalTime / n_predictions);
Weighted_rank = 0.9 * RMSE + 0.1 * totalTime

% rmpath(genpath('Banana-Certified Interfaces'))

end