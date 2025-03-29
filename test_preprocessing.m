% ====================================================================
% NOTE: DO NOT run before test_binsize or test_Norm, otherwise error. To solve,
% clear all, load training data and run test_binsize or test_Norm first.
% ====================================================================

% Mock parameters
numTrials = 5;              
numDirections = 8;          
numNeurons = 98;         
numTimeSteps = 500;         
binSize = 50;              

training_data = struct();

for trial = 1:numTrials
    for dir = 1:numDirections
        spikes = rand(numNeurons, numTimeSteps) > 0.95;
        handPos = rand(3, numTimeSteps);
        
        training_data(trial, dir).spikes = spikes;
        training_data(trial, dir).handPos = handPos;
    end
end

[X, Y_speed, Y_direction, Y_angle] = preprocessData(training_data, binSize);

disp('Sample outputs:');

disp('Size of binned spike data (X):');
disp(size(X));

disp('Size of speed labels (Y_speed):');
disp(length(Y_speed));

disp('Size of direction labels (Y_direction):');
disp(length(Y_direction));

disp('Size of angle labels (Y_angle):');
disp(length(Y_angle));

disp('Example binned spike data (X) for the first sample:');
disp(X{1}(1:5, :));  % First few neurons

disp('Example speed label (Y_speed) for the first sample:');
disp(Y_speed{1});

disp('Example direction label (Y_direction) for the first sample:');
disp(Y_direction{1});

disp('Example angle label (Y_angle) for the first sample:');
disp(Y_angle{1});
