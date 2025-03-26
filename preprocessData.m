function [X, Y] = preprocessData(training_data)
    numTrials = size(training_data, 1);
    numDirections = size(training_data, 2);
    
    X = cell(numTrials * numDirections, 1);
    Y = cell(numTrials * numDirections, 1);
    
    idx = 1;
    for trial = 1:numTrials
        for dir = 1:numDirections
            if ~isfield(training_data(trial, dir), 'handPos')
                error('Error: Field "handPos" not found in training_data(%d, %d)', trial, dir);
            end
            
            spikes = training_data(trial, dir).spikes; 
            handPos = training_data(trial, dir).handPos;
            
            fprintf('Trial %d, Direction %d: spikes size = %s, handPos size = %s\n', ...
                trial, dir, mat2str(size(spikes)), mat2str(size(handPos)));

            if size(handPos, 2) < 600
                warning('handPos has fewer than 600 time points in trial %d, direction %d.', trial, dir);
            end

            X{idx} = spikes'; 
            Y{idx} = handPos(1:2, :)';
            
            idx = idx + 1;
        end
    end

    fprintf('Processed data size: %d sequences\n', length(X));
    fprintf('Processed labels size: %d sequences\n', length(Y));
end
