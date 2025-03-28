function [X, Y_speed, Y_direction, Y_angle] = preprocessData(training_data, binSize)
    % =============================================================================
    %   Preprocesses spike data and hand movement trajectories to generate 
    %   training samples for neural decoding. Speed, direction, and angle 
    %   are computed separately and stored in different output variables.
    %
    % Inputs:
    %   - training_data: Structured array with neural spikes and hand positions.
    %   - binSize: Integer, number of time steps per bin.
    %
    % Outputs:
    %   - X: Binned spike activity (cell array, time x neurons).
    %   - Y_speed: Speed (cell array, scalar per bin).
    %   - Y_direction: Movement direction (cell array, 1x2 unit vector per bin).
    %   - Y_angle: Angle of movement (cell array, scalar angle in radians per bin).
    % =============================================================================

    numTrials = size(training_data, 1);
    numDirections = size(training_data, 2);
    
    X = {};
    Y_speed = {};
    Y_direction = {};
    Y_angle = {};

    % Loop through all trials and angles
    for trial = 1:numTrials
        for dir = 1:numDirections
            % Check if 'handPos' exists in trial and direction (debugging)
            if ~isfield(training_data(trial, dir), 'handPos')
                warning('handPos is missing for trial %d, direction %d', trial, dir);
                continue;  % Skip if field is missing
            end

            % Extract spikes and hand position data
            spikes = training_data(trial, dir).spikes;  % (neurons x time)
            handPos = training_data(trial, dir).handPos;  % (3 x time)

            % Check handPos dimensions (debugging asw)
            if size(handPos, 1) < 2
                warning('handPos for trial %d, direction %d has insufficient dimensions', trial, dir);
                continue;
            end

            % Remove non-firing neurons - May be useful for a different
            % model, but cannot be used for input layer due to fixed size.
            % nonFiringNeurons = all(spikes == 0, 2);  % Find neurons with zero spikes, all time steps
            % spikes(nonFiringNeurons, :) = [];  % Remove them

            % Velocity components
            dx = diff(handPos(1, :));
            dy = diff(handPos(2, :));

            % Speed
            speed = sqrt(dx.^2 + dy.^2);  % (1 x (time-1))

            % Direction (unit vector)
            direction = [dx; dy] ./ max(eps, vecnorm([dx; dy], 2, 1));  % (2 x (time-1))

            % Angle (rad)
            angle = atan2(dy, dx);  % (1 x (time-1))

            % Ensure spikes stored as [time x neurons]
            X{end+1} = spikes';

            % Create binned sequences
            numBins = floor(size(spikes, 2) / binSize);
            for b = 1:numBins
                % Bin spikes
                binSpikes = sum(spikes(:, (b-1)*binSize+1 : b*binSize), 2);

                % Bin speed, direction, and angle using the last time step in the bin
                idx = b * binSize;
                if idx > length(speed)
                    continue;
                end

                Y_speed{end+1} = speed(idx);  % Scalar speed
                Y_direction{end+1} = direction(:, idx)';  % 1x2 direction vector
                Y_angle{end+1} = angle(idx);  % Scalar angle in radians
            end
        end
    end
    
    % Debugging output
    %fprintf('Processed %d samples\n', length(X));
    %fprintf('Processed speed labels: %d samples\n', length(Y_speed));
    %fprintf('Processed direction labels: %d samples\n', length(Y_direction));
    %fprintf('Processed angle labels: %d samples\n', length(Y_angle));
end
