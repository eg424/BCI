function [speed, direction] = calculateSpeedAndDirection(handPos)
    % Calculates speed and direction from hand position.
    %
    % Args:
    %     handPos: Hand position data (2 x timePoints).
    %
    % Returns:
    %     speed: Speed of movement (timePoints x 1).
    %     direction: Direction of movement (timePoints x 1) in radians.

    
    dx = diff(handPos(1, :));
    dy = diff(handPos(2, :));
    
    speed = sqrt(dx.^2 + dy.^2);
    speed = [0, speed]; % Add a 0 at the beginning to match handPos length
    
    direction = atan2(dy, dx);
    direction = [0, direction]; % Add a 0 at the beginning
end