%% Velocity Kalman Filter (vKF) Implementation
% This script estimates movement velocity from neural activity using Kalman Filter

clc; clear; close all;

%% Simulated Data (Replace with actual neural data)
timesteps = 100;  % Number of time steps
neurons = 158;    % Number of neurons

% Simulate neural firing rates (random data for now)
neural_data = randn(timesteps, neurons);

% Simulate true velocity (ground truth, replace with real data)
true_velocity = randn(timesteps, 2); % [vx, vy]

%% Kalman Filter Initialization
A = eye(2);  % State transition matrix (Assume constant velocity model)
H = randn(neurons, 2); % Observation matrix (Random for now)
Q = 0.01 * eye(2);  % Process noise covariance
R = 0.1 * eye(neurons); % Measurement noise covariance
P = eye(2);  % Initial estimate covariance

x_est = zeros(timesteps, 2); % Store estimated velocities
x = [0; 0];  % Initial velocity estimate

%% Kalman Filtering Process
for t = 1:timesteps
    % Prediction Step
    x_pred = A * x;
    P_pred = A * P * A' + Q;
    
    % Measurement Update
    K = P_pred * H' / (H * P_pred * H' + R); % Kalman Gain
    x = x_pred + K * (neural_data(t, :)' - H * x_pred);
    P = (eye(2) - K * H) * P_pred;
    
    % Store estimated velocity
    x_est(t, :) = x';
end

%% Plot Results
figure;
plot(true_velocity(:,1), true_velocity(:,2), 'b', 'LineWidth', 2); hold on;
plot(x_est(:,1), x_est(:,2), 'r--', 'LineWidth', 2);
legend('True Velocity', 'Estimated Velocity (Kalman)');
xlabel('Vx'); ylabel('Vy'); title('Kalman Filter Velocity Estimation');
grid on;


%% LSTM-based Speed-Direction Estimation
% Requires MATLAB Deep Learning Toolbox

% Define LSTM Network Architecture
layers = [ ...
    sequenceInputLayer(neurons)
    lstmLayer(50, 'OutputMode', 'sequence')
    fullyConnectedLayer(2)  % Output: speed and direction
    regressionLayer];

% Prepare Data for Training (Neural Data -> Velocity)
XTrain = num2cell(neural_data', [1 2]); % Cell array for LSTM input
YTrain = num2cell(true_velocity', [1 2]);

% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 10, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress');

% Train the LSTM Model
net = trainbet(XTrain, YTrain, layers, options);

% Predict using trained LSTM
YPred = predict(net, XTrain);

% Plot Results
figure;
plot(true_velocity(:,1), true_velocity(:,2), 'b', 'LineWidth', 2); hold on;
YPredMat = cell2mat(YPred);  % Convert to matrix
plot(YPredMat(:,1), YPredMat(:,2), 'g--', 'LineWidth', 2);
legend('True Velocity', 'Estimated Velocity (LSTM)');
title('LSTM Velocity Estimation');
grid on;