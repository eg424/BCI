% =================================================================================
% NOTE: Run before test_preprocessing, otherwise error.
% Selects the highest and lowest firing neurons from a random trial, plots 
% their raw spike train, normalises them using Z-score & Min-Max, and compares.
%
% Outputs:
%   - A figure with two subplots per neuron:
%     1. Raw spike train vs. binned spikes
%     2. Normalized spike rates (Z-score & Min-Max)
% =================================================================================

% Select random trial
trial_idx = randi([1, 100]); 

% Compute firing rates for all neurons in selected trial
firing_rates = sum(trial(trial_idx).spikes, 2) / size(trial(trial_idx).spikes, 2);

% Select lowest and highest firing neurons
[~, low_idx] = min(firing_rates);
[~, high_idx] = max(firing_rates);

% Extract spike data for selected neurons
spike_data_low = trial(trial_idx).spikes(low_idx, :);
spike_data_high = trial(trial_idx).spikes(high_idx, :);

% Bin spikes for both neurons
[binnedLowBefore, binnedLow, timeBinsLow, ~, ~, binSizeLow] = binSpikes(spike_data_low);
[binnedHighBefore, binnedHigh, timeBinsHigh, ~, ~, binSizeHigh] = binSpikes(spike_data_high);

% Display bin size info
disp(['Low Firing Neuron: ' num2str(low_idx) ', Bin Size: ' num2str(binSizeLow) ' ms']);
disp(['High Firing Neuron: ' num2str(high_idx) ', Bin Size: ' num2str(binSizeHigh) ' ms']);

% Normalisation
normalizeSpikes = @(x) [(x - mean(x)) / (std(x) + eps);  % Z-score
                        (x - min(x)) / (max(x) - min(x) + eps)];  % Min-Max

normalized_low = normalizeSpikes(spike_data_low);
normalized_high = normalizeSpikes(spike_data_high);

% Plots
figure;

% LFN
subplot(2,2,1);
plot(find(spike_data_low), ones(size(find(spike_data_low))), 'k.');
xlabel('Time (ms)');
ylabel('Spikes');
title(['Low Firing Neuron (' num2str(low_idx) ') - Raw']);

subplot(2,2,2);
plot(normalized_low(1, :), 'b'); hold on;
plot(normalized_low(2, :), 'r');
xlabel('Time (ms)');
ylabel('Normalized Spikes');
title('Low Firing Neuron - Normalized');
legend('Z-Score', 'Min-Max');

% HFN
subplot(2,2,3);
plot(find(spike_data_high), ones(size(find(spike_data_high))), 'k.');
xlabel('Time (ms)');
ylabel('Spikes');
title(['High Firing Neuron (' num2str(high_idx) ') - Raw']);

subplot(2,2,4);
plot(normalized_high(1, :), 'b'); hold on;
plot(normalized_high(2, :), 'r');
xlabel('Time (ms)');
ylabel('Normalized Spikes');
title('High Firing Neuron - Normalized');
legend('Z-Score', 'Min-Max');
