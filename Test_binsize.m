% =================================================================================
% NOTE: Run before test_preprocessing, otherwise error.
%
% Selects a random trial and neuron, plots raw spike train, bins the spikes
% (binSpikes) and displays the spike counts and empty bin stats
%
% Also calculates the firing rate of the selected neuron and selects the 
% bin size based on that rate
%
% Outputs:
%   - A figure with two subplots:
%     1. Raw spike train
%     2. Binned spike counts after filtering
%   - Display of the number of empty bins before and after filtering in cmd window
% ================================================================================

% Select random neuron and trial
neuron_idx = randi([1, 98]); 
trial_idx = randi([1, 100]); 

% Extract spike data
spike_data = trial(trial_idx).spikes(neuron_idx, :);  
totalTime = length(spike_data);

% Find spike times
spike_times = find(spike_data);

% Plot raw spike train
figure;
subplot(2,1,1);
plot(spike_times, ones(size(spike_times)), 'k.');
xlabel('Time (ms)');
ylabel('Spikes');
title(['Raw Spike Train (Neuron ' num2str(neuron_idx) ', Trial ' num2str(trial_idx) ')']);

% Bin spikes
[binnedSpikesBefore, binnedSpikes, timeBins, emptyBinsBefore, emptyBinsAfter, binSize] = binSpikes(spike_data);

% Display empty bins count before and after filtering
disp(['Number of empty bins before filtering: ' num2str(emptyBinsBefore)]);
disp(['Number of empty bins after filtering: ' num2str(emptyBinsAfter)]);

% Plot binned spike counts
subplot(2,1,2);
bar(timeBins(1:end-1), binnedSpikesBefore, 'k');
xlabel('Time (ms)');
ylabel('Spike Count');
title(['Binned Spikes (BinSize = ' num2str(binSize) ' ms)']);
