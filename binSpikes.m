function [binnedSpikesBefore, binnedSpikes, timeBins, emptyBinsBefore, emptyBinsAfter, binSize] = binSpikes(spike_data)
    % ====================================================================
    % Takes spike data and bins the spikes into time intervals based 
    % on calculated bin size, which is determined by the firing rate. 
    % It returns the raw and filtered binned spike counts, time bins, 
    % and the count of empty bins before and after filtering (those with no spike
    % activity recorded).
    %
    % Inputs:
    %   - spike_data: A vector of spike events for a single neuron over time.
    %
    % Outputs:
    %   - binnedSpikesBefore: The binned spike counts before filtering empty bins.
    %   - binnedSpikes: The binned spike counts after filtering empty bins.
    %   - timeBins: The edges of the time bins used for binning.
    %   - emptyBinsBefore: The count of empty bins before filtering.
    %   - emptyBinsAfter: The count of empty bins after filtering.
    %   - binSize: The calculated bin size based on the firing rate.
    % ====================================================================

    % Get duration of trial (length of spike_data)
    totalTime = length(spike_data);
    
    % Compute firing rate
    firingRate = sum(spike_data) / totalTime * 1000;  % in Hz
    
    % Select bin size based on firing rate
    if firingRate > 30  
        binSize = 10; % Smaller bins for high activity
    elseif firingRate > 10  
        binSize = 20; % Medium bins for moderate activity
    else  
        binSize = 50; % Larger bins for low activity
    end

    % Create time bins
    timeBins = 0:binSize:(totalTime);
    
    % Binning spikes using histcounts
    binnedSpikes = histcounts(find(spike_data), timeBins);
    binnedSpikesBefore = histcounts(find(spike_data), timeBins); % Just for plotting
    
    % Count empty bins before filtering
    emptyBinsBefore = sum(binnedSpikes == 0);  % Bins with zero spikes
    
    % Filter out empty bins
    binnedSpikesFiltered = binnedSpikes(binnedSpikes > 0);
    
    % Count empty bins after filtering
    emptyBinsAfter = sum(binnedSpikesFiltered == 0);
    
    % Return binned spikes and empty bin counts
    binnedSpikes = binnedSpikesFiltered;  % Only filtered
end
