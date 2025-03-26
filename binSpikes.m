function binnedSpikes = binSpikes(spikes, binSize)
    numNeurons = size(spikes, 1);
    numBins = floor(size(spikes, 2) / binSize);
    binnedSpikes = zeros(numNeurons, numBins);
    
    for i = 1:numBins
        binnedSpikes(:, i) = sum(spikes(:, (i-1)*binSize+1 : i*binSize), 2);
    end
    
    % Debugging: Check binning results
    fprintf('Binned spike data size: %s\n', mat2str(size(binnedSpikes)));
    
    % Visualise spike binning
    % figure; imagesc(binnedSpikes); colorbar;
    % title('Binned Spike Activity');
    % xlabel('Time Bins'); ylabel('Neurons');
end