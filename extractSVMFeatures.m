function featureVector = extractSVMFeatures(spikes, round)

    if nargin < 2
        round = 1;  % Default to round 1 (300 ms)
    end

    numNeurons = size(spikes, 1);

    % ISI features
    ISI_mean = zeros(1, numNeurons);
    ISI_var = zeros(1, numNeurons);
    for neuron = 1:numNeurons
        spikeIndices = find(spikes(neuron, :) > 0);
        if length(spikeIndices) > 1
            ISI = diff(spikeIndices);
            ISI_mean(neuron) = mean(ISI);
            ISI_var(neuron) = var(ISI);
        end
    end

    meanFiring = mean(spikes, 2)';
    varFiring = var(spikes, 0, 2)';
    fanoFactor = varFiring ./ (meanFiring + 1e-6);
    tot_spikes = sum(spikes, 2)';



    % Temporal windows
    if round == 1
        windows = [1 100; 101 200; 201 300];
    elseif round == 2
        windows = [1 100; 101 200; 201 300; 301 320];
    elseif round == 3
        windows = [1 100; 101 200; 201 300; 301 340];
    elseif round == 4
        windows = [1 100; 101 200; 201 300; 301 360];
    elseif round == 5
        windows = [1 100; 101 200; 201 300; 301 380];
    end

    temporalMeans = [];
    for w = 1:size(windows, 1)
        wStart = windows(w, 1);
        wEnd = min(windows(w, 2), size(spikes, 2));
        meanWindow = mean(spikes(:, wStart:wEnd), 2)';
        temporalMeans = [temporalMeans, meanWindow];
    end

    featureVector = [meanFiring, varFiring, fanoFactor, tot_spikes, ISI_mean, ISI_var, temporalMeans];


end

