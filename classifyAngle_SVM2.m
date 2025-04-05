function predictedAngle = classifyAngle_SVM2(spikes, Model)
    numNeurons = size(spikes, 1);
    numFeatures = 6;
    featureVector = zeros(1, numNeurons * numFeatures);
    
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
    
    featureVector = [meanFiring, varFiring, fanoFactor, tot_spikes, ISI_mean, ISI_var];
    featureVector = (featureVector - mean(featureVector)) ./ (std(featureVector) + 1e-6);
    
    predictedAngle = predict(Model.svm, featureVector);
end