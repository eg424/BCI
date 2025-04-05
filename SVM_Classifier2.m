function Model = SVM_Classifier2(trainingData)
    % Extract the number of trials, angles, and neurons
    numTrials = size(trainingData, 1);
    numAngles = size(trainingData, 2);
    numNeurons = size(trainingData(1,1).spikes, 1);
    numFeatures = 6; % Mean, Variance, Fano Factor, Total Spike Count, ISI Mean, ISI Variance
    
    features = zeros(numTrials * numAngles, numNeurons * numFeatures);
    angles = zeros(numTrials * numAngles, 1);
    
    sampleIdx = 1;
    
    for trial = 1:numTrials
        for angle = 1:numAngles
            spikes = trainingData(trial, angle).spikes;
            
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
            features(sampleIdx, :) = featureVector;
            angles(sampleIdx) = angle;
            sampleIdx = sampleIdx + 1;
        end
    end
    
    % Mornalise features
    features = (features - mean(features)) ./ (std(features) + 1e-6);
    
    % Train SVM model
    Model.svm = fitcecoc(features, angles, 'Coding', 'onevsall', 'Learners', 'svm');
end