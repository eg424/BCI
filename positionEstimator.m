function [x, y] = positionEstimator(test_data, modelParameters)
    X_test = extractFeatures(test_data);
    
    fprintf('Test data size: %s\n', mat2str(size(X_test)));
    
    predictedPos = predict(modelParameters.net, X_test);
    
    if isempty(predictedPos)
        error('Prediction is empty. Check network output.');
    end
    
    fprintf('Predicted position size: %s\n', mat2str(size(predictedPos)));
    fprintf('First 5 predictions: \n');
    disp(predictedPos(1:min(5, size(predictedPos,1)), :));
    
    x = predictedPos(end, 1);
    y = predictedPos(end, 2);
    
    fprintf('Final predicted position: (%.2f, %.2f)\n', x, y);
end