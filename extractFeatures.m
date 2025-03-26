function X_test = extractFeatures(test_data)
    spikes = test_data.spikes; % 98 x T
    X_test = spikes'; % (T x 98)
    
    fprintf('Extracted test features size: %s\n', mat2str(size(X_test)));
    
    % Visualise a sample spike train
    % figure; imagesc(X_test'); colorbar;
    % title('Test Spike Activity');
    % xlabel('Time'); ylabel('Neurons');
end