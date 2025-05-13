load('monkeydata_training.mat');

%% RMSEs per trial (Retrieved after modifying testFunction)
% Kalman Filter
kalman_rmse = [19.6454, 26.1122, 24.3588, 21.3228, 26.3176, 21.0749, 26.6721, 23.2733, 23.4293, 24.5006, 29.5916, 21.2113, 20.5441, 25.4442, 24.6026, 24.1488, 33.8234, 18.5701, 19.9795, 27.6806, 25.0589, 21.4574, 23.8703, 18.0213, 24.1323, 17.8739, 30.0325, 24.4122, 20.9346, 23.3781, 24.1124, 31.4681, 26.4888, 26.2672, 21.4088, 22.6701, 23.1937, 23.7794, 26.7971, 24.4556, 25.1171, 26.6178, 22.5867, 31.4607, 25.7231, 26.8794, 28.8627, 24.6664, 25.8218, 19.3482];

% SVM + Angle-Specific KF
svm_k_rmse = [14.4366, 25.1855, 22.0898, 14.6304, 16.6497, 15.0177, 22.6779, 13.8377, 15.1102, 23.4099, 23.6466, 21.7100, 29.2814, 28.9432, 15.1574, 15.5639, 32.0359, 27.7317, 22.9720, 21.2929, 15.6934, 22.6115, 24.7330, 15.5100, 24.8571, 14.9415, 16.2388, 20.5045, 22.7957, 22.9834, 14.1804, 13.3977, 15.5319, 14.3596, 17.8595, 16.2694, 13.2400, 15.2985, 24.0217, 27.1620, 15.6704, 23.6785, 14.8782, 16.4037, 22.7975, 24.9390, 23.0846, 23.1613, 26.2870, 25.8974
];

% LSTM
lstm_rmse = [20.1945, 3.8637, 3.5713, 21.3502, 3.7331, 18.4989, 13.6514, 3.7470, 6.5516, 4.5025, 4.1311, 4.5232, 4.0126, 3.8820, 21.3251, 4.3393, 4.6640, 9.4467, 18.0078, 4.2375, 24.8621, 7.1923, 29.4184, 4.0108, 8.5742, 3.7543, 3.9847, 4.3197, 4.5089, 26.7932, 5.9158, 18.9028, 5.6231, 41.0013, 4.2788, 3.7807, 8.7545, 4.2744, 18.1151, 23.8724, 10.7651, 4.9857, 29.6184, 28.0709, 4.3687, 4.0544, 6.7119, 4.9483, 3.9682, 26.1886
];

%% Kolmogorov-Smirnov normality test
% Combine and transpose
rmse_data = [kalman_rmse; svm_k_rmse; lstm_rmse]';

[h_kf, p_kf] = kstest((kalman_rmse - mean(kalman_rmse)) / std(kalman_rmse));
[h_svm, p_svm] = kstest((svm_k_rmse - mean(svm_k_rmse)) / std(svm_k_rmse));
[h_lstm, p_lstm] = kstest((lstm_rmse - mean(lstm_rmse)) / std(lstm_rmse));

% fprintf('KF Normality p-value: %.4f\n', p_kf); % 0.7045 - Normal
% fprintf('SVM+KF Normality p-value: %.4f\n', p_svm); % 0.0352 - Not normal
% fprintf('LSTM Normality p-value: %.4f\n', p_lstm); % 0.0021 - Not normal

%% Friedman test (non-parametric repeated-measures ANOVA)
[p_friedman, tbl, stats] = friedman(rmse_data, 1, 'on'); % 4.49e-09 - Statistical Differences between RMSEs

%% Post-hoc pairwise comparisons
% Wilcoxon sign-rank (paired data - same participant)
[p_kf_vs_svm, ~, stats1] = signrank(rmse_data(:,1), rmse_data(:,2));
[p_kf_vs_lstm, ~, stats2] = signrank(rmse_data(:,1), rmse_data(:,3));
[p_svm_vs_lstm, ~, stats3] = signrank(rmse_data(:,2), rmse_data(:,3));

% Store results for sorting
results = [
    struct('p', p_kf_vs_svm,   'name', 'KF vs SVM+KF');
    struct('p', p_kf_vs_lstm,  'name', 'KF vs LSTM');
    struct('p', p_svm_vs_lstm, 'name', 'SVM+KF vs LSTM')
];

% Sort by p-value
[~, idx] = sort([results.p]);
results_sorted = results(idx);

% Bonferroni-Holm correction (increased power)
alpha = 0.05;
fprintf('Bonferroni-Holm corrected comparisons (alpha = %.2f):\n', alpha);
for i = 1:length(results_sorted)
    adjusted_alpha = alpha / (length(results_sorted) - i + 1);
    h = results_sorted(i).p < adjusted_alpha;
    fprintf('%s: p = %.4f | adjusted alpha = %.4f | %s\n', ...
        results_sorted(i).name, results_sorted(i).p, adjusted_alpha, ...
        ternary(h, 'Significant', 'Not significant'));
end

% Helper ternary function
function out = ternary(condition, valTrue, valFalse)
    if condition
        out = valTrue;
    else
        out = valFalse;
    end
end

%% Plot
model_labels = {'Kalman', 'SVM+KF', 'LSTM'};

% Combine data into one vector
all_rmse = [kalman_rmse, svm_k_rmse, lstm_rmse];
group_labels = [repmat({'Kalman'}, 1, length(kalman_rmse)), ...
                repmat({'SVM+KF'}, 1, length(svm_k_rmse)), ...
                repmat({'LSTM'}, 1, length(lstm_rmse))];

% Create boxplot
figure;
boxplot(all_rmse, group_labels, 'Symbol', 'o', ...
    'Widths', 0.6, 'Notch', 'off');

ylabel('RMSE per Trial');
set(gca, 'FontSize', 12)

% Overlay individual data points (optional)
hold on;
positions = [1, 2, 3];
colors = [0.2 0.2 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2];
for i = 1:3
    jitter = (rand(1, 50) - 0.5) * 0.2; % jitter to avoid overlap
    scatter(positions(i) + jitter, rmse_data(:,i), 30, ...
        'filled', 'MarkerFaceColor', colors(i,:), 'MarkerFaceAlpha', 0.5);
end

% Add legend for colored scatter points using dummy scatter objects
h_kf = scatter(nan, nan, 30, 'filled', 'MarkerFaceColor', colors(1,:), 'MarkerFaceAlpha', 0.5);
h_svm = scatter(nan, nan, 30, 'filled', 'MarkerFaceColor', colors(2,:), 'MarkerFaceAlpha', 0.5);
h_lstm = scatter(nan, nan, 30, 'filled', 'MarkerFaceColor', colors(3,:), 'MarkerFaceAlpha', 0.5);

legend([h_kf, h_svm, h_lstm], model_labels, ...
    'Location', 'northeast');

hold off;
