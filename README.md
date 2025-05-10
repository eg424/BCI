# BCI
The final LSTM code is in the `DeepLearning` branch. 
However, the current branch contains the following:

## Model Training and Testing
- **positionEstimatorTraining.m**: Previous version of sdLSTM (not currently used).
- **positionEstimatorTraining2.m**: LSTM version of the training model with high precision.
- **positionEstimator.m**: Previous predictor based on positionEstimatorTraining (not currently used).
- **positionEstimator2**: Current predictor based on positionEstimatorTraing2.
- **test_Prediction.m**: Tests the model's performance on four random trials and angle pairs for quicker computation.
- **testFunction_for_students.m**: Main testing function used for model evaluation.

## Trained Model
- **positionEstimatorTrained.mat**: The trained model currently used for predicting (x, y, angle) with high precision. This avoids having to run positionEstimatorTraining2 for training.

## Data Preprocessing
- **binSpikes**: Bins spike data into time intervals depending on spike trains and filters those with no activity (not currently used, may be relevant).
- **test_binsize**: Evaluates binSpikes implementation (spike-dependent time binning size and filtering).
- **preprocessData.m**: Code for preprocessing data before training (not currently used, may be relevant).
- **test_preprocessing**: Evaluates preprocessing implementation.
- **test_Norm.m**: Tests and evaluates best normalisation method before positionEstimatorTraining (not currently used).

## Results and Visualisations
- **SingleLayer.png**: Training loss for simpler version of `positionEstimatorTraining`.
- **SingleLayerPrediction.png**: Prediction results for the simpler trained `positionEstimator`.
- **DeeperNetwork.png**: Training loss for `positionEstimatorTraining` (sdLSTM).
- **DeeperNetworkPrediction.png**: Prediction results for the sdLSTM trained `positionEstimator`.
- **FinalPrediction.png**: Prediction results for final version `positionEstimator2`.
- **TrainingLoss.png**: Training loss for final version `positionEstimatorTraining2`.

## Notes
- Current code needs RMSE, might need toolbox-less implementation although only for comparison.
- binSpikes and preprocessData may be useful for other models.
- The trained model (`positionEstimatorTrained.mat`) is recommended for testing instead of retraining for efficiency. For further modifications, train a new model and replace `positionEstimatorTrained.mat` accordingly.
- `test_Prediction.m` provides a quick way to evaluate model performance on a subset of trials.
- Added currently unused figures and code for comparison in the report.
