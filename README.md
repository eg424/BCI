# BCI
## Model Training and Testing
- **positionEstimatorTraining.m**: LSTM version of the training model with high precision.
- **positionEstimator**: Current predictor based on positionEstimatorTraing.
- **test_Prediction.m**: Tests the model's performance on four random trials and angle pairs for quicker computation.
- **testFunction_for_students.m**: Main testing function used for model evaluation.

## Trained Model
- **positionEstimatorTrained.mat**: The trained model currently used for predicting (x, y, angle) with high precision. This avoids having to run positionEstimatorTraining for training.

## Results and Visualisations
- **FinalPrediction.png**: Prediction results for final version `positionEstimator`.
- **TrainingLoss.png**: Training loss for final version `positionEstimatorTraining`.

## Notes
- Current code needs RMSE, might need toolbox-less implementation although only for comparison.
- binSpikes and preprocessData may be useful for other models.
- The trained model (`positionEstimatorTrained.mat`) is recommended for testing instead of retraining for efficiency. For further modifications, train a new model and replace `positionEstimatorTrained.mat` accordingly.
- `test_Prediction.m` provides a quick way to evaluate model performance on a subset of trials.
