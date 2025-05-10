# Kalman Filter

## Model Description
This branch implements Kalman filter-based neural decoders, including classification-enhanced variants to improve prediction accuracy.

## Key Implementations
- **KalmanInit**: Initial baseline Kalman Filter implementation for (x, y) prediction.
- **HiddenStateKF**: Explores a hidden-state model. Underperforms vs standard KF.
- **ClassifierKF**: Introduces classification before filtering for angle-specific decoding.
  - **NBClassifier**: Naïve Bayes classifier + Kalman Filter. Suboptimal compared to SVM.
  - **SVMClassifier**: Support Vector Machine classifier + Kalman Filter. Best performing model in this family, committed to KalmanFilter.

## Results and Visualisations
- **KalmanSVM**: Prediction results for final version.
- **KalmanInit**: Prediction results for initial KF predictions.
- **HSKF**: Prediction results for Hidden State implementation.

## Notes
- Classifier-based decoding helped reduce angle-specific errors.
- The SVM+KF model achieves better performance than standalone KF or HiddenStateKF.
