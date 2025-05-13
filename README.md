# Neural Decoders for Hand Trajectory Estimation from M1 Activity

<div align="center">

| LSTM Model's Performance | Comparison of Models |
|------|-------------------------------------------|
| <img src="https://github.com/user-attachments/assets/38158b80-3f47-4297-a1a1-78b7e07291c0" alt="LSTM" width="300"/> | <img src="https://github.com/user-attachments/assets/df315e6c-bab9-4306-bed5-7b887ca4e71c" alt="RosePlots" width="300"/><br><img src="https://github.com/user-attachments/assets/8a555e34-4cb0-413e-9b0c-7fd96b362ca4" alt="ViolinPlots" width="300"/> |

</div>

## Contributions
- Mohammed AbuSadeh: Initial Github repository.
   - *Additional Models*: LDA, PCA, XGBoost, Population Vector.
   - **Report**: Statistical Analysis of models.
- Erik Garcia Oyono: Revised GitHub repository and statistical analyses.
   - **Models**: Multivariate Linear Regression, Euler, Initial SVM+Kalman, Hidden State Kalman, LSTM.
   - **Report**: Overall structure. LSTM Introduction, Method, Results, and Discussion. Normality, Friedman, Bonferroni-Holm Methods, Results, and Discussion.
- Virginia Greco:
   - **Models**: Kalman, NB+Kalman.
   - **Report**: Introduction, Kalman, Discussion, Results.
- Helena Kosovac Godart:
   - **Models**: Kalman, NB+Kalman.
   - **Report**: Introduction, Kalman Methods, Discussion and future improvements.
- Anna Pahl:
   - **Models**: SVM+Kalman.
   - *Additional Models*: Recurrent Neural Network (RNN).
   - **Report**: Abstract, SVM, SVM+Angle-Specific Kalman, Discussion, Conclusion.

## Overview
This repository contains implementations of various neural decoding models for brain-computer interface (BCI) applications. The goal is to predict hand movement trajectories from neural spike recordings using different machine learning and signal processing techniques.

## Repository Structure
The repository follows a structured branching strategy:

- **main**: Contains only the final, validated model for predictions (LSTM).
- **dev**: Used for development and testing before merging into main.

### Model Categories

#### [Linear Regression](https://github.com/eg424/BCI/tree/LinearRegression)
Contains Euler integration model. Stats and images added for comparison of initial model performance vs Euler integration.
- **LinRegInit**: Implements an initial linear regression model, later improved with Euler integration and comitted to LinearRegression.

#### [Kalman Filter](https://github.com/eg424/BCI/tree/KalmanFilter)
Contains SVM Classifier + Kalman Filter. Stats and images added for comparison of KF vs HSKF vs SVM+KF implementation.
- **KalmanInit**: Implements the initial Kalman filter (KF).
- **HiddenStateKF**: Hidden State Implementation. Worse performance than initial KF.
- **ClassifierKF**: Contains SVM classifier with KF for improved accuracy vs initial KF.
   - **NBClassifier**: Implements Naïve Bayes classification with KF. Worse performance than SVM classifier.
   - **SVMClassifier**: Implements Support Vector Machine classification with KF. Commited to `ClassifierKF` and `KalmanFilter` since best model within this family.

#### [Deep Learning](https://github.com/eg424/BCI/tree/DeepLearning)
Contains final LSTM model. Stats and images added for comparison of different hyperparameters.
- **LSTM**: Implements Long Short-Term Memory network. Commited to `DeepLearning`, where it was improved, and `main` since best overall performance.

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/eg424/BCI.git
   cd BCI
   ```
2. Add the repository to your MATLAB path. You can do this within MATLAB:
   ```sh
   addpath(genpath('path_to_cloned_repo'));
   ```
3. Run the test script in MATLAB. Make sure you have the following files in your working directory:
- monkeydata_training.mat
- positionEstimatorTrained.mat
- testFunction_for_students_MTb.m
4. Run the test script.
  In MATLAB, execute:
  ```sh
  RMSE = testFunction_for_students_MTb('YourTeamName');
  ```
  This script will:
  - Load the trained model parameters from positionEstimatorTrained.mat.
  - Test the decoder using a held-out portion of the monkeydata_training.mat dataset.
  - Plot predicted vs actual hand positions.
  - Output the Root Mean Squared Error (RMSE), total runtime, average time per prediction, and a weighted performance rank.

Note: If positionEstimatorTrained.mat does not exist, the script will train a new model from scratch using positionEstimatorTraining.m.

## Roadmap
- Developed an initial Linear Regression Model, later replaced with Euler's integration.
- Implemented a Kalman Filter. Tested and compared various methods for improved performance.
- Iteratively tested several Deep Learning approaches.
- Optimised and compared the top-three performing neural decoders.

## Notes
- The neural data have been generously provided by the laboratory of Prof. Krishna Shenoy at Stanford University.
