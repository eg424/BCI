# Banana-Certified Interfaces

## Contributions
- Mohammed AbuSadeh: Initial Github repository.
   - *Additional Models*: LDA, PCA, XGBoost, Population Vector.
   - **Report**: Statistical Analysis of models.
- Erik Garcia Oyono: Revised GitHub repository.
   - **Models**: Multivariate Linear Regression, Euler, Initial SVM+Kalman, Hidden State Kalman, LSTM.
   - **Report**: LSTM Introduction, Method, Results, and Discussion.
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
- **LSTM**: Implements Long Short-Term Memory network. Commited to `DeepLearning` and `main` since best overall performance.

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/eg424/BCI.git
   cd BCI
   ```
2. Switch to the development branch for testing new features:
   ```sh
   git checkout dev
   ```
3. To contribute, create a new branch from `dev`, implement changes, and merge only the best-performing models into `main`.

## Roadmap
- Developed an initial Linear Regression Model, later replaced with Euler's integration.
- Implemented a Kalman Filter. Tested and compared various methods for improved performance.
- Iteratively tested several Deep Learning approaches.
- Optimised and compared the top-three performing neural decoders.

## Notes
- The neural data have been generously provided by the laboratory of Prof. Krishna Shenoy at Stanford University.
