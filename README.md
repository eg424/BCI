# Banana-Certified Interfaces

## Overview
This repository contains implementations of various neural decoding models for brain-computer interface (BCI) applications. The goal is to predict hand movement trajectories from neural spike recordings using different machine learning and signal processing techniques.

## Repository Structure
The repository follows a structured branching strategy:

- **main**: Stores only the final, validated versions of models.
- **dev**: Used for development and testing before merging into main.

### Model Categories

#### Linear Regression
Contains multivariate linear regression model with Euler integration for improved accuracy. Stats and images added for comparison of initial model performance vs Euler integration.
- **LinRegInit**: Implements an initial linear regression model, later improved with Euler integration and comitted to LinearRegression.
- **XGBoost**: *Need to add Mo’s*.

#### Dimensionality Reduction
   - **PCA_LDA**: PCA for dimensionality reduction and LDA for classification. *Needs code for implementation and testing. Commit to `DimReduction` if best*.
#### Kalman Filter
Contains initial Kalman Filter for now. Stats and images added for comparison of KF vs HSKF implementation.
- **KalmanInit**: Implements the initial Kalman filter (KF).  The current best version is committed to `KalmanFilter` as better than HSKF.
- **HiddenStateKF**: Hidden State Implementation. Worse performance than initial KF.
- **ClassifierKF**: *No code yet as best classifier still needs to be determined. Once determined, it will contain the best classifier with KF for improved accuracy. Stats and images for comparison of initial KF vs each classifier (e.g., vs NB, vs SVM, vs SW)*.
   - **NBClassifier**: Implements Naïve Bayes classification with KF. Important finding in the last commit.
   - **SVM Classification + KF**: Implements SVM classification with KF. *Needs improvements, commit to `ClassifierKF` if best*.
   - **Sliding Window Classification + KF**: *Needs code for implementation and testing. Commit to `ClassifierKF` if best*.
- **Hybrid KF/HSKF**: *Needs KF + Multilayer Perceptron implementation*.

#### Deep Learning
- **Feedforward**: *Commit to `DeepLearning` if best*.
- **LSTM**: *Currently testing variations (vLSTM vs. sdLSTM). Will be committed to `DeepLearning` if best*.
- **GRU/TCN**: *Commit to `DeepLearning` if best*.

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
- Test and determine the best classifier-based KF model.
- Improve hybrid decoding by combining MLP with KF.
- Optimise deep learning approaches and finalise the best performing NN-based decoder.

