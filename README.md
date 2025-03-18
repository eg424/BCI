# Banana-Certified Interfaces

## Overview
This repository contains implementations of various neural decoding models for brain-computer interface (BCI) applications. The goal is to predict hand movement trajectories from neural spike recordings using different machine learning and signal processing techniques.

## Repository Structure
The repository follows a structured branching strategy:

- **main**: Stores only the final, validated versions of models.
- **dev**: Used for development and testing before merging into main.

### Model Categories

#### Linear Regression
- **InitialModel**: Implements an initial linear regression model, later improved with Euler integration and comitted to LinearRegression.
- **XGBoost**: *need to add Mo’s*.

#### Kalman Filter
- **ModelInit**: Implements the initial Kalman filter (KF); * need to test Hidden State Implementation. Commit to `KalmanFilter` whichever's best.*
- **ClassifierKF**: Use of a classifier combined with KF/Hidden State KF (HSKF).
- **NBClassifier**: Implements Naïve Bayes classification with KF/HSKF. The current best version is committed to `KalmanFilter`.
- **SVM Classification + KF/HSKF**: Implements SVM classification with KF/HSKF. * Commit to `ClassifierKF` if best*.
- **Sliding Window Classification + KF/HSKF**: *needs code for implementation and testing. Commit to `ClassifierKF` if best*.
- **Hybrid KF/HSKF**: *needs KF/HSKF + Multilayer Perceptron implementation*.

#### Deep Learning
- **Feedforward**: comit to `DeepLearning` if best.
- **LSTM**: Currently testing variations (vLSTM vs. sdLSTM). The best version will be committed to `DeepLearning`.
- **GRU/TCN**: commit to `DeepLearning` if best.

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
- Test Hidden State Kalman Filter (HSKF) implementation.
- Finalize and integrate the best classifier-based KF/HSKF model.
- Improve hybrid decoding by combining MLP with KF/HSKF.
- Optimize deep learning approaches and finalize the best performing neural network-based decoder.

## Contributors
- **eg424** and collaborators.

## License
This repository is licensed under [MIT License](LICENSE).

