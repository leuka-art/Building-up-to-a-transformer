#1 Linear Regression: I built multiple linear regression in Python using the numpy module.

This project implements linear regression (OLS) and ridge regression from first principles using NumPy. 
It also explores gradient descent and k-fold cross-validation to understand model behaviour under multicollinearity and regularisation.

My aims for this project were to understand these techniques as well as:
- How linear regression works geometrically from a vector projection viewpoint
- Why closed-form solutions can become unstable (ill-conditioning)
- How gradient descent converges to the same solution
- How ridge regression stabilises solutions under multicollinearity
- How cross-validation can be used to find optimal parameters (lambda in this case for ridge regression)

Dataset

The model is trained on the Abalone dataset, where the task is to predict the number of rings which is used as a measure for age.

Features:
- Categorical gender
- Physical measurements (length, diameter, height, etc.)

Preprocessing:
- Z-score normalisation applied to features
- Gender encoded as numeric values: male=1, infant=0, female=-1

Models Implemented

X is the data matrix with each row being a set of input data. The data is standardised. y is the number of rings for the corresponding features. For OLS X has a 1s
column on the left.

Ordinary Least Squares

We minimise the squared error between prediction and actual values (RSS).

Closed-form solution:
\[
beta = (X^T X)^{-1} X^T y
\]
- X^TX may not be invertible for multicolinear data in which case a unique solution doesn't exist. In the case of small singular values,
  the condition number may become large and hence the matrix is ill-conditioned resulting in high variance of the parameters.

Gradient Descent Linear Regression

Goal is to minimise the residual sum of squares:

\[
\min_w \|y - Xw\|^2
\]

- Gradient descent done through a for loop
- RSS tracked over iterations and plotted against iteration to verify convergence

Ridge Regression

We aim to minimise the following:

\[
\min_w \|y - Xw\|^2 + \lambda \|w\|^2
\]

Closed-form solution:

\[
w = (X^T X + \lambda I)^{-1} X^T y
\]

- Reduces variance in the presence of correlated features
- Improves numerical stability under ill-conditioning

Cross-Validation

Implemented k-fold cross-validation to:

- Evaluate model performance through RMSE
- Compare the variance of parameters in OLS vs Ridge
- Select the optimal regularisation parameter λ
