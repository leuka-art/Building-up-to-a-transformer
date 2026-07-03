This project is a growing project where I build up to a transformer from scratch starting from linear regression.

Linear Regression:

I implemented linear regression using numpy. Linear models are the foundations of multi-perceptron neural networks; I wanted to explore an analytical view on linear regression.

I built and discussed the following:
- Least squares regression using closed-form solution and gradient descent optimisation
- Ridge regression closed-form solution
- Both of the above using numpy arrays
- Cross-validation for finding: optimal lambda for Ridge regression and to compare RMSE of the methods
- Bootstrap for the training set to understand how the ridge might generalise compared to least squares, in particular the variance of parameters

Dataset:
I used the Kaggle abalone dataset; the aim is to predict the number of rings (measure of age).
Features are standardised using z-score normalisation (excluding non-binary features and gender).
Gender was coded as such: M=Male -> 1, I=Infant -> 0, F=Female -> -1.

I learned:
- How the closed-form solution arises in multi linear regression as well as the geometric intuition of the solution
- Ridge regressions utility in reducing variance of parameters and hence helping the model generalise better.
- How colinearity of data can theoretically affect results and your solution
- How cross-validation can be used when presented with limited training data

Logistic Regression:

I implemented multiclass logistic regression from first principles using numpy. Logistic regression builds on top of linear models by using softmax as well as an example of a model with no closed form solution, hence the use of gradient descent.

The objective was to understand how multiclass classification works mathematically by implementing each component manually, including the forward pass, softmax, cross entropy loss and gradient descent optimisation.

I built and discussed the following:
- Multiclass logistic regression trained using gradient descent
- Vectorised implementation in NumPy
- Numerically stable softmax
- Cross-entropy loss
- Training loss visualisation
- Classification accuracy evaluation

Dataset

The implementation is trained on the Kaggle Phone Price Range dataset.

Features are standardised using z-score normalisation (excluding binary features), and the target consists of four phone price classes.

I learned:
- The probabilistic interpretation of logistic regression
- Why softmax converts logits into class probabilities
- Derivation of the cross-entropy gradient
- The role of numerical stability in softmax implementations
- Differences between linear regression and logistic regression optimisation
