This repository documents my progression from classical machine learning to deep learning by implementing models from scratch, beginning with linear regression, building towards a basic transformer.

Linear Regression:

I implemented linear regression using numpy. Liner regression serves as a foundation for exploring how linear transformations might be used in neural networks. I wanted to explore an analytical view on linear regression.

I implemented the following:
- Least squares regression using closed-form solution and gradient descent optimisation
- Ridge regression closed-form solution
- Both of the above using numpy arrays
- Cross-validation for finding: optimal lambda for Ridge regression and to compare RMSE of the methods
- Bootstrap for the training set to investigate how ridge regression affects the variance of parameters and generalisation

Dataset: Kaggle abalone dataset
The aim is to predict the number of rings (measure of age).
Features are standardised using z-score normalisation (excluding non-binary features and gender).
Gender was coded as such: M=Male -> 1, I=Infant -> 0, F=Female -> -1.

I learned:
- How the closed-form solution arises in multi linear regression as well as the geometric intuition of the solution
- Ridge regressions utility in reducing variance of parameters
- How multicolinearity of data can theoretically cause illconditioning and hence unstable parameters
- How cross-validation can be used when presented with limited training data

Logistic Regression:

I implemented multiclass logistic regression from first principles using numpy. Logistic regression builds on top of linear models by using softmax as well as an example of a model with no closed form solution, hence the use of gradient descent.

The objective was to understand how multiclass classification works mathematically by implementing each component manually, including the forward pass, softmax, cross entropy loss and gradient descent optimisation.

I implemented the following:
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
