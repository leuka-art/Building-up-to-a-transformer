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

Autograd:

I implemented micrograd, a scalar version of autograd, in order to understand how the autograd I used in the MLP works. Using Karpathy's micrograd YouTube video to aid my understanding, I first built the micrograd then built it up into a basic Autograd for tensors.

I learned:
- Why autograd works in terms of forward and backward passes in combination with the chain rule
- Visualising the computation as a tree
- Understanding the topological sort from a depth first search implementation and why its used here
- How to handle broadcasting as well as the grads of broadcasted tensors

Multi Layer Perceptron (MLP):

I implemented an MLP, building the architecture from scratch. I used my Autograd engine for backpropagation.

I implemented:
- Linear class which has its parameters
- ReLU class which is the activation function
- MLP using a sequential class which takes as input a list of Linear and ReLU classes which makes up the MLP
- Training using full-batch gradient descent and mini-batch stochastic gradient descent

The dataset is the same as the one used for logistic regression and was processed and cleaned in the same way. I trained my MLP for the same classification task.

I learned:
- Why ReLU is used as an activation function as well as the drawbacks of other activation functions such as tanh and softmax in terms of optimisation (vanishing grads)
- The affect that activation functions have on the model such as the loss function not being convex and the ability to now learn non-linear relationships
- The importance of parameter initialisation and the Xavier intialisation
- How SGD works and why it intuitively may improve optimisation in the case of non convex functions


