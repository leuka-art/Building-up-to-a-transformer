from Autograd import Tensor
import numpy as np

def cross_entropy_loss(prediction,true_value):
    return (-true_value*prediction.log_softmax(axis=1)).sum(axis=1).mean()
def mean_squared_error(prediction,true_value):
    return ((true_value-prediction)**2).mean()