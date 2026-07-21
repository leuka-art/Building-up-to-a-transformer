from Autograd import Tensor
import numpy as np

def cross_entropy_one_hot(prediction, true_value):
    log_probs=prediction.log_softmax(axis=1)
    return (-true_value*log_probs).sum(axis=1).mean()

def cross_entropy_loss(prediction, true_value):
    log_probs=prediction.log_softmax(axis=1)
    batch_size=prediction.shape[0]
    loss=0
    for i in range(batch_size):
        loss+=-log_probs[i, int(true_value.data[i])]
    return loss/batch_size

def mean_squared_error(prediction,true_value):
    return ((true_value-prediction)**2).mean()