import torch

def cross_entropy_loss(prediction,true_value):
    return torch.mean(torch.sum(-true_value*torch.log_softmax(prediction,axis=1),axis=1))

def mean_squared_error(prediction,true_value):
    return torch.mean((true_value-prediction)**2)