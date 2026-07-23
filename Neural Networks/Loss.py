from Autograd import Tensor
import numpy as np

def cross_entropy_one_hot(prediction, true_value):
    log_probs=prediction.log_softmax(axis=1)
    return (-true_value*log_probs).sum(axis=1).mean()

def cross_entropy_loss(prediction, true_value):
    x=prediction.data
    batch_size=x.shape[0]
    labels=true_value.data.astype(int)
    shifted=x-np.max(x,axis=1,keepdims=True)
    exp_shifted=np.exp(shifted)
    softmax=exp_shifted/exp_shifted.sum(axis=1,keepdims=True)
    log_sum_exp=np.log(exp_shifted.sum(axis=1))
    losses=-shifted[np.arange(batch_size),labels]+log_sum_exp
    out=Tensor(losses.mean(),(prediction,),requires_grad=prediction.requires_grad)
    def _backward():
        if prediction.requires_grad:
            grad=softmax.copy()
            grad[np.arange(batch_size),labels]-=1
            prediction.grad+=(grad/batch_size)*out.grad
    out._backward=_backward
    return out

def mean_squared_error(prediction,true_value):
    return ((true_value-prediction)**2).mean()