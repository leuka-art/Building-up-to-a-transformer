import torch
from Model import Linear,Sequential
from Activation import ReLU
from Loss import cross_entropy_loss, mean_squared_error

def create_batches(data,truev,batch_size):
    nosamples=data.shape[0]
    indices=torch.randperm(nosamples)
    data=data[indices]
    truev=truev[indices]
    for i in range(0, nosamples, batch_size):
        data_batch=data[i:i+batch_size]
        truev_batch=truev[i:i+batch_size]
        yield data_batch, truev_batch

def training(data,model,true_value,loss_fn,iterations,learning_rate,batch_size):
    losses=[]
    params=model.parameters()
    for epoch in range(iterations):
        no_batches=0
        total_loss=0
        for x_batch, y_batch in create_batches(data, true_value, batch_size):
            prediction=model.forward_prop(x_batch)
            loss=loss_fn(prediction, y_batch)
            loss.backward()
            with torch.no_grad():
                for param in params:
                    param -= learning_rate*param.grad
                    param.grad.zero_()
            total_loss+=loss.item()
            no_batches+=1
        losses.append(total_loss/no_batches)
    return losses