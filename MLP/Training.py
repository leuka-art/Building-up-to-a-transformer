import torch
from Model import Linear,Sequential
from Activation import ReLU
from Loss import cross_entropy_loss, mean_squared_error

def create_batches(data,truev,batch_size):
    #Function to create a batch with batch_size samples for SGD, this is a generator
    nosamples=data.shape[0]
    indices=torch.randperm(nosamples)
    data=data[indices]
    truev=truev[indices]
    for i in range(0, nosamples, batch_size):
        data_batch=data[i:i+batch_size]
        truev_batch=truev[i:i+batch_size]
        yield data_batch, truev_batch

def training(data,model,true_value,loss_fn,iterations,learning_rate,batch_size):
    """data is the input data matrix, true_value is the true output values, loss_fn is the loss function used,
    iterations is the number of epochs done for GD, learning_rate and batch_size are as standard for SGD"""
    losses=[]
    params=model.parameters()
    for epoch in range(iterations):
        total_samples=0
        total_loss=0
        #Looping through the batch created for SGD
        for x_batch, y_batch in create_batches(data, true_value, batch_size):
            prediction=model.forward_prop(x_batch)
            loss=loss_fn(prediction, y_batch)
            loss.backward()
            #Updating parameters
            with torch.no_grad():
                for param in params:
                    param -= learning_rate*param.grad
                    param.grad.zero_()
            #Batch size weighted loss as the last sample may be smaller than the rest
            total_loss+=loss.item()*x_batch.shape[0]
            total_samples+=x_batch.shape[0]
        losses.append(total_loss/total_samples)
    return losses