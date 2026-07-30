from Autograd import Tensor
import numpy as np
from Model import Linear,Sequential
from Activation import ReLU
from Loss import cross_entropy_loss, mean_squared_error
from Optimiser import SGD

def create_batches(x,truev,batch_size):
    #Function to create a batch with batch_size samples for SGD, this is a generator
    nosamples=x.shape[0]
    indices=np.random.permutation(nosamples)
    x=x.data[indices]
    truev=truev.data[indices]
    for i in range(0, nosamples, batch_size):
        data_batch=x[i:i+batch_size]
        truev_batch=truev[i:i+batch_size]
        yield Tensor(data_batch), Tensor(truev_batch)

def training(data,model,true_value,loss_fn,epochs,optimiser,batch_size,scheduler=None):
    """data is the input data matrix, true_value is the true output values, loss_fn is the loss function used,
    iterations is the number of epochs done for GD, learning_rate and batch_size are as standard for SGD"""
    losses=[]
    for epoch in range(epochs):
        total_samples=0
        total_loss=0
        #Looping through the batch created for SGD
        for x_batch, y_batch in create_batches(data, true_value, batch_size):
            prediction=model.forward_prop(x_batch)
            loss=loss_fn(prediction, y_batch)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            if scheduler:
                scheduler.step()
            #Batch size weighted loss as the last sample may be smaller than the rest
            total_loss+=loss.data*x_batch.shape[0]
            total_samples+=x_batch.shape[0]
        losses.append(total_loss/total_samples)
        print("Epoch ",epoch+1, "loss: ",losses[-1])
    return losses

def text_generate_batch(x,batch_size,block_size):
    pos=np.random.randint(0,len(x)-block_size,batch_size)
    X_batch=np.stack([x[i:i+block_size] for i in pos])
    Y_batch=np.stack([x[i+1:i+block_size+1] for i in pos])
    return Tensor(X_batch),Tensor(Y_batch)

def text_training(data,model,loss_fn,epochs,steps_per_epoch,optimiser,batch_size,block_size,scheduler=None):
    losses=[]
    for epoch in range(epochs):
        total_samples=0
        total_loss=0
        for i in range(steps_per_epoch):
            X_batch,Y_batch=text_generate_batch(data,batch_size,block_size)
            prediction=model.forward_prop(X_batch)
            loss=loss_fn(prediction,Y_batch)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            if scheduler:
                scheduler.step()
            total_loss+=loss.data*X_batch.shape[0]*block_size
            total_samples+=X_batch.shape[0]*block_size
        losses.append(total_loss/total_samples)
        print("Epoch ",epoch+1," loss: ",losses[-1])
    return losses