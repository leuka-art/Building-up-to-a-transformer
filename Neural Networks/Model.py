from Autograd import Tensor
import numpy as np

class Module:
    def parameters(self):
        params=[]
        for value in self.__dict__.values():
            if isinstance(value, Tensor):
                params.append(value)
            elif isinstance(value, Module):
                params+=value.parameters()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module):
                        params+=item.parameters()
        return params

class Linear(Module):
    def __init__(self,nin,nout,bias=True):
        #Linear layer weights and biases
        self.weights=Tensor(np.random.randn(nin,nout)*(2/(nin+nout))**0.5,requires_grad=True)
        if bias:
            self.bias=Tensor(np.zeros(nout),requires_grad=True)
        else:
            self.bias=None
    def forward(self,x):
        #Forward propagation through linear layer
        if self.bias:
            return x@self.weights+self.bias
        else:
            return x@self.weights

class Sequential(Module):
    def __init__(self,layers):
        #List of layers in the model
        self.layers=layers
    def forward_prop(self,x):
        #Forward propagation through the model by looping through each layer from start to end
        y=x
        for layer in self.layers:
            y=layer.forward(y)
        return y
    
class Layernorm(Module):
    def __init__(self,nout):
        #Affine transformation parameters to be learned
        self.gamma=Tensor(np.ones((1,nout)),requires_grad=True)
        self.beta=Tensor(np.zeros((1,nout)),requires_grad=True)
    def forward(self,linOut):
        #Standardisation then affine transform
        mean=linOut.mean(axis=-1,keepdims=True)
        var=((linOut-mean)**2).mean(axis=-1,keepdims=True)
        sd=(var+1e-5)**(0.5)
        normalised=(linOut-mean)/sd
        return normalised*self.gamma+self.beta

class Embedding(Module):
    def __init__(self,vocab_size,dim):
        self.lookup=Tensor(np.random.randn(vocab_size,dim)*0.02,requires_grad=True)
    def forward(self,x):
        return self.lookup[x.data.astype(int)]
    
class Flatten(Module):
    def forward(self,x):
        return x.reshape((x.shape[0],-1))
