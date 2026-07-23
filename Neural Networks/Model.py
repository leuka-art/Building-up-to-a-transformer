from Autograd import Tensor
import numpy as np

class Linear:
    def __init__(self,nin,nout):
        #Linear layer weights and biases
        self.weights=Tensor(np.random.randn(nin,nout)*(2/(nin+nout))**0.5,requires_grad=True)
        self.bias=Tensor(np.zeros(nout),requires_grad=True)
    def forward(self,x):
        #Forward propagation through linear layer
        return x@self.weights+self.bias
    def parameters(self):
        #Return the parameters of the linear layer
        return [self.weights,self.bias]

class Sequential:
    def __init__(self,layers):
        #List of layers in the model
        self.layers=layers
    def forward_prop(self,x):
        #Forward propagation through the model by looping through each layer from start to end
        y=x
        for layer in self.layers:
            y=layer.forward(y)
        return y
    def parameters(self):
        #Returns all parameters in the model starting from the first layer sequentially to the last
        mylist=[]
        for layer in self.layers:
            p=layer.parameters()
            mylist.extend(p)
        return mylist
    
class Layernorm:
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
    def parameters(self):
        return [self.gamma,self.beta]

class Embedding:
    def __init__(self,vocab_size,dim):
        self.lookup=Tensor(np.random.randn(vocab_size,dim)*0.02,requires_grad=True)
    def forward(self,x):
        return self.lookup[x.data.astype(int)]
    def parameters(self):
        return [self.lookup]
    
class Flatten:
    def forward(self,x):
        return x.reshape((x.shape[0],-1))
    def parameters(self):
        return []
    
class EncodingPostionalEmbedding:
    def __init__(self,vocab_size,block_size,dim):
        self.block_size=block_size
        self.encode=Embedding(vocab_size,dim)
        self.position=Embedding(block_size,dim)
    def forward(self,x):
        return self.encode.forward(x)+self.position.forward(Tensor(np.arange(self.block_size)))
    def parameters(self):
        return [self.encode.lookup,self.position.lookup]