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