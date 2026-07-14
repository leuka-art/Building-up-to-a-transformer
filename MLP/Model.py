import torch

class Linear:
    def __init__(self,nin,nout):
        self.weights = torch.randn(nin,nout) * (2/(nin+nout))**0.5
        self.weights.requires_grad_()
        self.bias = torch.zeros(nout)
        self.bias.requires_grad_()
    def forward(self,x):
        return x@self.weights+self.bias
    def parameters(self):
        return [self.weights,self.bias]

class Sequential:
    def __init__(self,layers):
        self.layers=layers
    def forward_prop(self,x):
        y=x
        for layer in self.layers:
            y=layer.forward(y)
        return y
    def parameters(self):
        mylist=[]
        for layer in self.layers:
            p=layer.parameters()
            mylist.extend(p)
        return mylist