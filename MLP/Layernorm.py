from Autograd import Tensor
import numpy as np

class Layernorm:
    def __init__(self,nout):
        #Affine transformation parameters to be learned
        self.gamma=Tensor(np.ones((1,nout)),requires_grad=True)
        self.beta=Tensor(np.zeros((1,nout)),requires_grad=True)
    def forward(self,linOut):
        #Standardisation then affine transform
        mean=linOut.mean(axis=1,keepdims=True)
        var=((linOut-mean)**2).mean(axis=1,keepdims=True)
        sd=var**(0.5)
        normalised=(linOut-mean)/sd
        return normalised*self.gamma+self.beta
    def parameters(self):
        return [self.gamma,self.beta]
