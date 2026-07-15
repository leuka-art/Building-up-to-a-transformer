import torch

class Layernorm:
    def __init__(self,nout):
        #Affine transformation parameters to be learned
        self.gamma=torch.ones(1,nout,dtype=torch.float32,requires_grad=True)
        self.beta=torch.zeros(1,nout,dtype=torch.float32,requires_grad=True)
    def forward(self,linOut):
        #Standardisation then affine transform
        mean=torch.mean(linOut,dim=1,keepdim=True)
        sd=torch.std(linOut,dim=1,keepdim=True,unbiased=False)
        normalised=(linOut-mean)/sd
        return normalised*self.gamma+self.beta
    def parameters(self):
        return [self.gamma,self.beta]
