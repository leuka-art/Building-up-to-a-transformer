from Autograd import Tensor
from Activation import ReLU
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
        self.weights=Tensor(np.random.randn(nin,nout)*(2/(nin+nout))**0.5,requires_grad=True,decay=True)
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
        self.lookup=Tensor(np.random.randn(vocab_size,dim)*0.02,requires_grad=True,decay=True)
    def forward(self,x):
        return self.lookup[x.data.astype(int)]
    
class Flatten(Module):
    def forward(self,x):
        return x.reshape((x.shape[0],-1))

class FFN(Module):
    def __init__(self,input_dim,output_dim,hidden_dim=128):
        self.ffn=Sequential([Linear(input_dim,hidden_dim),ReLU(),Linear(hidden_dim,output_dim)])
    def forward(self,x):
        return self.ffn.forward_prop(x)

class MultiHeadAttention(Module):
    def __init__(self,embedding_dim,no_heads,block_size,mask=False):
        self.embedding_dim=embedding_dim
        self.no_heads=no_heads
        self.block_size=block_size
        self.head_dim=embedding_dim//no_heads

        self.mask=mask

        if mask:
            self.mask_matrix=mask=np.triu(np.ones((block_size,block_size)),k=1)

        self.Wv=Tensor(np.random.randn(embedding_dim,self.head_dim*no_heads)*np.sqrt(1/(embedding_dim)),requires_grad=True,decay=True)
        self.Wk=Tensor(np.random.randn(embedding_dim,self.head_dim*no_heads)*np.sqrt(1/(embedding_dim)),requires_grad=True,decay=True)
        self.Wq=Tensor(np.random.randn(embedding_dim,self.head_dim*no_heads)*np.sqrt(1/(embedding_dim)),requires_grad=True,decay=True)

        self.Wo=Tensor(np.random.randn(embedding_dim,embedding_dim)*np.sqrt(1/(embedding_dim)),requires_grad=True,decay=True)

    def forward(self,decoder_out,encoder_out):
        B,Td,E=decoder_out.shape
        Be,Te,Ee=encoder_out.shape

        query=decoder_out@self.Wq
        key=encoder_out@self.Wk
        value=encoder_out@self.Wv

        split_query=query.reshape((B,Td,self.no_heads,self.head_dim)).transpose(1,2)
        split_key=key.reshape((B,Te,self.no_heads,self.head_dim)).transpose(1,2)
        split_value=value.reshape((B,Te,self.no_heads,self.head_dim)).transpose(1,2)

        scores=split_query@split_key.transpose()/np.sqrt(self.head_dim)

        if self.mask:
            scores=scores+Tensor(np.where(self.mask_matrix[:Td,:Td],-np.inf,0))

        softmax_scores=scores.softmax()@split_value
        recombined_scores=softmax_scores.transpose(1,2).reshape((B,Td,E))

        return recombined_scores@self.Wo