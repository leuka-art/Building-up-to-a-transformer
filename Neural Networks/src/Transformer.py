from Autograd import Tensor
import numpy as np
from Model import Module,MultiHeadAttention,Layernorm,FFN,Embedding,Linear

class TransformerBlock(Module):
    def __init__(self,embedding_dim,no_heads,block_size,hidden_dim):
        self.attention=MultiHeadAttention(embedding_dim,no_heads,block_size,True)
        self.layernorm1=Layernorm(embedding_dim)
        self.ffn=FFN(embedding_dim,embedding_dim,hidden_dim)
        self.layernorm2=Layernorm(embedding_dim)
    def forward(self,x):
        attention_scores=self.attention.forward(x,x)
        layernormed=self.layernorm1.forward(x+attention_scores)
        ff=self.ffn.forward(layernormed)
        layernormed2=self.layernorm2.forward(ff+layernormed)
        return layernormed2

class Transformer(Module):
    def __init__(self,embedding_dim,no_heads,vocab_size,block_size,hidden_dim,no_blocks):
        self.block_size=block_size
        self.head_dim=embedding_dim//no_heads
        self.no_heads=no_heads
        self.no_blocks=no_blocks

        self.token_embedding=Embedding(vocab_size,embedding_dim)
        self.position_embedding=Embedding(block_size,embedding_dim)

        self.transformerblocks=[TransformerBlock(embedding_dim,no_heads,block_size,hidden_dim) for i in range(no_blocks)]

        self.linear=Linear(embedding_dim,vocab_size)
        
    def forward_prop(self,x):
        B,T=x.shape

        embedding=self.token_embedding.forward(x)+self.position_embedding.forward(Tensor(np.arange(T),requires_grad=False))
        
        for block in self.transformerblocks:
            embedding=block.forward(embedding)

        linear_out=self.linear.forward(embedding)

        return linear_out