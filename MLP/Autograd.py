import numpy as np

class Tensor:
    def __init__(self,data,prev=(),requires_grad=False):
        self.data=np.asarray(data, dtype=np.float64)
        self.parent=tuple(prev)
        self.requires_grad=requires_grad
        self.grad=np.zeros_like(self.data) if requires_grad else None
        self._backward=lambda: None
        self.shape=np.shape(self.data)
    
    def __add__(self,other):
        if not isinstance(other,Tensor):
            other=Tensor(other)
        out=Tensor(self.data+other.data,(self,other),requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad+=Tensor.unbroadcast(out.grad,self.shape)
            if other.requires_grad:
                other.grad+=Tensor.unbroadcast(out.grad,other.shape)
        out._backward=_backward
        return out

    def __mul__(self,other):
        if not isinstance(other,Tensor):
            other=Tensor(other)
        out=Tensor(self.data*other.data,(self,other),requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad+=Tensor.unbroadcast(other.data*out.grad,self.shape)
            if other.requires_grad:
                other.grad+=Tensor.unbroadcast(self.data*out.grad,other.shape)
        out._backward=_backward
        return out

    def __pow__(self,power):
        out=Tensor(self.data**power,(self,),requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad+=power*(self.data**(power-1))*out.grad
        out._backward=_backward
        return out
    
    def __truediv__(self,other):
        return self*(other**-1)
    
    def __neg__(self):
        return self*(-1)
    
    def __sub__(self,other):
        if not isinstance(other,Tensor):
            other=Tensor(other)
        return self+-other
    
    def __radd__(self,other):
        if not isinstance(other,Tensor):
            other=Tensor(other)
        return other+self
        
    def __rsub__(self,other):
        if not isinstance(other,Tensor):
            other=Tensor(other)
        return -other+self

    def __rmul__(self,other):
        if not isinstance(other,Tensor):
            other=Tensor(other)
        return other*self
    
    def exp(self):
        out=Tensor(np.exp(self.data),(self,),requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad+=out.data*out.grad
        out._backward=_backward
        return out
    
    def backward(self,grad=None):
        topo=[]
        visited=set()
        def topo_build(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parent:
                    topo_build(parent)
                topo.append(v)
        topo_build(self)
        for node in topo:
            if node.requires_grad:
                node.zero_grad()
        if grad is None:
            self.grad=np.ones_like(self.data)
        else:
            self.grad=grad
        for node in reversed(topo):
            node._backward()

    def __rtruediv__(self,other):
        if not isinstance(other,Tensor):
            other=Tensor(other)
        return other/self

    def zero_grad(self):
        if self.requires_grad:
            self.grad=np.zeros_like(self.data)
    
    def sum(self,axis=None,keepdims=False):
        out=Tensor(self.data.sum(axis=axis,keepdims=keepdims),(self,),requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                grad=out.grad
                if axis is None:
                    grad=np.ones_like(self.data)*grad
                else:
                    if not keepdims:
                        grad=np.expand_dims(grad,axis)
                    grad=np.broadcast_to(grad,self.data.shape)
                self.grad+=grad
        out._backward=_backward
        return out

    def mean(self,axis=None,keepdims=False):
        if axis is None:
            n=self.data.size
        else:
            n=self.data.shape[axis]
        return self.sum(axis=axis,keepdims=keepdims)/n
    
    @staticmethod
    def unbroadcast(grad,shape):
        while len(grad.shape)>len(shape):
            grad=grad.sum(axis=0)
        for i,dim in enumerate(shape):
            if dim==1 and grad.shape[i]!=1:
                grad=grad.sum(axis=i,keepdims=True)
        return grad
    
    def reshape(self, shape):
        out=Tensor(self.data.reshape(shape),(self,),requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad+=out.grad.reshape(self.shape)
        out._backward=_backward
        return out
    
    def transpose(self,axis1=-2,axis2=-1):
        out=Tensor(np.swapaxes(self.data,axis1,axis2),(self,),requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad+=np.swapaxes(out.grad,axis1,axis2)
        out._backward=_backward
        return out
    
    def __matmul__(self,other):
        if not isinstance(other,Tensor):
            other=Tensor(other)
        out=Tensor(self.data@other.data,(self,other),requires_grad=self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad+=Tensor.unbroadcast(out.grad@np.swapaxes(other.data,-2,-1),self.shape)
            if other.requires_grad:
                other.grad+=Tensor.unbroadcast(np.swapaxes(self.data,-2,-1)@out.grad,other.shape)
        out._backward=_backward
        return out
    
    def log(self):
        out=Tensor(np.log(self.data),(self,),requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad+=out.grad/self.data
        out._backward=_backward
        return out
    
    def ReLU(self):
        out=Tensor(np.maximum(0,self.data),(self,),requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad+=(self.data>0)*out.grad
        out._backward=_backward
        return out
    
    def max(self,axis=None,keepdims=False):
        out=Tensor(np.max(self.data, axis=axis, keepdims=keepdims),(self,),requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                grad=out.grad
                if axis is None:
                    mask=(self.data==np.max(self.data))
                    self.grad+=mask*grad
                else:
                    if not keepdims:
                        grad=np.expand_dims(grad,axis)
                    mask=(self.data==np.max(self.data,axis=axis,keepdims=True))
                    self.grad+=mask*grad
        out._backward=_backward
        return out
    
    def softmax(self,axis=-1):
        shifted=self-self.max(axis=axis,keepdims=True)
        exp=shifted.exp()
        return exp/exp.sum(axis=axis,keepdims=True)
    
    def __rmatmul__(self,other):
        if not isinstance(other,Tensor):
            other=Tensor(other)
        return other@self
    
    def log_softmax(self,axis=-1):
        shifted=self-self.max(axis=axis,keepdims=True)
        return shifted-shifted.exp().sum(axis=axis,keepdims=True).log()
    
    def __getitem__(self,index):
        out=Tensor(self.data[index],(self,),requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                np.add.at(self.grad,index,out.grad)
        out._backward=_backward
        return out