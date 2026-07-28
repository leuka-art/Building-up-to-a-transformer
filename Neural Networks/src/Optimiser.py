from Autograd import Tensor
import numpy as np

class SGD:
    def __init__(self,params,lr=0.05):
        self.parameters=params
        self.lr=lr

    def step(self):
        for p in self.parameters:
            p.data-=self.lr*p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

class Momentum:
    def __init__(self,params,lr=0.05,beta=0.9):
        self.parameters=params
        self.lr=lr
        self.beta=beta
        self.velocity=[np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i,p in enumerate(self.parameters):
            self.velocity[i]=self.beta*self.velocity[i]+p.grad
            p.data-=self.lr*self.velocity[i]

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

class Adagrad:
    def __init__(self,params,lr=0.05,eta=0.9,epsilon=0.000001):
        self.parameters=params
        self.lr=lr
        self.s=[np.zeros_like(p.data) for p in self.parameters]
        self.epsilon=epsilon

    def step(self):
        for i,p in enumerate(self.parameters):
            self.s[i]+=p.grad**2
            p.data-=self.lr*p.grad/np.sqrt(self.s[i]+self.epsilon)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

class RMSProp:
    def __init__(self,params,lr=0.05,gamma=0.9,epsilon=0.000001):
        self.parameters=params
        self.lr=lr
        self.gamma=gamma
        self.s=[np.zeros_like(p.data) for p in self.parameters]
        self.epsilon=epsilon

    def step(self):
        for i,p in enumerate(self.parameters):
            self.s[i]=self.gamma*self.s[i]+(1-self.gamma)*p.grad**2
            p.data-=self.lr*p.grad/np.sqrt(self.s[i]+self.epsilon)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

class Adadelta:
    def __init__(self,params,rho=0.9,epsilon=0.000001):
        self.parameters=params
        self.rho=rho
        self.s=[np.zeros_like(p.data) for p in self.parameters]
        self.delta=[np.zeros_like(p.data) for p in self.parameters]
        self.epsilon=epsilon

    def step(self):
        for i,p in enumerate(self.parameters):
            self.s[i]=self.rho*self.s[i]+(1-self.rho)*p.grad**2
            scaledp=np.sqrt(self.delta[i]+self.epsilon)/np.sqrt(self.s[i]+self.epsilon)*p.grad
            p.data-=scaledp
            self.delta[i]=self.rho*self.delta[i]+(1-self.rho)*scaledp**2

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

class Adam:
    def __init__(self,params,lr=0.001,beta1=0.9,beta2=0.999,epsilon=0.000001):
        self.parameters=params
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.v=[np.zeros_like(p.data) for p in self.parameters]
        self.s=[np.zeros_like(p.data) for p in self.parameters]
        self.epsilon=epsilon
        self.t=1

    def step(self):
        for i,p in enumerate(self.parameters):
            self.v[i]=self.beta1*self.v[i]+(1-self.beta1)*p.grad
            self.s[i]=self.beta2*self.s[i]+(1-self.beta2)*p.grad**2
            vnorm=self.v[i]/(1-self.beta1**self.t)
            snorm=self.s[i]/(1-self.beta2**self.t)
            gnorm=self.lr*vnorm/(np.sqrt(snorm)+self.epsilon)
            p.data-=gnorm
        self.t+=1

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

class AdamW:
    def __init__(self,params,lr=0.001,beta1=0.9,beta2=0.999,lam=0.01,epsilon=0.000001):
        self.lam=lam
        self.parameters=params
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.v=[np.zeros_like(p.data) for p in self.parameters]
        self.s=[np.zeros_like(p.data) for p in self.parameters]
        self.epsilon=epsilon
        self.t=1

    def step(self):
        for i,p in enumerate(self.parameters):
            self.v[i]=self.beta1*self.v[i]+(1-self.beta1)*p.grad
            self.s[i]=self.beta2*self.s[i]+(1-self.beta2)*p.grad**2
            vnorm=self.v[i]/(1-self.beta1**self.t)
            snorm=self.s[i]/(1-self.beta2**self.t)
            gnorm=self.lr*vnorm/(np.sqrt(snorm)+self.epsilon)
            if p.decay==True:
                p.data=p.data-gnorm-self.lr*self.lam*p.data
            else:
                p.data=p.data-gnorm
        self.t+=1

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

class warmup_cosine_scheduler:
    def __init__(self,optimiser,warm_up,total):
        self.optimiser=optimiser
        self.step_num=0
        self.lr=optimiser.lr
        self.warm_up=warm_up
        self.total=total
    def step(self):
        self.step_num+=1
        if self.step_num<self.warm_up:
            self.optimiser.lr=self.lr*self.step_num/self.warm_up
        else:
            frac=(self.step_num-self.warm_up)/(self.total-self.warm_up)
            self.optimiser.lr=0.5*self.lr*(1+np.cos(np.pi*frac))

