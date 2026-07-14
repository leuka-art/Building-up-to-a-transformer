import torch

class ReLU:
    def forward(self,x):
        return torch.relu(x)
    def parameters(self):
        return []
    
class Tanh:
    def forward(self,x):
        return torch.tanh(x)
    def parameters(self):
        return []