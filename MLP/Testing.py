import torch
def classification_accuracy(prediction,truev):
    prop=torch.mean(torch.argmax(prediction,axis=1)==torch.argmax(truev,axis=1),dtype=torch.float32)
    return prop