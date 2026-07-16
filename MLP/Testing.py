from Autograd import Tensor
import numpy as np

def classification_accuracy(prediction,truev):
    prop=np.mean(np.argmax(prediction.data,axis=1)==np.argmax(truev.data,axis=1))
    return prop