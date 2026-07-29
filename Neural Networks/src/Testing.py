from Autograd import Tensor
import numpy as np

def classification_accuracy(prediction,truev):
    prop=np.mean(np.argmax(prediction.data,axis=1)==np.argmax(truev.data,axis=1))
    return prop

def non_hot_one_accuracy(prediction,truev):
    prop=np.mean(np.argmax(prediction.data,axis=1)==truev.data.astype(int))
    return prop

def generate_text(model,prompt,encode_dict,decode_dict,max_tokens=200,block_size=64,adjustment=1.0):
    # Encode prompt
    tokens=[encode_dict[ch] for ch in prompt]
    for i in range(max_tokens):
        context=tokens[-block_size:]
        x=Tensor(np.array([context]))
        logits=model.forward_prop(x)
        logits=logits.data[0,-1,:]
        logits=logits/adjustment
        logits=logits-np.max(logits)
        probs=np.exp(logits)
        probs=probs/np.sum(probs)
        next_token=np.random.choice(len(probs),p=probs)
        tokens.append(next_token)
    return "".join([decode_dict[t] for t in tokens])