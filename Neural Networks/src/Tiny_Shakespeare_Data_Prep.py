import numpy as np
from Autograd import Tensor

def data_read(split):
    with open("../data/Tiny_Shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars=sorted(list(set(text)))
    vocab_size=len(chars)

    encode_dict={ch:id for id,ch in enumerate(chars)}
    decode_dict={id:ch for ch,id in encode_dict.items()}

    tokens=np.array([encode_dict[ch] for ch in text])

    n=int(len(tokens)*split)

    return vocab_size,encode_dict,decode_dict,tokens[:n],tokens[n:]


def generate_dataset(tokens,block_size):
    input=[]
    output=[]

    for i in range(len(tokens)-block_size):
        input.append(tokens[i:i+block_size])
        output.append(tokens[i+1:i+1+block_size])

    input=Tensor(np.array(input),requires_grad=False)
    output=Tensor(np.array(output),requires_grad=False)

    return input,output