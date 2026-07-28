import numpy as np

def save_model(model, filename):
    params=model.parameters()
    weights=[p.data for p in params]
    np.save(filename,weights,allow_pickle=True)

def load_model(model,filename):
    weights=np.load(filename,allow_pickle=True)
    params=model.parameters()
    for p,w in zip(params,weights):
        p.data=w
