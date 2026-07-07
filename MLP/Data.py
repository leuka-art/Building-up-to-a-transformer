import pandas as pd
import torch

def training_data_processing():
    training_data=pd.read_csv('phone_pricing_range_train.csv')
    training_data["p0"] = (training_data["price_range"] == 0).astype(int) #Adding a column for each price range class
    training_data["p1"] = (training_data["price_range"] == 1).astype(int)
    training_data["p2"] = (training_data["price_range"] == 2).astype(int)
    training_data["p3"] = (training_data["price_range"] == 3).astype(int)
    features=torch.tensor(training_data.drop(["p0","p1","p2","p3","price_range"],axis=1).values,dtype=torch.float32)
    price_range=torch.tensor(training_data[["p0","p1","p2","p3"]].values,dtype=torch.float32)
    nonbinaryindex=torch.tensor([ 0,  2,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])
    normalisedfeatures=torch.tensor(training_data.drop(["p0","p1","p2","p3","price_range"],axis=1).values,dtype=torch.float32)
    #Standardisation used to reduce chance of divergence or overflow during optimisation
    columnmean=torch.mean(normalisedfeatures[:,nonbinaryindex],axis=0)
    columnstd=torch.std(normalisedfeatures[:,nonbinaryindex],axis=0)
    normalisedfeatures[:,nonbinaryindex]=(normalisedfeatures[:,nonbinaryindex]-columnmean)/columnstd
    nofeatures=normalisedfeatures.shape[1]
    noclasses=price_range.shape[1]
    nosamples=normalisedfeatures.shape[0]
    return normalisedfeatures,price_range,columnmean,columnstd,nofeatures,noclasses,nosamples

def testing_data_processing(columnmean,columnstd):
    test1data=pd.read_csv('test1.csv')
    test1data['p0']=(test1data['price_range']==0).astype(int)
    test1data['p1']=(test1data['price_range']==1).astype(int)
    test1data['p2']=(test1data['price_range']==2).astype(int)
    test1data['p3']=(test1data['price_range']==3).astype(int)
    nonbinaryindex=torch.tensor([ 0,  2,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])
    normalised=torch.tensor(test1data.drop(["p0","p1","p2","p3","price_range"],axis=1).values,dtype=torch.float32)
    normalised[:,nonbinaryindex]=(normalised[:,nonbinaryindex]-columnmean)/columnstd
    testprice=test1data[["p0","p1","p2","p3"]]
    y=torch.tensor(testprice.values,dtype=torch.float32)
    return normalised,y