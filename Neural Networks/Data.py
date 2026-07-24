import pandas as pd
import numpy as np
from Autograd import Tensor

def training_data_processing():
    """Function which loads training data set as a data frame then processes data"""
    training_data=pd.read_csv('phone_pricing_range_train.csv')

    #Adding a column for each price range class
    training_data["p0"] = (training_data["price_range"] == 0).astype(int) 
    training_data["p1"] = (training_data["price_range"] == 1).astype(int)
    training_data["p2"] = (training_data["price_range"] == 2).astype(int)
    training_data["p3"] = (training_data["price_range"] == 3).astype(int)

    #Creates tensor from the pandas data frames for the output price range classes
    price_range=np.array((training_data[["p0","p1","p2","p3"]].values))

    #The indices of the nonbinary columns
    nonbinaryindex=np.array([0,2,6,7,8,9,10,11,12,13,14,15,16])

    #Creates a tensor from the pandas data frame of the input data, hence excluding price range and the price classes
    normalisedfeatures=(np.array(training_data.drop(["p0","p1","p2","p3","price_range"],axis=1)))

    #z-score standardisation used to reduce chance of divergence or overflow during optimisation
    columnmean=np.mean(normalisedfeatures[:,nonbinaryindex],axis=0)
    columnstd=np.std(normalisedfeatures[:,nonbinaryindex],axis=0)
    normalisedfeatures[:,nonbinaryindex]=(normalisedfeatures[:,nonbinaryindex]-columnmean)/columnstd

    #Number of features, classes and samples
    nofeatures=normalisedfeatures.shape[1]
    noclasses=price_range.shape[1]
    nosamples=normalisedfeatures.shape[0]
    return Tensor(normalisedfeatures),Tensor(price_range),columnmean,columnstd,nofeatures,noclasses,nosamples

def testing_data_processing(columnmean,columnstd):
    """Function which loads training data set as a data frame then processes data, 
    inputs are the column mean and column population standard deviation for standardisation"""
    test1data=pd.read_csv('test1.csv')

    #Adding a column for each price range class
    test1data['p0']=(test1data['price_range']==0).astype(int)
    test1data['p1']=(test1data['price_range']==1).astype(int)
    test1data['p2']=(test1data['price_range']==2).astype(int)
    test1data['p3']=(test1data['price_range']==3).astype(int)

    #The indices of the nonbinary columns
    nonbinaryindex=np.array([0,2,6,7,8,9,10,11,12,13,14,15,16])

    #Creates a tensor from the pandas data frame of the input data
    normalised=np.array((test1data.drop(["p0","p1","p2","p3","price_range"],axis=1)))

    #z-score standardisation used to reduce chance of divergence or overflow during optimisation
    normalised[:,nonbinaryindex]=(normalised[:,nonbinaryindex]-columnmean)/columnstd

    #Creating pandas data frame of price classes then creating tensor for these
    testprice=test1data[["p0","p1","p2","p3"]]
    y=np.array(testprice.values)
    return Tensor(normalised),Tensor(y)