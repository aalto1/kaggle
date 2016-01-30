# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 11:23:36 2016

@author: aalto
"""
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.cross_validation import train_test_split
 
train = None
test = None
trainMode = False
numPD=10
numAdd=23228
numCrimes= 39
conv = {}
crimeList = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY',
       'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE',
       'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
       'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING',
       'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING',
       'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES',
       'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE',
       'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE',
       'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE',
       'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT',
       'WARRANTS', 'WEAPON LAWS']


def getCleanData():
    """ Loads in main memory the datasets and clean it. Column -Dates- is
    converted into a timeseries and used as index, -Description- and 
    -Resolution- are dropped because not present in the final test set-
    If the trainMode is active it split the dataset in train-80 test-20, 
    if not it uses the kaggle's datasets as given """
    
    train = pd.read_csv("data/train.csv")
    train = train.set_index(pd.to_datetime(train["Dates"]))
    train = train.drop(["Dates", "Descript","Resolution"], axis=1) 
    aux = np.unique(train["Category"], return_counts=True)
    freq = aux[1]*1.0/sum(aux[1])
    freq.dump("data/probs.data")
    if trainMode:
        train, test = train_test_split(train, test_size = 0.2)
    else:
        test = pd.read_csv("data/test.csv")
        test = test.set_index(pd.to_datetime(test["Dates"])) 
    return train, test
    
    
def cornersHandler(addr):
    """ It orders alphabetically any address that has the / charachter in it:
    those address are crossroads and in this way we bring the number of
    unique address from 22k to 17k, increasing the accuracy """
    
    if type(addr) == type("ciao"):
        if " / " in addr:
            streets = sorted(addr.split(" / "))
            addr = streets[0] + " / " + streets [1]
            return addr
        else:
            return addr
    else:
        for elem, k in zip(addr, range(len(addr))):
            if " / " in elem:
                streets = sorted(elem.split(" / "))
                addr[k] = streets[0] + " / " + streets [1]
        print "Unique addresses:", np.unique(addr).shape[0]
        return np.unique(addr)
            

def getDics():
    """ Returns dictionaries to handle numpy arrays in a dataframe-like
    fashion. For each label a number is assigned in this way we can use
    the efficiency of dictionaries and numpy with the easiness to call the
    variable using labels """
    
    crimes  = np.unique(train["Category"].values)
    PDs     = train["PdDistrict"].unique()
    addr    = cornersHandler(train["Address"].unique())
    dof     = train["DayOfWeek"].unique()    
    dicCrime= dict(zip(crimes, range(crimes.size)))
    invCrime= dict(zip(range(crimes.size), crimes))
    dicPDs  = dict(zip(PDs, range(PDs.size)))
    dicAddr = dict(zip(addr, range(addr.size)))
    dicDof = dict(zip(dof, range(dof.size)))
    return {"crime" : dicCrime, "PD" : dicPDs, "addr" : dicAddr, "dof": dicDof, "inv" : invCrime}

def insertCrime(LHT, index, row):
    """ Increase by one the slot value for each of the 8 likelihood table on the 
    basis of the infos given by the dataset """
    
    crime = conv["crime"][row["Category"]]
    dof = conv["dof"][row["DayOfWeek"]]
    PD = conv["PD"][row["PdDistrict"]]
    addr = conv["addr"][cornersHandler(row["Address"])]
    LHT[0][index.hour-1][crime] += 1
    LHT[1][index.day-1][crime] += 1
    LHT[2][dof][crime] += 1
    LHT[3][index.month-1][crime] += 1
    LHT[4][index.year-2003][crime] += 1
    LHT[5][PD][crime] += 1
    LHT[6][addr][crime] += 1
    LHT[7][index.minute-1][crime] +=1

def computeLHTables():
    """ Create, populate and smooth (laplacian) the likelihood tables for each of the
    features of the Naive Bayesian Classifier: 
    * Minutes
    * Hours
    * Day
    * Day of the week
    * Month
    * Year
    * Police Department (not used in the predictions) """
    
    print "Computing LHTables..."
    LHT = [24, 31, 7, 12, 13, 10, 17812, 60]
    for rows, k in zip(LHT, range(len(LHT))):
        LHT[k] = np.zeros((rows, 39), dtype=int)
    for index, row in train.iterrows():
        insertCrime(LHT, index, row)
    for table, k in zip(LHT, range(len(LHT))):
        print k
        LHT[k] = (table+1.0)/(table.sum(axis=0, keepdims=True)+table.shape[0])
    return LHT
    
def getKDTree():
    """ Creates a K-dimensional tree to get, in case an address misses, the
    nearest address in the train set based on the crime X-Y coordinates """
    
    dx = train[["X", "Y", "Address"]][train["Y"]<39].drop_duplicates().values
    dx = pd.DataFrame(dx, index = range(dx.shape[0]), columns = ["X", "Y", "Address"])
    A = dx[["X","Y"]].values
    return spatial.KDTree(A), dx
    
    
def oracle(LHT, probs):
    """ Compute for each crime the posterior probability that, given the 
    features, the crime belongs to a certain category. If the address is not
    present in the train set the oracle function uses a k-dimensional tree
    to find the nearest address using the X and Y coordinates.
    (After feature engineering procedure the Police Department LHT has been not
    used in the prediction).
    Since zipping and uploading the solution
    takes time in trainMode we have a quick testing by dividing the the
    right solution over the total of crimes, just to have a sense if the
    algorithm is getting better """
    
    print "Computing Predictions..."
    tree, dx = getKDTree()
    predMat = np.zeros((test.shape[0],39))
    tester=0
    k=0
    for index, row in test.iterrows():
        dof = conv["dof"][row["DayOfWeek"]]
        #PD = conv["PD"][row["PdDistrict"]] #not used
        try:
            addr = conv["addr"][cornersHandler(row["Address"])]
            addressLine = LHT[6][addr]
        except: 
            B = tree.query([row["X"], row["Y"]])[1]
            #print "Crime",k ,":Address not found. Retriving it from the nearest point:", B
            addr = conv["addr"][cornersHandler(dx.ix[B, "Address"])]
            addressLine = LHT[6][addr]
        mat = np.matrix([LHT[0][index.hour-1], LHT[1][index.day-1], LHT[2][dof], LHT[3][index.month-1], LHT[4][index.year-2003], addressLine, LHT[7][index.minute-1]])
        aux = np.prod(mat, axis=0)
        predMat[k] = np.multiply(aux,probs)
        if trainMode and conv["inv"][np.argmax(predMat[k])]==row["Category"]:
            tester += 1
        k += 1
    if trainMode:
        print "Metrics for this model:", test.shape[0], tester, tester/float(test.shape[0])
    finale = pd.DataFrame(predMat, index = range(predMat.shape[0]), columns=crimeList)
    finale.index.name= "Id" 
    finale.to_csv("1494355.csv", sep=",")
    
        
def main():  
    """ Welcome procedure: it prompts the user asking if he wants to test the
    accuracy of the system or to produce a Kaggle Submission File. The
    boolean flag -trainMode- to save this choice during the computation """
    
    global conv
    global train
    global test
    global trainMode
    trainMode = raw_input("Hi Welcome to Aalterity, the crimes predictor. \nType t for testing, any key to generate kaggle submission: ") == "t"
    print "Train Mode:", trainMode     
    train, test = getCleanData()
    conv = getDics()
    oracle(computeLHTables(), np.load("data/probs.data"))
    
if __name__== "__main__":
    main()  
    
"""23228"""