#Import all the required packages
import os
import datetime
import click
import matplotlib
import tables

import random as rn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy import stats
from math import pi

from quantileNetwork import QuantileNet, sample_net

fifthInput = True

modelName = "genToRecoMid"
if(fifthInput):
    modelName = "genToRecoMidPartonTypes"
sampleNum=1000
batchSize = 1000




#Set random number seeds
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)    



#Extract data
data = np.load("trainDataFlavor.npy").T
partonData = data[0:4,:]
genData = data[4:8,:]
recoData = data[8:12,:]
partonTypes = data[12:13,:]
genData[0,:] = np.log(genData[0,:])
genData[3,:] = np.log(genData[3,:]+2)
recoData[0,:] = np.log(recoData[0,:])
recoData[3,:] = np.log(recoData[3,:]+2)

inputData=genData
outputData=(recoData+10)/(inputData+10)

if(fifthInput):
    inputData=np.concatenate([genData, partonTypes], axis=0)

trainIn = inputData.T
trainOut = outputData.T

data = np.load("testDataFlavor.npy").T
partonData = data[0:4,:]
genData = data[4:8,:]
recoData = data[8:12,:]
partonTypes = data[12:13,:]
genData[0,:] = np.log(genData[0,:])
genData[3,:] = np.log(genData[3,:]+2)
recoData[0,:] = np.log(recoData[0,:])
recoData[3,:] = np.log(recoData[3,:]+2)

inputData=genData
outputData=(recoData+10)/(inputData+10)

if(fifthInput):
    inputData=np.concatenate([genData, partonTypes], axis=0)


testIn = inputData.T
testOut = outputData.T


normInfoOut=[[0,1],[0,1],[0,1],[0,1]]
normInfoIn=[[0,1],[0,1],[0,1],[0,1]]

for x in [0,1,2,3]:
    testIn[:,x]=(testIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
    
    normInfoIn[x] = [np.mean(trainIn[:,x]), np.std(trainIn[:,x])]

    normInfoOut[x] = [np.mean(trainOut[:,x]), np.std(trainOut[:,x])]



if(fifthInput):
    testIn[:,4]=(testIn[:,4]-np.mean(trainIn[:,4]))/(np.std(trainIn[:,4]))


filename = 'genToRecoFlavor.h5'
ROW_SIZE = testIn.shape[0]
NUM_COLUMNS = sampleNum*4
f = tables.open_file(filename, mode='w')
atom = tables.Float32Atom()
allData = f.create_earray(f.root, 'data', atom, (0, NUM_COLUMNS))


#Normalization 

model = QuantileNet(network_type="not normalizing") #Used to load network
newModel = tf.keras.models.load_model(modelName, model.custom_objects())

for y in range(0, testIn.shape[0], batchSize):
    currentTestIn = testIn[y:min(y+batchSize, testIn.shape[0]),:]
    currentTestIn=tf.transpose(currentTestIn)
    currentTestIn=tf.cast(currentTestIn, tf.float32)
    out=sample_net(newModel, #Network
                    sampleNum, #Number of samples
                    currentTestIn, #Input
                    currentTestIn.shape[1], #Number of examples (batch size)
                    currentTestIn.shape[0], #4d input
                    4, network_type="not normalizing") #4d output
    out = np.array(out)
    currentTestIn = np.array(currentTestIn).T
    currentTestIn = currentTestIn[:,:4]
    for x in range(4):
        out[:,:,x] = out[:,:,x]*normInfoOut[x][1]+normInfoOut[x][0]
        currentTestIn[:,x] = currentTestIn[:,x]*normInfoIn[x][1]+normInfoIn[x][0]
    for x in range(sampleNum):
        out[:,x,:] = out[:,x,:]*(10+currentTestIn)-10
    out[:,:,0] = np.exp(out[:,:,0])
    out[:,:,2] = out[:,:,2]%(2*pi)
    out[:,:,2] = np.float32(tf.where(out[:,:,2]>pi, out[:,:,2]-2*pi, out[:,:,2]))
    out[:,:,3] = np.exp(out[:,:,3])-2
    ranges=[[0,500],[-5,5],[-5,5],[0,500]]
    out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
    allData.append(out)
    percentDone=min(y+batchSize, testIn.shape[0])*100/testIn.shape[0]
    percentDone=round(percentDone,4)
    print("Processing is " + str(percentDone)+"% done")
