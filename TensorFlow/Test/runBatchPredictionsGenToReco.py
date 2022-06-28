#Import all the required packages
import os
import click
import tables

import random as rn
import numpy as np
import tensorflow as tf

from math import pi

from quantileNetwork import QuantileNet, sample_net


@click.command()
@click.option('--trainDataName', default="trainData.npy", help='Name of train data file')
@click.option('--testDataName', default="testData.npy", help='Name of test data file')
@click.option('--networkName', default="genToReco", help='Name of network')
@click.option('--h5', default="genToReco.h5", help='Name of h5 file for data')
def main(traindataname, testdataname, networkname, h5):

    modelName = networkname
    sampleNum=1000
    batchSize = 1000
    
    
    #Set random number seeds
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    np.random.seed(42)
    rn.seed(12345)
    tf.random.set_seed(3)    
    
    #Extract data
    data = np.load(traindataname).T
    genData = data[4:8,:]
    recoData = data[8:12,:]
    genData[0,:] = np.log(genData[0,:])
    genData[3,:] = np.log(genData[3,:]+2)
    recoData[0,:] = np.log(recoData[0,:])
    recoData[3,:] = np.log(recoData[3,:]+2)
    
    inputData=genData
    outputData=(recoData+10)/(inputData+10)
    
    trainIn = inputData.T
    trainOut = outputData.T
    
    data = np.load(testdataname).T
    genData = data[4:8,:]
    recoData = data[8:12,:]
    genData[0,:] = np.log(genData[0,:])
    genData[3,:] = np.log(genData[3,:]+2)
    recoData[0,:] = np.log(recoData[0,:])
    recoData[3,:] = np.log(recoData[3,:]+2)
    
    inputData=genData
    outputData=(recoData+10)/(inputData+10)
    
    testIn = inputData.T
    normInfoOut=[[0,1],[0,1],[0,1],[0,1]]
    normInfoIn=[[0,1],[0,1],[0,1],[0,1]]
    
    for x in [0,1,2,3]:
        testIn[:,x]=(testIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
        
        normInfoIn[x] = [np.mean(trainIn[:,x]), np.std(trainIn[:,x])]
        trainIn[:,x]=(trainIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
        
        normInfoOut[x] = [np.mean(trainOut[:,x]), np.std(trainOut[:,x])]
    
    filename = h5
    NUM_COLUMNS = sampleNum*4
    f = tables.open_file(filename, mode='w')
    atom = tables.Float32Atom()
    allData = f.create_earray(f.root, 'data', atom, (0, NUM_COLUMNS))
    
    
    #Normalization 
    
    model = QuantileNet() #Used to load network
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
                        4) #4d output
        out = np.array(out)
        currentTestIn = np.array(currentTestIn).T
        for x in range(4):
            out[:,:,x] = out[:,:,x]*normInfoOut[x][1]+normInfoOut[x][0]
            currentTestIn[:,x] = currentTestIn[:,x]*normInfoIn[x][1]+normInfoIn[x][0]
        for x in range(sampleNum):
            out[:,x,:] = out[:,x,:]*(10+currentTestIn)-10
        out[:,:,0] = np.exp(out[:,:,0])
        out[:,:,2] = out[:,:,2]%(2*pi)
        out[:,:,2] = np.float32(tf.where(out[:,:,2]>pi, out[:,:,2]-2*pi, out[:,:,2]))
        out[:,:,3] = np.exp(out[:,:,3])-2
        out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
        allData.append(out)
        percentDone=min(y+batchSize, testIn.shape[0])*100/testIn.shape[0]
        percentDone=round(percentDone,4)
        print("Processing is " + str(percentDone)+"% done")
    f.close()

if __name__ == '__main__':
    main()
