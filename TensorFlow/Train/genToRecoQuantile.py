""" 
Training code for the raw to gen quantile network. This network is used to
go from a raw reco jet to a gen jet
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #supress tensorflow warning messages
warnings.filterwarnings("ignore", category=FutureWarning) 

import click

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import random as rn

from sklearn.model_selection import train_test_split

from quantileNetwork import QuantileNet, make_dataset

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)

@click.command()
@click.option('--hidden', default=5, help='Number of hidden layers')
@click.option('--width', default=50, help='Width of the hidden layers')
@click.option('--alpha', default=0.3, help='Slope for leaky relu')
@click.option('--initialLR', default=0.001, help='initial learning rate')
@click.option('--batch', default=512, help='batch size')
@click.option('--cycles', default=4, help='Number of cylces to train for')
@click.option('--epochs', default=100, help='Number of epochs in a cylce')
@click.option('--patience', default=100, help='Number of epochs with no improvement before ending')
@click.option('--trainDataName', default="trainData.npy", help='Name of train data file')
@click.option('--valDataName', default="valData.npy", help='Name of validation data file')
@click.option('--networkName', default="genToReco", help='Name of network')
def main(hidden, width, alpha, initiallr, batch, cycles, epochs, patience, traindataname, valdataname, networkname):
    data = np.load(traindataname).T
    partonData = data[0:4,:]
    genData = data[4:8,:]
    recoData = data[8:12,:]
    genData[0,:] = np.log(genData[0,:])
    genData[3,:] = np.log(genData[3,:]+2)
    recoData[0,:] = np.log(recoData[0,:])
    recoData[3,:] = np.log(recoData[3,:]+2)

    inputData=np.concatenate([genData, partonTypes], axis=0)
    outputData=(recoData+10)/(inputData[:4,:]+10)

    trainIn = inputData.T
    trainOut = outputData.T

    data = np.load(valdataname).T
    partonData = data[0:4,:]
    genData = data[4:8,:]
    recoData = data[8:12,:]
    genData[0,:] = np.log(genData[0,:])
    genData[3,:] = np.log(genData[3,:]+2)
    recoData[0,:] = np.log(recoData[0,:])
    recoData[3,:] = np.log(recoData[3,:]+2)

    inputData=np.concatenate([genData, partonTypes], axis=0)
    outputData=(recoData+10)/(inputData[:4,:]+10)

    valIn = inputData.T
    valOut = outputData.T


    #trainIn, testIn, trainOut, testOut = train_test_split(trainIn,
    #                                                    trainOut,
    #                                                    test_size=1/3,
    #                                                    random_state=42)
    normInfoOut=[[0,1],[0,1],[0,1],[0,1]]
    normInfoOut=[[0,1],[0,1],[0,1],[0,1]]
    print(trainIn.shape, valIn.shape, trainOut.shape, valOut.shape)

    for x in [0,1,2,3]:
        valIn[:,x]=(valIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
        trainIn[:,x]=(trainIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
        normInfoOut[x] = [np.mean(trainOut[:,x]), np.std(trainOut[:,x])]
        valOut[:,x]=(valOut[:,x]-np.mean(trainOut[:,x]))/(np.std(trainOut[:,x]))

        trainOut[:,x]=(trainOut[:,x]-np.mean(trainOut[:,x]))/(np.std(trainOut[:,x]))
    

    trainIn, trainOut = make_dataset(trainIn, #input x
          trainOut, #input y
          4, # x dims
          4, # y dims
          trainIn.shape[0]) # examples
    
    valIn, valOut = make_dataset(valIn, #input x
          valOut, #input y
          4, # x dims
          4, # y dims
          valIn.shape[0]) # examples
    #print(x_val.shape, y_val.shape)
    #trainIn, valIn, trainOut, valOut = train_test_split(x_val,
    #                                                    y_val,
    #                                                    test_size=1/10,
    #                                                    random_state=42)

    print(trainIn.shape, valIn.shape, trainOut.shape, valOut.shape)
    model = QuantileNet(network_type="not normalizing")

    model.add(
        tf.keras.layers.Dense(
            width,
            kernel_initializer="glorot_uniform",
            activation=None
            ))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
    

    for n in range(hidden - 1):
        model.add(
            tf.keras.layers.Dense(
                width,
                kernel_initializer="glorot_uniform",
                activation=None))
        model.add(tf.keras.layers.LeakyReLU(alpha=alpha))

    model.add(
        tf.keras.layers.Dense(
            1,
            kernel_initializer="glorot_uniform",
            activation=None))
   
    callbackMetric="val_loss"
    callback = tf.keras.callbacks.EarlyStopping(
            monitor=callbackMetric, patience=patience, restore_best_weights=True)
    trainOut = tf.expand_dims(trainOut,1)
    valOut = tf.expand_dims(valOut,1)

    epochList=[]
    trainLossList=[]
    valLossList=[]
    for x in range(cycles):
        model.compile(optimizer=tf.keras.optimizers.Adam(initiallr * (10**(-x)),
                      amsgrad=True),
                      loss=model.loss,
                     run_eagerly=False)


        history = model.fit(
            trainIn,
            trainOut,
            validation_data=(
                valIn,
                valOut),
            epochs=epochs,
            batch_size=batch,
            verbose=2,
            callbacks=[callback])#, callback2])
        
        trainLoss = history.history["loss"]
        valLoss = history.history["val_loss"]
        epochList.append(len(trainLoss))
        trainLossList.append(trainLoss)
        valLossList.append(valLoss)
        #Save the network
        model.save(networkname, save_traces=False)
        
        
        
    epochList = range(1,sum(epochList)+1)
    trainLoss = np.concatenate(trainLossList, axis=0)
    valLoss = np.concatenate(valLossList, axis=0)
    plt.figure()
    plt.plot(epochList[1:], trainLoss[1:], label="trainLoss")
    plt.plot(epochList[1:], valLoss[1:], label="valLoss")
    plt.legend()
    plt.savefig(networkname+"_loss_curve.png")
    plt.close()


    plt.figure()
    plt.plot(epochList[1:], np.log(trainLoss[1:]), label="trainLoss")
    plt.plot(epochList[1:], np.log(valLoss[1:]), label="valLoss")
    plt.legend()
    plt.savefig(networkname+"_log_loss_curve.png")
    plt.close()


if __name__ == '__main__':
    main()
