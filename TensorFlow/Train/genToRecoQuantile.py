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
from tools import extract

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)


@click.command()
@click.option('--hidden', default=5, help='Number of hidden layers')
@click.option('--width', default=50, help='Width of the hidden layers')
@click.option('--alpha', default=0.2, help='Slope for leaky relu')
@click.option('--initialLR', default=0.001, help='initial learning rate')
@click.option('--batch', default=512, help='batch size')
@click.option('--cycles', default=4, help='Number of cylces to train for')
@click.option('--epochs', default=100, help='Number of epochs in a cylce')
@click.option('--patience', default=100, help='Number of epochs with no improvement before ending')
@click.option('--dataName', default="smallData.npy", help='Name of input data file')
@click.option('--networkName', default="genToReco2Scale", help='Name of network')
@click.option('--scale', default=100, help='loss scale for gradient loss')
@click.option('--target', default=4, help='0:Pt, 1:Eta, 2:Phi, 3:Mass, 4:All')
def main(hidden, width, alpha, initiallr, batch, cycles, epochs, patience, dataname, networkname, scale, target):
    networkname+="_scale_"+str(scale)+"_target_"+str(target)

    # Extract, scale, and normalizeData
    trainIn, trainOut, valIn, valOut, normInfoIn, normInfoOut = extract("trainData2.npy", "validationData2.npy", singleTarget=target)

    # Make the quantile datasets
    trainIn, trainOut = make_dataset(trainIn, #input x
                                     trainOut, #input y
                                     trainIn.shape[1], # x dims
                                     trainOut.shape[1], # y dims
                                     trainIn.shape[0]) # examples
    
    valIn, valOut = make_dataset(valIn, #input x
                                 valOut, #input y
                                 valIn.shape[1], # x dims
                                 valOut.shape[1], # y dims
                                 valIn.shape[0]) # examples

    # Define the quantile network
    model = QuantileNet(network_type="not normalizing", grad_loss_scale=scale)

    # Add the first layer and activation function
    model.add(
        tf.keras.layers.Dense(
            width,
            kernel_initializer="glorot_uniform",
            activation=None
            ))
    model.add(tf.keras.layers.LeakyReLU())
    
    # Add the hidden layers and activation functions
    for n in range(hidden - 1):
        model.add(
            tf.keras.layers.Dense(
                width,
                kernel_initializer="glorot_uniform",
                activation=None))
        model.add(tf.keras.layers.LeakyReLU())

    # Add the output layer with no activation
    model.add(
        tf.keras.layers.Dense(
            1,
            kernel_initializer="glorot_uniform",
            activation=None))

    # Watch the validation loss, restore best weights based on if after each
    # learning rate
    callbackMetric="val_loss"
    callback = tf.keras.callbacks.EarlyStopping(
            monitor=callbackMetric, patience=patience, restore_best_weights=True)

    epochList=[]
    trainLossList=[]
    valLossList=[]

    print(trainIn.shape, valIn.shape, trainOut.shape, valOut.shape)

    # Go through a series of decreasing learning rates keeping the best model
    # from the previous set
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

        # Record some variables for plotting
        trainLoss = history.history["loss"]
        valLoss = history.history["val_loss"]
        epochList.append(len(trainLoss))
        trainLossList.append(trainLoss)
        valLossList.append(valLoss)

        #Save the network
        model.save(networkname, save_traces=False)

    # Save some loss curves
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
