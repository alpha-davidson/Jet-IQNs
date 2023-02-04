# TensorFlow
This section of the repo houses the code from the TensorFlow implementation of the IQNs, as well as the training code and analysis code for the models. To install the necesary pacakges run the following command:
```
pip install click matplotlib numpy sklearn tables tensorflow
```

The code was tested with the following versions of pacakges:

Python: 3.6.8
Click: 8.0.4
Matplotlib: 3.3.4
Numpy: 1.19.5
Sklearn: 0.24.2
Tables: 3.7.0
TensorFlow: 2.4.0

Note that the IQN implementation as used in this repo does not work with current versions of TensorFlow as it does not save properly. A modified version of the code is included in the IQN folder (quantileNetRevised.py) which does properly save the network. It has a few internal adjustments in how the data is stored, but is functionally equivalent to the other version of the code.


## Train
This folder contains the file for training models with jet data. This can be run simply by
```
python genToRecoQuantile.py
```

There are many command line arguments from click for this training script. They are as follows.

- hidden: Number of hidden layers, defualt is 5
- width: Width of the hidden layers, default is 50
- alpha: Slope for leaky relu, default is 0.3
- initialLR: initial learning rate, default is 1e-3
- batch: batch size, default is 512
- cycles: Number of cylces to train for, default is 4
- epochs: Number of epochs in a cylce, default is 100
- patience: Number of epochs with no improvement before ending, default is 100
- trainDataName: Name of train data file, default is trainData.npy
- valDataName: Name of validation data file, default is valData.npy
- networkName: Name of network, default is genToReco
- target: Selects the network target 0:Pt, 1:Eta, 2:Phi, 3:Mass, 4:All, default is 4.


## Test
This folder contains the files for examining the performance of a trained model. The scripts are intended to be run as follows:
```
python runBatchPredictionsGenToReco.py
python basicDataExtractionGenToReco.py
python plotBasicDataGenToReco.py
python plotMarginalsGenToReco.py
```

There are many command line arguments from click for these scripts. They are as follows and are the same for all the scripts.

- trainDataName: Name of train data file, default is trainData.npy
- testDataName: Name of test data file, default is testData.npy
- networkName: Name of network, default is genToReco
- h5: Name of h5 file for data, default is genToReco.h5

## IQN
This folder contains files to do some simple examples with the IQN on known distributions. See the README in the folder for additional information and some general discussion on IQNs.

