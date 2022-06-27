# TensorFlow
This section of the repo houses the code from the TensorFlow implementation of the IQNs, as well as the training code and analysis code for the models.

## Train
This folder contains the files for training models with jet data. There are two mappings, which can be run as follows:

```
python genToRecoQuantile.py
```
or

```
python partonToGenQuantile.py
```


## Test
This folder contains the files for examining the performance of a trained model. The scripts are intended to be run as follows:
```
python runBatchPredictionsGenToReco.py
python basicDataExtractionGenToReco.py
python plotBasicDataGenToReco.py
python plotMarginalsGenToReco.py
```
