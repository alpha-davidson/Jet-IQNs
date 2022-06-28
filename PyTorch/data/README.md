make sure you have generated the data. We assume that you have produced the files [`smallData.npy, testFull_10M.npy, trainFull_10M.npy, validationFull_10M.npy`] with the data generation process in the Tensorflow directory. Then copy these data in the this directory, and do

`python PreprocessData.py`

in the `PyTorch/` (previous) directory. This command produces [`data_100k.csv, train_data_10M.csv, test_data_10M.csv, validation_data_10M.csv`] in the this directory.


`python PreprocessData.py`

This command runs preprocessing on the files in this and produces dataframes with tau and column names from them for later use (you will get *.csv files in the this directory, these files are not included in the repository since they are too large!)
