# PyTorch Implementation

### Data Preprocessing

Before anything else, make sure you have generated the data. We assume that you have produced the files [`smallData.npy, testFull_10M.npy, trainFull_10M.npy, validationFull_10M.npy`] with the data generation process in the Tensorflow directory. Then copy these data in the `PyTorch/data` directory, and do

`python PreprocessData.py`

in the `PyTorch/` directory. This command runs preprocessing on the `*.npy` files in `data/` and produces dataframes with tau and column names from them for later use (you will get *.csv files in the `data/` directory, these files are not included in the repository since they are too large!)
Specifically, it produces [`data_100k.csv, train_data_10M.csv, test_data_10M.csv, validation_data_10M.csv`] in the `PyTorch/data` directory.

------

### Train

To train an IQN, do

`python RunTrainingGenToReco.py --T RecoDatapT`

Where `RecoDatapT` is the the $p_T$ of the reconstructed jet, as the target you want to predict. This is the only **required parameter** to train the IQN, and the available options for it are ['RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam']. 



Running `RunTrainingGenToReco.py` generates a dataframe of the predicted target in `predicted_data/` for later analysis. It is also possible to save the trained model parameters (in `src/trained_models/`), the loss curves (in `src/images/loss_curves`) and/or show these plots by varying the optional parameters. To see all the available parameters that you can specify, do

`python RunTrainingGenToReco.py --help`

The optional parameters are. 

  * `--T `:               the target that you want. Options: [RecoDatapT,
                        RecoDataeta, RecoDataphi, RecoDatam]

  * ` --N`:                  size of the dataset you want to use. Options are 10M
                        and 100K, the default is 10M
  * `--n_iterations`
                        The number of iterations for training, the default is
  * `--n_layers` :   The number of layers in your NN, the default is 5
  * `--n_hidden`:   The number of hidden layers in your NN, the default is
                        5
  * `--starting_learning_rate`:
                        Starting learning rate, the default is 10^-3
  * `--show_loss_plots`:
                        Boolean to show the loss plots, defialt is False.

  * `--save_model`:
                        Boolean to save the trained model dictionary

* `--save_loss_plots`: Boolean to save the loss plots, default is True.


For eaxample, you can train a fully-customized IQN by doing

`python RunTrainingGenToReco.py --T RecoDatapT --N 100K --n_iterations 500 --n_layers 6 --n_hidden 6 --starting_learning_rate 1.e4 --show_loss_plots True --save_model False`



If you want to train all 4 independent IQNs with the default parameters, go to the `src/` directory and do 

`bash train_all.sh`


--------

# Plotting your results

To plot the results that you attained from training your IQN(s), do

`python plot_results.py --T RecoDatapT`

Where `RecoDatapT` is the the $p_T$ of the reconstructed jet, as the target you want to predict. This is the only **required parameter** to train the IQN, and the available options for it are [`'RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam'`]. This plots histograms to compare the output of your models to the reco-jet distributions, and save those plots in the `images/` directory.

