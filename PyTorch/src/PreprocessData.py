#!/usr/bin/env python
# coding: utf-8



import numpy as np
# import torch; import torch.nn as nn
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

df_columns=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam']


def convert_np_to_df(npy):
    npy=np.load(npy).T
    parton_Data=npy[0:4,:].T
    gen_Data=npy[4:8,:].T
    reco_Data=npy[8:12,:].T
    gen_reco_Data=np.concatenate([gen_Data,reco_Data],axis=1)
    df=pd.DataFrame(gen_reco_Data)
    return df


#100 K
data_100k_df=convert_np_to_df('data/smallData.npy')
data_100k_df.columns=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam']
data_100k_df['tau'] = np.random.uniform(0,1, size = data_100k_df.shape[0])
print('data_100k_df shape  after tau = ', data_100k_df.shape)

print()
print('data_100k_df')
print(data_100k_df.head())
data_100k_df.to_csv('data/data_100k.csv')







train_df=df=convert_np_to_df('data/trainFull_10M.npy')
# print('train_df shape', train_df.shape)




# plt.hist(train.iloc[:,0], label='gen $p_T$',bins=100,range=(0,100))
# plt.hist(df.iloc[:,4], label='reco $p_T$',bins=100,range=(0,100))
# plt.legend();plt.show()



train_df.columns=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam']
train_df['tau'] = np.random.uniform(0,1, size = train_df.shape[0])
print('train_df shape after tau = ', train_df.shape)




print('train df after tau')
print()
print(train_df)




train_df.to_csv('data/train_data_10M.csv')




test_df=df=convert_np_to_df('data/testFull_10M.npy')
print(test_df.shape)
test_df.columns=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam']
test_df['tau'] = np.random.uniform(0,1, size = test_df.shape[0])
print()
print('test_df shape after tau = ',test_df.shape)
print(test_df)
print()
test_df.to_csv('data/test_data_10M.csv')



validation_df=df=convert_np_to_df('data/validationFull_10M.npy')
print(validation_df.shape)
validation_df.columns=['genDatapT', 'genDataeta', 'genDataphi', 'genDatam','RecoDatapT', 'RecoDataeta', 'RecoDataphi', 'RecoDatam']
validation_df['tau'] = np.random.uniform(0,1, size = validation_df.shape[0])
print('\nvalidation_df after tau = ' ,  validation_df.shape)
print()
print(validation_df)

validation_df.to_csv('data/validation_data_10M.csv')
print()
print('all dataframes are saved in the data/ directory')





# data    = pd.read_csv('data/Data.csv')
# print('number of entries:', len(data))

# columns = list(data.columns)[1:]
# print('\nColumns:', columns)
# print()

# fields  = list(data.columns)[5:]
# print('fields',fields)





