#!/usr/bin/env python

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm



fs = 5 #sampling frequency
fmax = 25 #sampling period
dt = 1/fs #length of signal
n = 75

def plot_vibration_data(accelaration_df : pd.DataFrame, title : str) -> None:
    """ Plot the vibration data
        :params accelaration_df: accelaration data for one id
        :params title: string
    """
    features = ["S1","S2","S3", "S4"]
    fig = plt.figure(figsize=(10,6))
    #fig.tight_layout(pad=30.0)
    #fig.suptitle(title)
    
    for idx,feature in enumerate(features):
        ax = fig.add_subplot(2,2,idx+1)
        accelaration_df[feature].plot(kind='line',
                                     title = "{0} Parameter {1}".format(title,feature),
                                     ax=ax)
    fig.tight_layout()

    


def fft_features(data_set : pd.DataFrame) -> np.ndarray:
    """ Convert the dataset to fourier transfomed
        :params data_set: original collider params data
        :returns ft_data: Fourier transformed data
        #Reference - https://dacon.io/competitions/official/235614/codeshare/1174
    """
    ft_data = list()
    
    features = ["S1","S2","S3", "S4"]
    
    id_set = list(data_set.id.unique())
    
    for ids in tqdm(id_set):
        s1_fft = np.fft.fft(data_set[data_set.id==ids]['S1'].values)*dt
        s2_fft = np.fft.fft(data_set[data_set.id==ids]['S2'].values)*dt
        s3_fft = np.fft.fft(data_set[data_set.id==ids]['S3'].values)*dt
        s4_fft = np.fft.fft(data_set[data_set.id==ids]['S4'].values)*dt
        
        ft_data.append(np.concatenate([np.abs(s1_fft[0:int(n/2+1)]),
                                       np.abs(s2_fft[0:int(n/2+1)]),
                                       np.abs(s3_fft[0:int(n/2+1)]),
                                       np.abs(s4_fft[0:int(n/2+1)])]))
    
    return np.array(ft_data)


def generate_agg_feats(data_set : pd.DataFrame) -> pd.DataFrame:
    """ Create aggrage features from the data
        :param data_set: Base data as DataFrame
        :returns agg_data: Aggragated DataFrame
    """
    
    max_feats = data_set.groupby(['id']).max().add_suffix('_max').iloc[:,1:]
    min_feats = data_set.groupby(['id']).min().add_suffix('_min').iloc[:,1:]
    mean_feats = data_set.groupby(['id']).mean().add_suffix('_mean').iloc[:,1:]
    std_feats = data_set.groupby(['id']).std().add_suffix('_std').iloc[:,1:]
    median_feats = data_set.groupby(['id']).median().add_suffix('_median').iloc[:,1:]
    skew_feats = data_set.groupby(['id']).skew().add_suffix('_skew').iloc[:,1:]
    
    agg_data = pd.concat([max_feats,min_feats,
                          mean_feats,std_feats,
                          median_feats,skew_feats],
                        axis=1)
    
    return agg_data

def create_train_test_split(data_set : pd.DataFrame) -> dict:
    """ Create a Training, Test and Validation Split from the data
        :params data_set; Dataset for splitting
        :returns data_split: training testing and validation as list of DataFrames
        Reference - https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
    """
    train, validate, test = np.split(data_set.sample(frac=1),
                                     [int(.6*len(data_set)),
                                      int(.8*len(data_set))])
    
    return {'train':train, 'test':test,'validate':validate}

if __name__ == "__main__":
    print()