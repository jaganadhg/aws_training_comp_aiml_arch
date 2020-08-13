#!/usr/bin/env python

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.append("..")

from utils import (create_train_test_split,
                   generate_agg_feats,
                   fft_features,
                   plot_vibration_data)


@pytest.fixture
def get_traain_test_sp_data():
    """ Fixture for train test split"""
    #data/test_feat.csv
    data = pd.read_csv("../data/test_feat.csv")
    
    return data

@pytest.fixture
def raw_data():
    """ Fixture for fft and aggragate features"""
    #data/test_feat.csv
    data = pd.read_csv("../data/test_feats_raw.csv")
    
    return data

#

def test_train_test_split(get_traain_test_sp_data):
    """ Test the train test split"""
    train_test_split = create_train_test_split(get_traain_test_sp_data)
    
    assert "train" in list(train_test_split.keys())
    assert "test" in list(train_test_split.keys())
    assert "validate" in list(train_test_split.keys())
    
    assert type(train_test_split["train"]) == pd.DataFrame
    assert type(train_test_split["test"]) == pd.DataFrame
    assert type(train_test_split["validate"]) == pd.DataFrame
    
    assert train_test_split["train"].shape[0] > 1
    assert train_test_split["test"].shape[0] > 1
    assert train_test_split["validate"].shape[0] > 1

def test_fft_features(raw_data):
    """ Test FFT features"""
    fft_data = fft_features(raw_data)
    
    assert type(fft_data) == np.ndarray
    assert fft_data.shape[0] != 0

def test_agg_feat(raw_data):
    """ test the aggragte feature generation"""
    
    cols = ['S1_max', 'S2_max', 'S3_max', 'S4_max', 'S1_min', 'S2_min', 'S3_min',
       'S4_min', 'S1_mean', 'S2_mean', 'S3_mean', 'S4_mean', 'S1_std',
       'S2_std', 'S3_std', 'S4_std', 'S1_median', 'S2_median', 'S3_median',
       'S4_median', 'S1_skew', 'S2_skew', 'S3_skew', 'S4_skew']
    
    agg_feats = generate_agg_feats(raw_data)
    
    assert type(agg_feats) == pd.DataFrame
    
    assert list(agg_feats.columns) == cols
    
    assert agg_feats.shape[0] > 1
    
    

