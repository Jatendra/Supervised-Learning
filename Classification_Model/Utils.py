'''This file contains basic utility functions before ML Model'''

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def load_data(path):
    if path.split('.')[-1] == 'csv':
        data = pd.read_csv(path)
    elif path.split('.')[-1] == 'xlsx':
        data = pd.read_excel(path)
    # elif pd.read_html 
    # elif pd.read_json
    else :  data = pd.read_table(path)
    return data


def null_finding(data):
    df_null = data.isnull().sum().reset_index().sort_values(by=0,ascending=False).rename(columns={0:'count'})
    df_null['null_ratio'] = df_null['count']/data.shape[0]
    return df_null[['index','null_ratio']]


def train_test_validation_splits(data,target,test_ratio,validation_ratio,imbalanced,remove_cols):
    features = list(set(data.columns) - set([target]))
    features = list(set(features)- set(remove_cols))
    if (not bool(validation_ratio)) and imbalanced :
        train_x, test_x, train_y, test_y = train_test_split(data[features],data[target],test_size=test_ratio,random_state=42,stratify=data[target])
        val_x, val_y = 0,0
         
    elif not (bool(validation_ratio)) and (not imbalanced) :
        train_x, test_x, train_y, test_y = train_test_split(data[features],data[target],test_size=test_ratio,random_state=42)
        val_x, val_y = 0,0

    elif bool(validation_ratio) and (imbalanced) :
        X, test_x, Y, test_y = train_test_split(data[features],data[target],test_size=test_ratio,random_state=42,stratify=data[target])
        train_x, val_x, train_y, val_y = train_test_split(X,Y,test_size=validation_ratio,random_state=42,stratify=Y)
        
    else:
        X, test_x, Y, test_y = train_test_split(data[features],data[target],test_size=test_ratio,random_state=42)
        train_x, val_x, train_y, val_y = train_test_split(X,Y,test_size=validation_ratio,random_state=42)
    
    return train_x, train_y, val_x, val_y, test_x, test_y

