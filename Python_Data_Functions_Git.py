# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:32:05 2023

@author: jaten
"""

import warnings
warnings.filterwarnings('ignore')

def null_finding(data):
    df_null = data.isnull().sum().reset_index().sort_values(by=0,ascending=False)
    df_null['null_ratio'] = df_null[0]/data.shape[0]
    return df_null[['index','null_ratio']]


from sklearn.model_selection import train_test_split
    
def train_test_validation_splits(data,target,test_ratio,validation_ratio,imbalanced):
    features = list(set(data.columns) - set([target]))
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



import pandas as pd

input_data = pd.read_csv(r"C:\Users\jaten\Downloads\Sapient_Data\data\Refined_Dataset.csv")

del input_data['ID']


train_x, train_y, val_x, val_y, test_x, test_y = train_test_validation_splits(input_data,'Default',0.33,0.0,True)
   
from sklearn.ensemble import GradientBoostingClassifier as gbc

def gbc_base_model_train(X,Y,**kwargs):
    '''add other hyperparameter in dynamic way 
    for example : n_estimators, learning rate, objective, split criteria'''
    
    base_model = gbc(random_state=42)
    
    base_model.fit(X,Y)
    
    return base_model


model = gbc_base_model_train(train_x, train_y)

'''
add hyperparamter tuning
grid search/ random search
'''

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, roc_curve, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score

''' we can modify the below function to include regression metrics as well'''

def find_model_performance(trained_model,x,y):
    '''Can add more performance metrics if required 
    right now I will cover PR Curve, ROC AUC, Classification Report'''
    
    pred = trained_model.predict(x)
    pred_proba = trained_model.predict_proba(x)
    
    # classification report - precision, recall, f1 score, weighted accuracy
    print(classification_report(y,pred))
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)

    # auc score
    auc_value = roc_auc_score(y, pred_proba[:,1])
    print("AUC SCORE : {}".format(auc_value))
    
    # plot roc-auc curve
    tpr, fpr, thresholds = roc_curve(y, pred_proba[:,1])
    
    plt.plot(fpr,tpr,marker=".")
    plt.plot(tpr,tpr,marker="_")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    
    # plot PR Curve
    precision, recall, _ = precision_recall_curve(y, pred_proba[:,1])
    dis_plot = PrecisionRecallDisplay(precision=precision, recall=recall)
    dis_plot.plot()
    plt.show()
    
    return prec,rec,auc_value
    
    
precision,recall,auc_value = find_model_performance(model,test_x,test_y)


import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

def gbc_best_model_selection(X,Y):
    ''' currently using randomized search '''
    
    base_model = gbc(random_state=42)

    custom_params = {'n_estimators':np.random.randint(100,1000,10),
                    'loss' : ['log_loss','exponential'],
                    'learning_rate' : np.arange(.01,.1,.01),
                    'subsample' : np.linspace(.8,1,5),
                    'max_depth' : np.random.randint(3,32,10),
                    'criterion' : ['friedman_mse', 'squared_error'],
                    'min_samples_leaf' : np.arange(1,5),
                    'min_weight_fraction_leaf' : np.linspace(0.05,0.2,5),
                    'max_features' : ['sqrt','log2', None],
                    'min_impurity_decrease' : np.linspace(0.1,1,4),
                    'validation_fraction' : np.linspace(0.05,0.2,5),
                    'n_iter_no_change' : np.random.randint(2,10,5)
                    }

#    scoring = {'precision' : make_scorer(precision_score),'recall' : make_scorer(recall_score)}
    fold = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)

    refined_model = RandomizedSearchCV(estimator = base_model,param_distributions = custom_params ,n_iter=5,scoring = 'recall',cv = fold,\
                      n_jobs=-1,verbose=1,random_state=42,error_score='raise')

    refined_model.fit(X,Y)

    return refined_model

best_gbc = gbc_best_model_selection(train_x,train_y).best_estimator_
                                    
best_gbc
                                   
precision,recall,auc_value = find_model_performance(best_gbc,test_x,test_y)

    
    

    
    
    
    
    
    
    
    
    
    
    


    
    