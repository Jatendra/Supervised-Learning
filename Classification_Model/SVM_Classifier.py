'''
File is created by Jatendra Gautam

Objective : Loan prediction by SVC Classifier

Will tune model with help of both gamma and C

Let's see how it solves imbalance data problem

'''

import os

os.chdir(r'C:\Users\jaten\OneDrive\Documents\GitHub\Supervised-Learning\Classification_Model')
os.getcwd()

import Utils

import numpy as np

from IPython.display import display
import logging
logging.basicConfig(level=logging.INFO)

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

path = r"C:\Users\jaten\Downloads\Sapient_Data\data\Refined_Dataset.csv"

## base model

def svc_base_model_train(X,Y):
    
    '''base model characteristics'''

    logging.info("base_model_initialization")

    base_model = SVC(probability=True,class_weight='balanced',random_state=42)

    base_model.fit(X,Y)

    logging.info("model fitting completed")

    return base_model

# base_model = svc_base_model_train()

## best model selection

def set_params_svc():
    
    '''Commonly used Hyperparamters in SVM Classifier

    1) C : regularization parameter, inverse of lambda(l2), bigger c gives hard svm 
    2) kernel : 'rbf', 'linear', 'poly', 'sigmoid'
    3) gamma : kernel coff ('scale', 'auto')  default = scale
    4) degree : degree of kernel('poly'), default =3
    5) class_weight : dict, 'balanced'
    6) decision_function_shape : 'ovo', 'ovr' - multi class 
    7) tol : Tolerance for stopping criterion
    8) probability : bool, must be called before .fit

    '''

    set_params = {
        'C' : np.random.randint(1,50,10),
        'kernel' : ['rbf','linear','poly','sigmoid'],
        'gamma' : ['auto','scale', 0.05, 0.1, 0.5],
        'degree' : [2,3,5,7,10],
        'tol' : [0.001,0.005,0.01]
    }

    return set_params



def best_svm_classifier_selection(params,X,Y,cv):

    logging.info('model_initialization')

    model_svc = SVC(probability=True,class_weight='balanced',random_state=42)

    kfold_cv = StratifiedKFold(n_splits=cv,shuffle=True,random_state=42)

    cv_results =   RandomizedSearchCV(model_svc,param_distributions=params,n_iter=10,\
                                      cv=kfold_cv,scoring='f1',error_score='raise',\
                                    return_train_score=True,random_state=42,n_jobs=-1,verbose=True)
    
    cv_results.fit(X,Y)

    logging.info('hyperparameter tuning ends')

    return cv_results
    

def main():

    input_data = Utils.load_data(path)
    
    # check the data size
    display(input_data.shape)

    # check data imbalance or not
    display(input_data.Default.value_counts()) 
    '''data is imbalanced, has to check how svm handles this'''

    # take sample data if training time is very high
    sample_data = input_data.sample(10000)

    ## Train Test Splits
    train_x, train_y, val_x, val_y, test_x, test_y = Utils.train_test_validation_splits(sample_data,'Default',test_ratio=0.33,validation_ratio=0.0,imbalanced=True,remove_cols='ID')

    # check training size
    display(train_x.shape)

    # training of base model - SVC

    base_svc = svc_base_model_train(train_x,train_y)

    display(Utils.find_model_performance_sklearn(base_svc,test_x,test_y))

    svc_cv_results = best_svm_classifier_selection(set_params_svc(),train_x, train_y,5)

    display(svc_cv_results.best_score_)

    best_svm_classifier = svc_cv_results.best_estimator_

    display(Utils.find_model_performance_sklearn(best_svm_classifier,test_x,test_y))


if __name__ == '__main__':
    main()



    















