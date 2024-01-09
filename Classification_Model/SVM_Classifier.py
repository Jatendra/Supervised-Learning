'''
File is created by Jatendra Gautam

Objective : Loan prediction by SVC Classifier

Will tune model with help of both gamma and C

Let's see how it solves imbalance data problem

'''

import os
import Utils

from IPython.display import display

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

path = r"C:\Users\jaten\Downloads\Sapient_Data\data\Refined_Dataset.csv"

## base model

def svc_base_model_train(X,Y):
    
    '''base model characteristics'''

    base_model = SVC(random_state=42)

    base_model.fit(X,Y)

    return base_model

'''Commonly used Hyperparamters in SVM Classifier

1) C : regularization parameter, inverse of lambda(l2), bigger c gives hard svm 
2) kernel : 'rbf', 'linear', 'poly', 'sigmoid'
3) gamma : kernel coff ('scale', 'auto')  
4) degree : degree of kernel('poly')
5) class_weight : dict, 'balanced'
6) decision_function_shape : 'ovo', 'ovr' 
7) tol : Tolerance for stopping criterion
8) probability : bool, must be called before .fit

'''


if __name__ == '__main__':

    input_data = Utils.load_data(path)
    
    # check the data size
    display(input_data.shape)

    # check data imbalance or not
    display(input_data.Default.value_counts()) 
    '''data is imbalanced, has to check how svm handles this'''

    ## Train Test Splits
    train_x, train_y, val_x, val_y, test_x, test_y = Utils.train_test_validation_splits(input_data,'Default',test_ratio=0.33,validation_ratio=0.0,imbalanced=True,remove_cols='ID')

    # check training size
    display(train_x.shape)

    # training of base model - SVC

    base_svc = svc_base_model_train(train_x,train_y)

    display(Utils.find_model_performance_sklearn(base_svc,test_x,test_y))


















