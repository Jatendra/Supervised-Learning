'''This module covers the model building of XGBoost Classifier
@ author : Jatendra Gautam
on 24th Feb 2023
'''

import pandas as pd
import numpy as np

from IPython.display import display
import matplotlib.pyplot as plt

import os
os.getcwd()

os.chdir(r'C:\Users\jaten\OneDrive\Documents\GitHub\Supervised-Learning\Classification_Model')

import Utils

import xgboost as xgb
from sklearn.metrics import precision_recall_curve, \
    PrecisionRecallDisplay,recall_score, precision_score,classification_report,\
        roc_auc_score, roc_curve

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

## Load Data

input_data = Utils.load_data(r"C:\Users\jaten\Downloads\Sapient_Data\data\Refined_Dataset.csv")

display(input_data.shape)

display(input_data.head())

display(Utils.null_finding(input_data).head(10))


# check data types inconsistency
# missing values
# check for outliers
# check duplicates

def outlier_treatment(data):
    '''
    Decision Making:

    Consider the impact on analysis: Assess how outliers affect model outcomes and conclusions.
    Data size and context: Smaller datasets are more sensitive to outlier removal.
    Outlier cause: Understand if outliers are errors, true extreme values, or influential points.
    Balance accuracy and information loss: Avoid sacrificing valuable data while ensuring reliable results. 
    
    Using IQR method 
    max + 1.5 *IQR
    min - 1.5* IQR
    for each col
    replace each outlier value to mean/median/mode/upper/lower
    or remove them 
    '''
    return None

def missing_value_imputation(data):
    ''' 
    Choosing the Best Technique:

    Consider the nature of the data (numerical, categorical, time series).
    Analyze the pattern of missingness (random, non-random).
    Evaluate the impact of imputation on subsequent analysis.
    Experiment with different techniques to find the most suitable one.    
            
    Using multiple mathods
    1) mean/median/mode
    2) knn imputation
    3) double layer : replace null to mean value and create new col to assign that there was null value earlier
    4) deletion
    5) intepolation
    6) prediction model to compute missing values (reg, tree)
    7) arbitatory value to identity (not a good method)
    8) forward fill, backward fill   
    '''
    return None

check_duplicate_records = input_data.duplicated()
check_duplicate_records

input_data = input_data.drop_duplicates()
'''
short comings
data : basic anomaly, missing value, distribution
model : choice, learning rate, overfitting/underfitting, 
features : encoding, feature selection, 
performance, event rate : 
'''

'''
model performance : overfitting vs underfitting (train error/validation vs interations) : how much iteration are required to get the minima 
'''

train_x, train_y, val_x, val_y, test_x, test_y = Utils.train_test_validation_splits(input_data,'Default',0.25,0.0,True,['ID'])

'''
This model is based on XGBoost Library
'''
def Dmatrix(X,Y,x,y):
    dtrain = xgb.DMatrix(X,Y)
    dtest = xgb.DMatrix(x,y)
    
    return dtrain, dtest

dtrain, dtest = Dmatrix(train_x,train_y,test_x,test_y)

def set_params(**kwargs):
    params = {'objective' : 'binary:logistic',
        'eval_metric' : 'logloss',
        'seed' : 42
        }
    
    return params

def xgb_base_model_train_v1(X,**kwargs):
    '''add other hyperparameter in dynamic way 
    for example : n_estimators, learning rate, objective, split criteria'''
    
    base_model = xgb.train(params=set_params(),dtrain=X,num_boost_round=100,verbose_eval=1)

    return base_model

xgb_base= xgb_base_model_train_v1(dtrain)

def find_model_performance_v1(trained_model,dtest,y):
    '''Can add more performance metrics if required 
    right now I will cover PR Curve, ROC AUC, Classification Report'''
    
    y = y
    pred_proba = trained_model.predict(dtest)
    pred = (pred_proba>0.5).astype(int)
    
    # classification report - precision, recall, f1 score, weighted accuracy
    print(classification_report(y,pred))
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)

    # auc score
    auc_value = roc_auc_score(y, pred_proba)
    print("AUC SCORE : {}".format(auc_value))
    
    # plot roc-auc curve
    tpr, fpr, thresholds = roc_curve(y, pred_proba)
    
    plt.plot(fpr,tpr,marker=".")
    plt.plot(tpr,tpr,marker="_")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    
    # plot PR Curve
    precision, recall, _ = precision_recall_curve(y, pred_proba)
    dis_plot = PrecisionRecallDisplay(precision=precision, recall=recall)
    dis_plot.plot()
    plt.show()
    
    return prec,rec,auc_value


find_model_performance_v1(xgb_base,dtest,test_y)

# -------------------------------------------------------------------------------------------------------
####################################### Using Another Library ###########################################
# -------------------------------------------------------------------------------------------------------

'''
This model is based on Sklearn Wrapper Library - XGBClassifier
'''

def xgb_base_model_train_v2(X,Y,x,y,**kwargs):
    '''add other hyperparameter in dynamic way 
    for example : n_estimators, learning rate, objective, split criteria'''
    
    base_model = xgb.XGBClassifier(objective='binary:logistic',max_depth= 10,
                                    n_estimators = 100,seed=42)
    # base_model.get_params()

    base_model.fit(X,
             Y,
             verbose=1,
             early_stopping_rounds=10,
             eval_metric='logloss',
             eval_set=[(x,y)])

    return base_model

xgb_base_v2 = xgb_base_model_train_v2(train_x,train_y,test_x,test_y)


def find_model_performance_sklearn(trained_model,x,y):
    '''Can add more performance metrics if required 
    right now I will cover PR Curve, ROC AUC, Classification Report'''
    
    y = y
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


# get performance matrics for sklearn model

find_model_performance_sklearn(xgb_base_v2,test_x,test_y)

'''
Objective candidate: rank:ndcg
Objective candidate: rank:pairwise
Objective candidate: rank:map
Objective candidate: survival:aft
Objective candidate: binary:hinge
Objective candidate: multi:softmax
Objective candidate: multi:softprob
Objective candidate: reg:quantileerror
Objective candidate: reg:squarederror
Objective candidate: reg:squaredlogerror
Objective candidate: reg:logistic
Objective candidate: binary:logistic
Objective candidate: binary:logitraw
Objective candidate: reg:linear
Objective candidate: reg:pseudohubererror
Objective candidate: count:poisson
Objective candidate: survival:cox
Objective candidate: reg:gamma
Objective candidate: reg:tweedie
Objective candidate: reg:absoluteerror
'''

'''
objective = learning task & objective function (see above list @227)
base_score = base score value for classifier, 0.5
booster = type of boosting model, gbtree, gblinear, dart(dropouts meet multiple additive regression trees)
callbacks = list of callback function to call while training
colsample_bylevel = # random cols for each split during level of tree
colsample_bynode =  # random cols for each split at tree node
colsample_bytree = kinda <max_features>
device = CPU, GPU
early_stopping_rounds = if eval metric won't stop till this
enable_categorical = flag to indicate to enable categorical features
eval_metric = logloss, error
feature_types = list to specify type of features c : categorical, q : quantitaitve, i = integer
gamma = min loss reduction required to make a further partition on leaf node, max gain required to split further
grow_policy - USELESS (method used to grow trees,eg depthwise,lossguide)
importance_type = method to calculate feature importance score (eg weight, gain)
interaction_constraints = USELESS (constraints for interaction groups)
learning_rate = step size
max_bin = max numbers of bins to bucket the feature values in
max_cat_threshold = max numbers of categories allowed to categorical features
max_cat_to_onehot = max no of categories to consider converting to one hot encoding
max_delta_step = controls the learning rate in optimzation
max_depth = max depth of individual tree
max_leaves = max no of terminal nodes in a tree
min_child_weight = hessian weights within child nodes
missing = a value to be treated as missing data
monotone constraints = USELESS (list of constraints for the monotonicity of feature)
multi_strategy = multiclass classification, "onevsrest" or "onevsall"
n_estimators = total no of boosting rounds(trees) to train
n_jobs = no of parallel threads to use tree construction
num_parallel_tree = no of parallel trees to build in the ensemble
random_state = seed value
reg_alpha = L1 regularization parameter
reg_lambda = L2 regularization parameter
sampling_method = method used to sampling data when building tree (eg hist, gpu_hist), it can take values like uniform or gradient based
scale_pos_weight = high imp to minority class in case of imbalance data
subsample = fraction of training data to randomly sample during boosting round, overfittig
tree_method = specifies method to use for constructing trees - 'auto', 'exact', 'approx', 'hist'
validate_parameters =  raise flag if any error in hyperparameters
verbosity = 0-silent, 1- warning, 2- info
seed - similar to random_state
'''

############### Model Selection #####################

'''
Rather than tuning all hyper paramters, we can tune few imp paramters first
This includes tuning one, two most imp hyperparameters
Later tuning other 6-8 hyper parameters
'''

learning_rate(eta)
n_estimators (num_boost_round)
max_depth
min_child_weight
subsample
colsample_bytree
gamma(min_split_loss)
reg_lambda
reg_alpha
scale_pos_weight


# np.random.random_sample(4) -> [0,1]; size : height,row,column/row,column/column

'''can't use randomized search cv for xgboost library directly, 
if you want to use just follow for loop for random choice of parameters and run it.
'''

# using randomized search for sklearn wrapper model

def set_tuning_params_v1(**kwargs):
    params = {'objective':['binary:logistic','reg:logistic'],
              'eval_metric':['error','logloss'],
              'learning_rate': [0.01,0.1,0.3,0.5]}
    
    return params


def xgb_classifier_best_model(params,X,Y,**kwargs):

    '''
    xgb.xgbclassifier is being used here to make it less complex.
    will append the parameters sequentially in order to speed up the process.
    '''

    xgb_model = xgb.XGBClassifier(random_state=42)
    
    cv = StratifiedKFold(n_splits=6,shuffle=True,random_state=42)
    xgb_cv = RandomizedSearchCV(xgb_model,param_distributions=params,n_iter=10,scoring='recall',n_jobs=-1,cv=cv,verbose=3,error_score='raise',random_state=42)
    xgb_cv.fit(X,Y)
    
    return xgb_cv


xgb_cv_results = xgb_classifier_best_model(params=set_tuning_params_v1(),X=train_x,Y=train_y)

xgb_cv_results.best_score_

best_xgb_classifier= xgb_cv_results.best_estimator_

# get performance matrics for tuned model

find_model_performance_sklearn(best_xgb_classifier,test_x,test_y)


# took a sample to check how kwargs argument works, args is simple : positional arguments

def sample(**kwargs):
    for key, value in kwargs.items():
        if key == 'first' : a= value
        else : b= value
    return a+b    


sample(first=1,second=2)


'''
tuned model = 1/0
tuned =0 : simple fit with some parameters
tuned =1 : cv + fit with parameters
'''