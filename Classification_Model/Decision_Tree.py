# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 21:51:05 2023

@author: jaten
"""

# This code contains decision tree model

'''
Decision tree is a supervised learning algorithm that predicts the new labels 
by recursively splitting the predictor space into non overlapping regions. 

The tree is constructed through a greedy top down approach. They work for both 
regression and classification problems.

'''

'''
Steps to follow 

* Import Library
* Read Data
* Data sanity - data type check
* Basic cleaning -  imputation of missing values
* Train Test Split
* Base line model
* Choose best Model - hyperparamter tuning
* Performnace Matrices
'''


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV,\
    RandomizedSearchCV, StratifiedKFold
     
from sklearn.metrics import make_scorer, precision_recall_fscore_support, precision_recall_curve, \
    PrecisionRecallDisplay,recall_score, precision_score,classification_report, f1_score,\
        roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, roc_curve
        
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
                       
from sklearn.compose import ColumnTransformer



## Load data

Path = r"C:\Users\jaten\Downloads\Sapient_Data\data\Dataset.csv"

Total_Data = pd.read_csv(Path,low_memory=False)

## Initial Data checks

print(round(Total_Data.Default.value_counts()*100/len(Total_Data)))

# 8% - minortiy class 

'''Data is Imbalanced'''


## Data Cleaning

for col in ['Client_Income','Credit_Amount','Loan_Annuity','Population_Region_Relative','Age_Days','Employed_Days','Registration_Days','ID_Days','Score_Source_3']:
  Total_Data[col] = pd.to_numeric(Total_Data[col],errors='coerce')

print(Total_Data.info())

print("________________________________________________")

print(Total_Data.head(10))


## Null Impuation


null_check = Total_Data.isnull().sum()
null_cols = null_check[null_check>0].index.to_list()

for col in null_cols:
  if Total_Data[col].dtypes == 'object':
    Total_Data[col].fillna(Total_Data[col].mode()[0],inplace=True)
  else :   Total_Data[col].fillna(Total_Data[col].median(),inplace=True)

print(Total_Data.info())


## Data Visualization

plt.figure(figsize=(30,20))
sns.heatmap(Total_Data.select_dtypes(exclude='object').corr(),annot=True)
plt.show()
        

for col in Total_Data.columns:
  if Total_Data[col].dtypes == 'object':
    print(Total_Data[col].value_counts())
    print("________________________________________")

    

# One hot encoding for categorical variables

Total_Data = pd.get_dummies(Total_Data,drop_first=True)


## Train Test Split

x = Total_Data.drop(['Default'],axis=1)
y = Total_Data['Default']


# Train test split

train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.25,random_state=42,stratify=y)


## Model Building
    
def DT_base_model_training(X,Y,x,y):
    
        model = DecisionTreeClassifier(random_state=42)
          
        model.fit(X,Y)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        precision, recall, _ = precision_recall_curve(Y, model.predict_proba(X)[:,1])
        disp1 = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp1.plot(ax=axes[0])
        axes[0].set_title('Precision-Recall Curve - Training Data')
        
        precision2, recall2, _ = precision_recall_curve(y, model.predict_proba(x)[:,1])
        disp2 = PrecisionRecallDisplay(precision=precision2, recall=recall2)
        disp2.plot(ax=axes[1])
        axes[1].set_title('Precision-Recall Curve - Test Data')
        
        plt.tight_layout()
        plt.show()
        
        print("Precision on Test data is {}".format(precision_recall_fscore_support(y, model.predict(x))[0]))
        print("Recall on Test data is {}".format(precision_recall_fscore_support(y, model.predict(x))[1])) 
        
        return model  

trained_base_model = DT_base_model_training(train_x,train_y,test_x,test_y)



def Prediction(trained_model,X,Y,x,y):
    # fitted model
    model = trained_model
    
    
    # prediction
    
    y_pred = model.predict(x)
    y_pred_proba = model.predict_proba(x)[:,1]
    y_true = y
    
    return y_true, y_pred, y_pred_proba

''' Apply training and prediction function'''

y_true, y_pred, y_pred_proba =  Prediction(trained_base_model, \
                                          train_x, train_y, test_x, test_y)


    
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

print("AUC Score Base DT Model : {}".format(roc_auc_score(y_true, y_pred_proba)))


# Hyperparamter Tuning

'''

Decision Tree has multiple hyperparamters which define the construction of 
the tree to predecit though it has chance of overfitting which needs to be maintained

criterian - gini, entropy, log loss
splitter - best, random
max depth
min_samples_leaf - 
min_weight_fraction_leaf -
max_features - "auto", "sqrt", "log2"
min_impurity_decrease - while splitting min decrease in impurity at leaf node
random_state - 

'''

def Best_DT_Model(X,Y):
    
    base_model = DecisionTreeClassifier(random_state=42)
    
    custom_params = {
                    'criterion' : ['gini','entropy','log loss'],
                    'splitter' :  ['best','random'],
                    'max_depth' : np.random.randint(3,32,10),
                    'min_samples_leaf' : np.arange(1,5),
                    'min_weight_fraction_leaf' : np.linspace(0.05,0.2,5),
                    'max_features' : ['sqrt','log2', 'auto'],
                    'min_impurity_decrease' : np.linspace(0.1,1,4)
                    }
    
    fold = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    
    model_dt = RandomizedSearchCV(estimator = base_model,param_distributions = custom_params ,n_iter=10,scoring = 'recall',cv = fold,\
                      n_jobs=-1,verbose=1,random_state=42,error_score='raise')

    model_dt.fit(X,Y)

    return model_dt


''' Apply model selection function'''

model_dt = Best_DT_Model(train_x,train_y)

print(model_dt.best_params_)

print(model_dt.cv_results_)


best_dt = model_dt.best_estimator_

y_true = test_y
y_pred_v1 = best_dt.predict(test_x)
y_pred_proba_v1= best_dt.predict_proba(x)[:,1]


# Final Performance Metric
    
fpr_1, tpr_1, thresholds_1 = roc_curve(y_true, y_pred_proba_v1)


print("AUC Score Best DT Model : {}".format(roc_auc_score(y_true, y_pred_proba_v1)))

    
    























