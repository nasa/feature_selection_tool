# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:49:19 2021

@author: Nishan Senanayake
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm 
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score


#Reading the CSV

data = pd.read_csv("AMSII_data.csv", nrows= 12) 

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------


#Removing features with low variance of processing numerical data 

processing_data= data.iloc[:, 6:135]

sel = VarianceThreshold(threshold=0.0001)
sel.fit_transform(processing_data/processing_data.mean())

constant_columns= [column  for column in processing_data.columns
                    if column not in processing_data.columns[sel.get_support()]]




#------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
# PSD feature selection using Pearson Correlation


process_data_no_cons= processing_data.drop(constant_columns, axis=1)

psd= process_data_no_cons.iloc[:, 0:69]

# select highly correlated features and remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) > threshold) or (abs(corr_matrix.iloc[i, j]) < (-threshold)): # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

#Selecting corelated columns
corr_psd_features = correlation(psd, 0.90)
len(set(corr_psd_features))
print(corr_psd_features)

# Dropping corelated columns

psd_non_cor= psd.drop(corr_psd_features,axis=1)
list(psd_non_cor.columns)




#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


#Rhelogoy data check Correlation

rhelogy =process_data_no_cons.iloc[:, 69:86]

corr_rhelogy_features = correlation(rhelogy, 0.9)
len(set(corr_rhelogy_features))

# no rhelogy corelation 


#--------------------------------------------------------------------------------
#chemistry corlation
#--------------------------------------------------------------------------------

chemistry =process_data_no_cons.iloc[:, 86:129]

#selecting corelated chemistry columns
corr_chemistry_features = correlation(chemistry, 0.95)
len(set(corr_chemistry_features))
print(corr_chemistry_features)

# Dropping corelated columns

chemistry_non_cor= chemistry.drop(corr_chemistry_features,axis=1)
list(chemistry_non_cor.columns)

#--------------------------------------------------------------------------------------------------------
# Removing all features with low variance and corelated
#-----------------------------------------------------------------------------------------------------


process_data = data.iloc[:, 6:135]
process_data = process_data .drop(corr_psd_features,axis=1)
process_data = process_data .drop(corr_chemistry_features,axis=1)
process_data = process_data .drop(constant_columns, axis=1)


X=process_data


#----------------------------------------------------------------------------------------
#Target varibales
#----------------------------------------------------------------------------------------

property_data = pd.read_csv("AMSII_data.csv", nrows= 12) 


#Assign depent varibale
y1=property_data['Hardness, FHT'] 
y2=property_data['Ra T']
y3=property_data['Ra L']
y4=property_data['Avg HCF life']
y5=property_data['Avg HCF stress']
y6=property_data['Elastic_Modulus']
y7=property_data['Prop_Limit']
y8=property_data['0.02_YS']
y9=property_data['0.2_YS']
y10=property_data['UTS']
y11=property_data['Ef']


y=y8
name="y8"

X_norm= ((X-X.min())/(X.max()-X.min()))



# idx= np.where(pd.isnull(y))
# y=y.drop(idx[0][0])
# X_norm= X_norm.drop(idx[0][0])


cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)
sfs_ridge_forward= SFS(Ridge(alpha=0.01),
          k_features=5,
          forward=True,
          floating=True,
          scoring = 'neg_mean_squared_error',
          verbose=2,
          cv = cv)
sfs_ridge_forward.fit(X_norm, y)
sfs_ridge_forward.k_feature_names_



fig1 = plot_sfs(sfs_ridge_forward.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.ylabel('Perfomance')
plt.grid()
plt.savefig("forward_processing_Porperty_ridge_"+name+".png", dpi=300)
plt.show()


X_selcted_columns= list(sfs_ridge_forward.k_feature_names_)
X_selected=X_norm[X_selcted_columns]
ridge=Ridge()
cv = RepeatedKFold(n_splits=4, n_repeats=10, random_state=1)
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=cv)
result_ridge= ridge_regressor.fit(X_selected,y)


print('MAE: %.3f' % result_ridge.best_score_)
print('Config: %s' % result_ridge.best_params_)


model=Ridge(alpha=result_ridge.best_params_['alpha'])
model.fit(X_selected,y)
prediction_ridge=model.predict(X_selected)
y_pred=prediction_ridge
# only for Grain_Structure
#prediction_ridge= np.round(prediction_ridge)


font = {#'family' : 'normal',
        'weight' : 'bold',
        }

plt.figure(figsize=(6,6))
plt.plot(y,prediction_ridge, 'ro', alpha=0.5)
plt.ylabel('Predicted Value')
plt.xlabel('Actual Value')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
fmin = min(xmin, ymin)
fmax = max(xmax, ymax)
plt.xlim(fmin, fmax)
plt.ylim(fmin, fmax)
y_lim = plt.ylim()
x_lim = plt.xlim()
plt.rc('font', **font)
plt.rc('axes', labelsize=18)
plt.plot(x_lim, y_lim, '-', color = 'b')
plt.savefig(name+".png", dpi=300, bbox_inches = 'tight')
plt.show()
coefficient_of_dermination = r2_score(y, prediction_ridge)

print(model.intercept_)
intercept= model.intercept_
columns_importance=  np.array(list(zip(sfs_ridge_forward.k_feature_names_, model.coef_)))



#----------------------------------------------------------------------------------------
#Check Assupmtions
#----------------------------------------------------------------------------------------

# 1 Independence  no correlation between consecutive residuals



def cal_residual(y_pred, y_org):
     return  (y_pred-y_org)




residual = cal_residual(prediction_ridge,y)
print(durbin_watson(residual))
plt.plot(prediction_ridge,residual, 'o', color='green', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.ylabel('Residuals')
plt.xlabel('Predicted Value')
plt.rc('font', **font)
plt.rc('axes', labelsize=18)
plt.savefig("ridge_residual_process"+name+".png", dpi=300, bbox_inches = 'tight')
plt.show()



sm.qqplot(residual, alpha=0.5) 
y_lim = plt.ylim()
x_lim = plt.xlim()
plt.plot(x_lim, y_lim, '-', color = 'r')
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.rc('font', **font)
plt.rc('axes', labelsize=18)
plt.savefig("qq_process_prop"+name+".png", dpi=300,bbox_inches = 'tight')
plt.show()








