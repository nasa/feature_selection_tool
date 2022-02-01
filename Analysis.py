# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:33:54 2020

@author: Nishan Senanayake
"""


#----------------------------------------------------------------------------------------------
# Importing libraries
#----------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from scipy.stats import anderson



#---------------------------------------------------------------------------------------------
# Calculating residulas
#---------------------------------------------------------------------------------------------
def cal_residual(y_pred, y_org):
     return  (y_pred-y_org)


#---------------------------------------------------------------------------------------------
# Reading the CSV file
#---------------------------------------------------------------------------------------------

data = pd.read_csv("My_csv.csv", nrows= 12) 

#-----------------------------------------------------------------------------------------------
#Removing features with low variance of processing numerical data 
#-----------------------------------------------------------------------------------------------


processing_data= data.iloc[:, 6:135]


sel = VarianceThreshold(threshold=0.0001)
sel.fit_transform(processing_data)

constant_columns= [column  for column in processing_data.columns
                    if column not in processing_data.columns[sel.get_support()]]



#------------------------------------------------------------------------------------------------
# Feature selection for Particle Size Distribution  using Pearson Correlation
#------------------------------------------------------------------------------------------------


process_data_no_cons= processing_data.drop(constant_columns, axis=1)


psd= process_data_no_cons.iloc[:, 0:69]


# plot  Pearson Correlation

plt.figure(figsize=(20,14))
cor = psd.corr()
sns.heatmap(cor, annot=False, cmap=plt.cm.CMRmap_r)
plt.savefig("psd_corr.png", dpi=100)
plt.show()

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

# Selecting correlated columns


corr_psd_features = correlation(psd, 0.90)
len(set(corr_psd_features))
print(corr_psd_features)

# Drop correlated columns

psd_non_cor= psd.drop(corr_psd_features,axis=1)
list(psd_non_cor.columns)


#plotting non-corlealted heatmap

plt.figure(figsize=(20,14))
non_cor = psd_non_cor.corr()
sns.heatmap(non_cor, annot=False, cmap=plt.cm.CMRmap_r)
plt.savefig("psd_non_corr.png", dpi=100)
plt.show()

#--------------------------------------------------------------------------------------------------
#Correlation check for Rhelogoy data
#--------------------------------------------------------------------------------------------------


rhelogy =process_data_no_cons.iloc[:, 69:86]

plt.figure(figsize=(14,14))
cor = rhelogy.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.savefig("rehelogy.png", dpi=300)
plt.show()

corr_rhelogy_features = correlation(rhelogy, 0.9)
len(set(corr_rhelogy_features))

# no rhelogy corelation 


#--------------------------------------------------------------------------------
# Correlation check for chemistry
#--------------------------------------------------------------------------------

chemistry =process_data_no_cons.iloc[:, 81:130]


plt.figure(figsize=(20,14))
corr = chemistry.corr()
sns.heatmap(corr, cmap="RdGy",annot=False, fmt=".2f",vmin=-1, vmax=1, cbar_kws={"shrink": .8})
plt.savefig("chemistry.png", dpi=100)
plt.show()


# Select correlated chemistry columns
corr_chemistry_features = correlation(chemistry, 0.9)
len(set(corr_chemistry_features))
print(corr_chemistry_features)






from heatmap import heatmap, corrplot

plt.figure(figsize=(8, 8))
corrplot(chemistry.corr(), size_scale=300);
plt.savefig('test2.png', dpi=100)


# Drop correlated columns

chemistry_non_cor= chemistry.drop(corr_chemistry_features,axis=1)
list(chemistry_non_cor.columns)


#Plot non-corlealted heatmap

plt.figure(figsize=(20,14))
non_cor = chemistry_non_cor.corr()
sns.heatmap(non_cor, annot=False, cmap=plt.cm.CMRmap_r)
plt.savefig("chemistry_non_corr.png", dpi=100)
plt.show()



#--------------------------------------------------------------------------------------------------------
# Remove features with low variance
#-----------------------------------------------------------------------------------------------------


process_data = data.iloc[:, 6:135]
process_data = process_data .drop(corr_psd_features,axis=1)
process_data = process_data .drop(corr_chemistry_features,axis=1)
process_data = process_data .drop(constant_columns, axis=1)


#-------------------------------------------------------------------------------------------------------
#Processing - Structure Linkage (P-S)
#-------------------------------------------------------------------------------------------------------

# Assing input varibales
X=process_data

# Scale values of input variables
X_norm= ((X-X.min())/(X.max()-X.min()))

# All target varibales

y1=data['Porosity_VF_GS'] 
y2=data['Porosity_Size_GS']
y3=data['Grain_Size_FHT']
y4= "crystal Struc"
y5= data['Avg_Carbide_Dia']
y6=data['Carbide_VF']
y7=data['Avg._Nitride_Dia']
y8=data['Nitride_VF']
y9= data["Grain_Structure_FHT"].astype('category')
y9=y9.cat.codes
y11=data['UTS']

#Assign target varibale

name="y11"
y=y11


#-------------------------------------------------------------------------------------------------
#(P-S_1) Feature selction with univariate feature selection
#-------------------------------------------------------------------------------------------------

# powder impact to the microstructure using Univariate feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression



featureSelector = SelectKBest(score_func=f_regression, k=30)
featureSelector.fit(X,y)
for i in range(len(featureSelector.scores_)):
	print('Feature %d: %f' % (i, featureSelector.scores_[i]))
# plot the scores
plt.bar([i for i in range(len(featureSelector.scores_))], featureSelector.scores_)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig("processing_porosity_vf.png", dpi=300)
plt.show()

features= pd.DataFrame(featureSelector.scores_,columns=["Score"])
feature_cols= pd.DataFrame(X.columns)
feature_im_prosity_vf= pd.concat([feature_cols,features], axis=1)


#-------------------------------------------------------------------------------------------------
#(P-S_2) Feature selction with forward selection - linear regression
#-------------------------------------------------------------------------------------------------

sfs = SFS(LinearRegression(),
          k_features=20,
          forward=True,
          floating=False,
          scoring = 'neg_mean_squared_error',
          verbose=2,
          cv = 4)
sfs.fit(X, y)
sfs.k_feature_names_


fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.ylabel('Performance')
plt.grid()
plt.savefig("forward_processing_linear_porosity_vf.png", dpi=300)
plt.show()


#------------------------------------------------------------------------------------
#(P-S_3) Feature selction with forward selection - ridge regression
#------------------------------------------------------------------------------------

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm 


cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)
sfs_ridge_forward= SFS(Ridge(alpha=0.01),
          k_features=4,
          forward=True,
          floating=True,
          scoring = 'neg_mean_squared_error',
          verbose=2,
          cv = cv)
sfs_ridge_forward.fit(X_norm, y)
sfs_ridge_forward.k_feature_names_


# Plot and save

fig1 = plot_sfs(sfs_ridge_forward.get_metric_dict(), kind='std_err' ,figsize=(7,6))
plt.ylabel('Standard Error')
plt.grid()
plt.savefig("forward_processing_ridge_"+name+".png", dpi=300)
plt.show()


#-------------------------------------------------------------------------------------------
# Rigde regression with  forward feature selection - Ridge Regression gives the best for P-S
#-------------------------------------------------------------------------------------------


X_selcted_columns= list(sfs_ridge_forward.k_feature_names_)
X_selected=X_norm[X_selcted_columns]
ridge=Ridge()
cv = RepeatedKFold(n_splits=4, n_repeats=10, random_state=1)

# Hyperparameter tuning

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=cv)
result_ridge= ridge_regressor.fit(X_selected,y)


print('MAE: %.3f' % result_ridge.best_score_)
print('Config: %s' % result_ridge.best_params_)

# Model with seleceted hyperparameter

model=Ridge(alpha=0.0001)
model.fit(X_selected,y)
prediction_ridge=model.predict(X_selected)
y_pred=prediction_ridge


# Plot Predicted vs Actual

font = {'family' : 'normal',
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
plt.plot(x_lim, y_lim, 'k-', color = 'b')
plt.savefig(name+".png", dpi=300, bbox_inches = 'tight')
plt.show()
coefficient_of_dermination = r2_score(y, prediction_ridge)

print(model.intercept_)
columns_importance=  np.array(list(zip(sfs_ridge_forward.k_feature_names_, model.coef_)))


#----------------------------------------------------------------------------------------
# Check Assupmtions for regerssion
#----------------------------------------------------------------------------------------

#Independence  no correlation between consecutive residuals


residual = cal_residual(prediction_ridge,y)

# conduct Durbin-Watson Test

print(durbin_watson(residual))
plt.plot(prediction_ridge,residual, 'ro', color='green', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.ylabel('Residuals')
plt.xlabel('Predicted Value')
plt.rc('font', **font)
plt.rc('axes', labelsize=18)
plt.savefig("ridge_residual"+name+".png", dpi=300, bbox_inches = 'tight')
plt.show()


# QQ plot

sm.qqplot(residual, alpha=0.5) 
y_lim = plt.ylim()
x_lim = plt.xlim()
plt.plot(x_lim, y_lim, 'k-', color = 'r')
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.rc('font', **font)
plt.rc('axes', labelsize=18)
plt.savefig("qq"+name+".png", dpi=300,bbox_inches = 'tight')
plt.show()


#conduct Anderson -Darling test

anderson(residual)

#------------------------------------------------------------------------------
#Structure - Property
#------------------------------------------------------------------------------

# Read structure data
structure_data = pd.read_csv("My_csv.csv", nrows= 18) 
structure=structure_data.iloc[:, 135:145]
structure['Grain_Structure_FHT']=structure["Grain_Structure_FHT"].astype('category')
structure['Grain_Structure_FHT']=structure['Grain_Structure_FHT'].cat.codes
X=structure
X_norm= ((X-X.min())/(X.max()-X.min()))


# Read target varibales

y1=structure_data['Hardness_FHT'] 
y2=structure_data['RaT']
y3=structure_data['RaL']
y4=structure_data['AvgHCFlife']
y5=structure_data['AvgHCFstress']
y6=structure_data['Elastic_Modulus']
y7=structure_data['Prop_Limit']
y8=structure_data['0.02_YS']
y9=structure_data['0.2_YS']
y10=structure_data['UTS']
y11=structure_data['Ef']

#Assign Target variable
y=y10

# Check for NaN Values
idx= np.where(pd.isnull(y))
y=y.drop(idx[0][0])
X_norm= X_norm.drop(idx[0][0])



#-------------------------------------------------------------------------------------------------
#Feature selction with ensemble tree
#-------------------------------------------------------------------------------------------------
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X_norm,y)
ranked_features= pd.Series(model.feature_importances_, index= X.columns)
ranked_features.nlargest(30).plot(kind="barh")
feature_imp_extra_tree=pd.DataFrame(ranked_features.nlargest(20))
plt.figure(figsize=(14,14))

plt.savefig("extra_tree.png", dpi=300)
plt.show()

#-------------------------------------------------------------------------------------------------
#Feature selction with random forest Tree
#-------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

model= RandomForestRegressor()
cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_norm, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
model.fit(X_norm,y)
ranked_features= pd.Series(model.feature_importances_, index= X_norm.columns)
ranked_features.nlargest(5).plot(kind="barh")
feature_imp_rf_tree=pd.DataFrame(ranked_features.nlargest(4))
plt.figure(figsize=(14,14))

X_selected= X_norm[pd.DataFrame(ranked_features.nlargest(4)).index]
model.fit(X_selected,y)
prediction_RF=model.predict(X_selected)




plt.figure(figsize=(7,6))
plt.plot(y,prediction_RF, 'ro',alpha=0.5)
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
plt.plot(x_lim, y_lim, 'k-', color = 'b')
plt.savefig('RF_tree.png', dpi=300, bbox_inches = 'tight')



coefficient_of_dermination = r2_score(y, prediction_RF)

print(coefficient_of_dermination)

plt.savefig("RF_tree.png", dpi=300)
plt.show()

