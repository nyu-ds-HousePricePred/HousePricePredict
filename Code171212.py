#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:10:44 2017

@author: Louie
"""

#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score 
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('nyc-rolling-sales.csv')

# Change " -  ", " ", 0 into missing values:
data = data.replace(' -  ',np.nan)
data = data.replace(' ',np.nan)
data['YEAR BUILT'] = data['YEAR BUILT'].replace('0',np.nan)

# Drop rows with missing target values:
data = data.dropna(axis=0,subset=['SALE PRICE'])

#missing value counts in each of these columns
miss = data.isnull().sum()/len(data)
miss = miss[miss > 0]
miss.sort_values(inplace=True)

#visualising missing values
miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index

#plot the missing value count
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=miss)
plt.xticks(rotation = 90)
sns.plt.show()

# Drop columns with over 60% missing value:
del data['EASE-MENT']
del data['APARTMENT NUMBER']

##
import dateutil
data['SALE PRICE'] = data['SALE PRICE'].astype(float)
data['LAND SQUARE FEET'] = data['LAND SQUARE FEET'].astype(float)
data['GROSS SQUARE FEET'] = data['GROSS SQUARE FEET'].astype(float)
data['BOROUGH'] = data['BOROUGH'].astype('category')
data['BLOCK'] = data['BLOCK'].astype('category')
data['YEAR BUILT'] = data['YEAR BUILT'].astype('category')
data['ZIP CODE'] = data['ZIP CODE'].astype('category')
data['TAX CLASS AT TIME OF SALE'] = data['TAX CLASS AT TIME OF SALE'].astype('category')
data['SALE DATE'] = data['SALE DATE'].apply(lambda x: dateutil.parser.parse(x).month).astype('category')

data['BUILDING CLASS AT TIME OF SALE'] = data['BUILDING CLASS AT TIME OF SALE'].astype('category')
data['NEIGHBORHOOD'] = data['NEIGHBORHOOD'].astype('category')
data['BUILDING CLASS CATEGORY'] = data['BUILDING CLASS CATEGORY'].astype('category')
data['TAX CLASS AT PRESENT'] = data['TAX CLASS AT PRESENT'].astype('category')
data['BUILDING CLASS AT PRESENT'] = data['BUILDING CLASS AT PRESENT'].astype('category')
data['ADDRESS'] = data['ADDRESS'].astype('category')

#now transforming the target variable
data['SALE PRICE'].describe()
data['SALE PRICE'] = data['SALE PRICE'].replace(0,np.nan)

# Drop rows with outliers in target variable (0, 1, 10)
data = data[data['SALE PRICE'] > 10]
data['SALE PRICE'].describe()

#separate variables into new data frames
num_data = data.select_dtypes(include=[np.number])
del num_data['SALE PRICE']


cat_data = data.select_dtypes(exclude=[np.number])

# Drop categorical column with too many unique key:
del cat_data['ADDRESS']
print ("There are {} numeric and {} categorical columns in train data".format(num_data.shape[1],cat_data.shape[1]))

## Fill in the missing values for the features:
 
# check which features have missing data, and how many
print(num_data.isnull().sum())
print(cat_data.isnull().sum())

# for numerical features, fill in with median
num_data['LAND SQUARE FEET'] = num_data['LAND SQUARE FEET'].fillna(
        num_data['LAND SQUARE FEET'].mean())
num_data['GROSS SQUARE FEET'] = num_data['GROSS SQUARE FEET'].fillna(
        num_data['GROSS SQUARE FEET'].mean())
# For categorical features, fill in with mode;

cat_data['BUILDING CLASS AT PRESENT'] = cat_data['BUILDING CLASS AT PRESENT'].fillna(cat_data['BUILDING CLASS AT PRESENT'].mode()[0])
cat_data['TAX CLASS AT PRESENT'] = cat_data['TAX CLASS AT PRESENT'].fillna(cat_data['TAX CLASS AT PRESENT'].mode()[0])
cat_data['YEAR BUILT'] = cat_data['YEAR BUILT'].fillna(cat_data['YEAR BUILT'].mode()[0])

print(num_data.isnull().sum())
print(cat_data.isnull().sum())


#SalePrice
sns.distplot(data['SALE PRICE'])
#skewness
data['SALE PRICE'].skew()

target = np.log(data['SALE PRICE'])

##
num_data['SALE PRICE_TRANS'] = target.values

del num_data['SALE PRICE_TRANS']

#numerical data
# From helperFunction script 
def normalize_df(frame):
	'''
	Helper function to Normalize data set
	Intializes an empty data frame which 
	will normalize all floats types
	and just append the non-float types 
	so basically the class in our data frame
	'''
	Norm = pd.DataFrame()
	for item in frame:
		
			Norm[item] = 100 * ((frame[item] - frame[item].min()) / 
			(frame[item].max() - frame[item].min()))

		
	return Norm

Norm = normalize_df(num_data)

Norm_trans = pd.DataFrame()
Norm_trans['LOT'] = np.log(Norm['LOT'])
Norm_trans['RESIDENTIAL UNITS'] = np.log1p(Norm['RESIDENTIAL UNITS'])
Norm_trans['COMMERCIAL UNITS'] = np.log1p(Norm['COMMERCIAL UNITS'])
Norm_trans['TOTAL UNITS'] = np.log1p(Norm['TOTAL UNITS'])
Norm_trans['LAND SQUARE FEET'] = np.log1p(Norm['LAND SQUARE FEET'])
Norm_trans['GROSS SQUARE FEET'] = np.log1p(Norm['GROSS SQUARE FEET'])

##
le = preprocessing.LabelEncoder()
for column in cat_data:
    cat_data[column] = le.fit_transform(cat_data[column].astype(str))
    
cat_data.head(5)

del Norm['Unnamed: 0']
#X = pd.concat([Norm,cat_data],axis = 1)
#X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, random_state=42)
num_data_new = num_data.loc[:,['LOT','LAND SQUARE FEET','GROSS SQUARE FEET']]
X = pd.concat([num_data_new,cat_data],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.3,random_state=42)


##
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))
  
## tuning parameter: k 
from sklearn.metrics import mean_squared_error
myKs = []
for i in range(0, 50):
    if (i % 2 != 0):
        myKs.append(i)

cross_vals = []
for k in myKs:

    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn,
                             X_train, 
                              y_train, 
                             cv = 10, 
                             scoring='neg_mean_squared_error')
    cross_vals.append(scores.mean())

MSE = [1 - x for x in cross_vals]
optimal_n = myKs[MSE.index(min(MSE))]
print("Optimal K is {0}".format(optimal_n))

##final KNN
neigh = KNeighborsRegressor(n_neighbors=optimal_n)
neigh.fit(X_train, y_train) 
y_pred = neigh.predict(X_test)
print("KNN RMSE on training set: ", rmse(y_test, y_pred))
print("KNN R2 on training set: ", r2_score(y_test, y_pred))

##Lasso
#tunning parameter: alpha
alpha_options = [0.1,0.01,0.001,0.0001,0.00001]
cross_vals = []

for alp in alpha_options :
    LA = Lasso(alpha=alp, max_iter=100000)
    scores = cross_val_score(LA,
                             X_train, 
                              y_train, 
                             cv = 10, 
                             scoring='neg_mean_squared_error')
    cross_vals.append(scores.mean())

MSE = [1 - x for x in cross_vals]
optimal_alpha = alpha_options[MSE.index(min(MSE))]

print("Optimal alpha is {0}".format(optimal_alpha))

from sklearn.linear_model import Lasso

#found this best alpha through cross-validation

regr = Lasso(alpha=optimal_alpha, max_iter=100000)
regr.fit(X_train, y_train)

# run prediction on the training set to get a rough idea of how well it does
y_pred = regr.predict(X_test)
print("Lasso score on training set: ", rmse(y_test, y_pred))
print("Lasso R2 on training set: ", r2_score(y_test, y_pred))

##Random Forest
##tunning parameter: n_estimatiors, the number of trees
n_estimators_options = [100,200,300,400,500]
cross_vals = []

for n_tree in n_estimators_options :
    RF = RandomForestRegressor(n_estimators=n_tree, min_samples_split=2, max_features = 4)
    scores = cross_val_score(RF,
                             X_train, 
                              y_train, 
                             cv = 10, 
                             scoring='neg_mean_squared_error')
    cross_vals.append(scores.mean())

MSE = [1 - x for x in cross_vals]
optimal_trees = n_estimators_options[MSE.index(min(MSE))]

print("Optimal number of tree is {0}".format(optimal_trees))

#final RandomForest
regr = RandomForestRegressor(n_estimators=optimal_trees, min_samples_split=2,max_features = 4)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print("RF score on training set: ", rmse(y_test, y_pred))
print("RF R2 on training set: ", r2_score(y_test, y_pred))