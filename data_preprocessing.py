#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:49:29 2017

@author: Louie
"""

# Load libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm

# Load dataset:
df = pd.read_csv('nyc-rolling-sales.csv')
df_main = df[df.columns[1:]]

# Change " -  ", " ", 0 into missing values:
df_main = df_main.replace(' -  ',np.nan)
df_main = df_main.replace(' ',np.nan)
df_main['YEAR BUILT'] = df_main['YEAR BUILT'].replace('0',np.nan)
df_main

# Missing value count:
miss = df_main.isnull().sum()/len(df_main)
miss.sort_values(inplace=True)

## Visualising missing values
miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index

# Plot the missing value count: Plot 1
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=miss)
plt.xticks(rotation = 90)
sns.plt.show()

# Drop columns with over 50% missing value:
del df_main['EASE-MENT']
del df_main['APARTMENT NUMBER']

# Drop rows with missing target values:
df_dropna = df_main.dropna(axis=0,subset=['SALE PRICE'])

# Change the data type for different columns:
'''df_dropna.loc[:,['BOROUGH','NEIGHBORHOOD','BUILDING CLASS CATEGORY',
          'TAX CLASS AT PRESENT','BLOCK','LOT',
          'BUILDING CLASS AT PRESENT','ADDRESS','ZIP CODE',
          'TAX CLASS AT TIME OF SALE',
          'BUILDING CLASS AT TIME OF SALE','SALE DATE']] = df_dropna.loc[:,['BOROUGH','NEIGHBORHOOD','BUILDING CLASS CATEGORY',
                                                       'TAX CLASS AT PRESENT','BLOCK','LOT',
                                                       'BUILDING CLASS AT PRESENT','ADDRESS','ZIP CODE',
                                                       'TAX CLASS AT TIME OF SALE',
                                                       'BUILDING CLASS AT TIME OF SALE','SALE DATE']].astype('str')'''

df_dropna.loc[:,['RESIDENTIAL UNITS','COMMERCIAL UNITS','TOTAL UNITS',
           'LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT','SALE PRICE']] = df_dropna.loc[:,['RESIDENTIAL UNITS',
                                                               'COMMERCIAL UNITS','TOTAL UNITS',
                                                               'LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT',
                                                               'SALE PRICE']].apply(pd.to_numeric)

# Drop rows with outliers in target variable (0, 1, 10)
df_dropna = df_dropna[df_dropna['SALE PRICE'] > 10]

# Break up the dataset into numerical and categorical variables:
df_numerical = df_dropna.select_dtypes(include=[np.number])
df_categorical = df_dropna.select_dtypes(exclude=[np.number])
df_target = df_numerical['SALE PRICE']
del df_numerical['SALE PRICE']

# Plot the distribution of the target variable: Plot 2
plt.figure()
sns.distplot(df_target) # Extremely skewed to the right

# Log transformation, and plot: Plot 3
df_target_trans = np.log(df_target)
plt.figure()
sns.distplot(df_target_trans) # Normally distributed at last.

## Numerical Variables Analysis:
# Correlation:
# Heatmap: Plot 4
df_numerical['SALE PRICE_TRANS'] = df_target_trans.values
corr = df_numerical.corr()
plt.figure()
sns.heatmap(corr)
'''Note: No feature has a strong correlation with the target variable.'''

## Categorical Variables: Anova, Disparity Score
# Code the categories into dummy categories: (Not sure if it is necessary..)
cat = [f for f in df_categorical.columns]
class_mapping = [] # The list of dictionary marking the labels and the corresponding indices
for c in cat:
    new_class_mapping = {label:idx for idx,label in enumerate(df_categorical[c].dropna().unique())}
    class_mapping.append(new_class_mapping)
    df_categorical[c] = df_categorical[c].map(new_class_mapping)
    
## Anova and Disparity Score
def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = cat
    pvals = []
    for c in cat:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SALE PRICE'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

df_categorical['SALE PRICE'] = df_target_trans.values
k = anova(df_categorical)
k['disparity'] = np.log(1./k['pval'].values)
plt.figure() # plot the disparity score, Plot 5
sns.barplot(data=k,x='features',y='disparity')
plt.xticks(rotation=90)
'''Note: the disparity scores of most features are infinity, and their p-values are 0'''

# Create histogram for numerical variables:
num = [f for f in df_numerical.columns]
num.remove('SALE PRICE_TRANS')
nd = pd.melt(df_numerical,value_vars=num)
plt.figure() # plot the histograms, Plot 6
n1 = sns.FacetGrid(nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')
'''Note: right skewed: RESIDENTIAL UNITS, COMMERCIAL UNITS, TOTAL UNITS, LAND SQUARE FEET,
GROSS SQUARE FEET'''

# Create boxplot for categorical variables:
# Note: This part takes ages to run, and the boxplots do not look good. Seems that some
# of the categorical features have so many levels.
'''def boxplot(x,y,**kwargs):
    sns.boxplot(x=x,y=y)
    x = plt.xticks(rotation=90)
cat = [f for f in df_categorical.columns]
df_categorical['SALE PRICE_TRANS'] = df_numerical['SALE PRICE_TRANS'].values
plt.figure()
p = pd.melt(df_categorical, id_vars='SALE PRICE_TRANS', value_vars=cat)
g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, 'value','SALE PRICE_TRANS')
'''

## Fill in the missing values for the features:
# For categorical features, fill in with mode; for numerical features, fill in with mean
# df_dropna.isnull().sum() # check which features have missing data, and how many
''' BOROUGH                               0
    NEIGHBORHOOD                          0
    BUILDING CLASS CATEGORY               0
    TAX CLASS AT PRESENT                593
    BLOCK                                 0
    LOT                                   0
    BUILDING CLASS AT PRESENT           593
    ADDRESS                               0
    ZIP CODE                              0
    RESIDENTIAL UNITS                     0
    COMMERCIAL UNITS                      0
    TOTAL UNITS                           0
    LAND SQUARE FEET                  21051
    GROSS SQUARE FEET                 21592
    YEAR BUILT                         4211
    TAX CLASS AT TIME OF SALE             0
    BUILDING CLASS AT TIME OF SALE        0
    SALE PRICE                            0
    SALE DATE                             0
'''

df_categorical['TAX CLASS AT PRESENT'] = df_categorical['TAX CLASS AT PRESENT'].fillna(
        stats.mode(df_categorical['TAX CLASS AT PRESENT']).mode[0])
df_categorical['BUILDING CLASS AT PRESENT'] = df_categorical['BUILDING CLASS AT PRESENT'].fillna(
        stats.mode(df_categorical['BUILDING CLASS AT PRESENT']).mode[0])
df_numerical['LAND SQUARE FEET'] = df_numerical['LAND SQUARE FEET'].fillna(
        np.nanmedian(df_numerical['LAND SQUARE FEET']))
df_numerical['GROSS SQUARE FEET'] = df_numerical['GROSS SQUARE FEET'].fillna(
        np.nanmedian(df_numerical['GROSS SQUARE FEET']))
df_numerical['YEAR BUILT'] = df_numerical['YEAR BUILT'].fillna(
        np.nanmedian(df_numerical['YEAR BUILT']))

## Transform numerical features into normal distribution: Create a new dataframe df_numerical_trans
df_numerical_trans = df_numerical[['RESIDENTIAL UNITS','COMMERCIAL UNITS',
                                   'TOTAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET',
                                   'YEAR BUILT']]

df_numerical_trans['RESIDENTIAL UNITS'] = np.log1p(df_numerical['RESIDENTIAL UNITS'])
df_numerical_trans['COMMERCIAL UNITS'] = np.log1p(df_numerical['COMMERCIAL UNITS'])
df_numerical_trans['TOTAL UNITS'] = np.log1p(df_numerical['TOTAL UNITS'])
df_numerical_trans['LAND SQUARE FEET'] = np.log1p(df_numerical['LAND SQUARE FEET'])
df_numerical_trans['GROSS SQUARE FEET'] = np.log1p(df_numerical['GROSS SQUARE FEET'])

## Standardize the numerical features:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_numerical_trans)
scaled = scaler.transform(df_numerical_trans)
df_numerical_scaled = pd.DataFrame()

for i,col in enumerate(df_numerical_trans.columns):
    df_numerical_scaled[col] = scaled[:,i]

## Arrange datasets:
# Numerical features: df_numerical_scaled
# Categorical features: df_categorical

del df_categorical['SALE PRICE']
# Change the dummy variables back to original values / labels
# Takes some time, do not know if it is necessary...
  # for i in np.arange(df_categorical.shape[1]):
    #   new_dict = {x:y for y,x in class_mapping[i].items()}
    #   df_categorical.iloc[:,[i]] = df_categorical.iloc[:,[i]].replace(new_dict,inplace=True)
df_categorical[['BOROUGH','BLOCK','LOT','ZIP CODE','TAX CLASS AT TIME OF SALE']] = df_numerical[[
        'BOROUGH','BLOCK','LOT','ZIP CODE','TAX CLASS AT TIME OF SALE']]
# Target variable: df_target_trans