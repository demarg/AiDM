# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 15:04:09 2016

@author: marc

Additional info: https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as sm

# SETUP
M_path
T_path
os.getcwd()
os.chdir('C:\\Users\\marc\\Dropbox\\private Dokumente\\Studium\\Master\\Courses\\Advances in Data Mining\\Assignment 1\\ml-1m')

# pass in column names for each CSV and read them using pandas. 
# Column names available in the readme file

#Reading users file:
u_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']
users = pd.read_csv('users.dat', sep='::', names=u_cols,
 encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ratings.dat', sep='::', names=r_cols,
 encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'genre']
items = pd.read_csv('movies.dat', sep='::', names=i_cols,
 encoding='latin-1')
 
print users.shape
users.head()
print ratings.shape
ratings.head()
print items.shape
items.head()


# 1. Naive Approaches


## Global mean

np.mean(ratings["rating"]) #average rating = 3.58
np.var(ratings["rating"]) #var rating = 1.25

## Item-specific means

a = ratings.groupby(['movie_id'])['rating'].mean() #average rating per item

## User-specific means
b = ratings.groupby(['user_id'])['rating'].mean()

## Regression

### Attach indexes
df.a = pd.DataFrame({'movie_id': a.index, 'movie_rating': a})
df.b = pd.DataFrame({'user_id': b.index, 'user_rating': b})

### Merge datasets
df_rating_a = pd.merge(ratings, df.a, how = 'left', on = 'movie_id')
df = pd.merge(df_rating_a, df.b, how = 'left', on = 'user_id')

### Set up regression equation
result = sm.ols(formula = 'rating ~ movie_rating + user_rating', data = df).fit()
print(result.params)

# 2. Gradient descent with regularization

## Matrix factorization

###
X = ratings.pivot(index='movie_id', columns='user_id', values='rating')
X.head
where_are_NaNs = np.isnan(X)
X[where_are_NaNs] = 0
X.head

I = 3706
K = 2
J = 6040

U = np.random.rand(I,K)
M = np.random.rand(K,J)



### wrong
U_red = U[0:500]
where_are_NaNs = np.isnan(U_red)
U_red[where_are_NaNs] = 0
M_red = np.transpose(U_red)

UM = np.dot(U_red, M_red)
U_red.shape
M_red.shape
UM.shape
###

test = np.dot(U, M)
test.shape


