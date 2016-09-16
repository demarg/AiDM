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
# path = 'C:\\Users\\marc\\Dropbox\\private Dokumente\\Studium\\Master\\Courses\\Advances in Data Mining\\Assignment 1\\ml-1m'
path = 'C:/Users/Tomas/OneDrive/Documents/leiden/AiDM/ml-1m'
os.getcwd()
os.chdir(path)

# pass in column names for each CSV and read them using pandas. 
# Column names available in the readme file

#Reading users file:
u_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']
users = pd.read_csv('users.dat', sep='::', names=u_cols,
 encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ratings.dat', sep='::', names=r_cols, encoding='latin-1')

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

# test with smaller dataset
X_red = X.loc[:, 1:500]
I = 3706
K = 2
J = 500

# make NAs zero
where_are_NaNs = np.isnan(X_red)
X_red[where_are_NaNs] = 0

# generate random starting values u and m
U = np.random.rand(I,K)
M = np.random.rand(K,J)

# calcualte SE
X_hat = np.dot(U, M)
e = X_red - X_hat
SE = (e**2).sum()

eta = 0.01
lam = 0.01

# work out derivative
step_U = -2*np.dot(e, M.transpose())
step_M = -2*np.dot(U.transpose(), e)

# take a step towards direction
# implied by derivative
U = U + eta*step_U
M = M + eta*step_M

# repeat process until SE is low enough
# and then keep those us and ms


