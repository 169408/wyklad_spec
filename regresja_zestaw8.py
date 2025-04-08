import csv
import random
import math
import numpy as np

def loadDataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        headers = dataset[0]
        dataset = dataset[1: len(dataset)]
        return dataset, headers
dataset, headers = loadDataset('zad1/Zestav8.csv')
print("HEADERS")
print(headers)

print("Dataset Size")
print(len(dataset), "X", len(dataset[0]))

dataset = np.array(dataset)
dataset = dataset.astype(float)

X = dataset[:, 0:-1]
#taking columns with index 0 to 4 as x
Y = dataset[:, -1]
#taking the last column as y
print("Size of X")
print(X.shape)
print("Size of Y")
print(Y.shape)

#adding ones to X
one = np.ones((len(X),1))
X = np.append(one, X, axis=1)
#reshape Y to a column vector
Y = np.array(Y).reshape((len(Y),1))
print(X.shape)
print(Y.shape)

# print(X)
# print(Y)

from IPython.display import display
print("X head")
display(X[0:5])
print("Y head")
display(Y[0:5])

def train_test_split(X, Y, split):

    #randomly assigning split% rows to training set and rest to test set
    indices = np.array(range(len(X)))
    
    train_size = round(split * len(X))

    random.shuffle(indices)

    train_indices = indices[0:train_size]
    test_indices = indices[train_size:len(X)]

    X_train = X[train_indices, :]
    X_test = X[test_indices, :]
    Y_train = Y[train_indices, :]
    Y_test = Y[test_indices, :]
    
    return X_train,Y_train, X_test, Y_test

split = 0.75
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, split)

print ("TRAINING SET")
print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)

print("TESTING SET")
print("X_test.shape: ", X_test.shape)
print("Y_test.shape: ", Y_test.shape)

def normal_equation(X, Y):
    beta = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,Y))

    return beta
def predict(X_test, beta):
    return np.dot(X_test, beta)

beta = normal_equation(X_train, Y_train)
predictions = predict(X_test, beta)

print(predictions.shape)

def metrics(predictions, Y_test):

    #calculating mean absolute error
    MAE = np.mean(np.abs(predictions-Y_test))

    #calculating root mean square error
    MSE = np.square(np.subtract(Y_test,predictions)).mean() 
    RMSE = math.sqrt(MSE)

    #calculating r_square
    rss = np.sum(np.square((Y_test- predictions)))
    mean = np.mean(Y_test)
    sst = np.sum(np.square(Y_test-mean))
    r_square = 1 - (rss/sst)
    

    return MAE, RMSE, r_square

mae, rmse, r_square = metrics(predictions, Y_test)
print("Mean Absolute Error: ", mae)
print("Root Mean Square Error: ", rmse)
print("R square: ", r_square)


import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# Obliczenie reszt modelu
residuals = Y_test - predictions

# Test Breuscha-Pagana
bp_test = het_breuschpagan(residuals, X_test)

print("Breusch-Pagan test (heteroskedastyczność)")
print("LM Statistic:", bp_test[0])
print("p-value:", bp_test[1])

from statsmodels.stats.stattools import durbin_watson

dw_statistic = durbin_watson(residuals)

print("Durbin-Watson Statistic:", dw_statistic)

# # Estymacja modelu
# model = sm.OLS(y, X).fit()

# # Wyniki regresji
# print(model.summary())
