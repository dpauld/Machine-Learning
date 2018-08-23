# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:43:16 2018

@author: idpaul
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

original_dataset = pd.read_csv('datasets/50_Startups.csv');
print("...........................Snap of dataset...........................")
print(original_dataset.head(5))
dataset = original_dataset.copy()
dsrow,dscol = np.shape(dataset)

X = dataset.iloc[:, 0:dscol-1]
y = dataset.iloc[:, dscol-1:dscol].values

'''#.........extracting categorical features and numerical features columns........#'''

x_features_labels = list(X.columns.values)
x_numerical_labels = list(X._get_numeric_data().columns.values)
x_categorical_labels = list(set(x_features_labels) - set(x_numerical_labels))

'''#...............................Data Preprocessing..............................#'''

def normalization(x,numerical_features):
    mean_x = [];
    std_x = [];
    X_normalized = x; 
    for i in numerical_features:
        m = np.mean(dataset[i])
        s = np.std(dataset[i])
        mean_x.append(m)
        std_x.append(s)
        X_normalized[i] = (X_normalized[i] - m) / s
    return X_normalized, mean_x, std_x

def oneHotEncoding(dataset,categorical_features):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    for i in categorical_features:
        labelencoder_X = LabelEncoder()    
        dataset[i]=labelencoder_X.fit_transform(dataset[i])
        onehotencoder = OneHotEncoder(categorical_features = [dataset.columns.get_loc(i)])
        dataset = onehotencoder.fit_transform(dataset).toarray()
    return dataset

x_scale, mean_r, std_r = normalization(X,x_numerical_labels)
X = oneHotEncoding(X,x_categorical_labels)
xrow,xcol = np.shape(X)
yrow,ycol = np.shape(y)
    
'''#.................Finding the values of weights that makes cost minimal..........#'''

def hypothesis(X,w):
    return np.dot(X,w.T)

def cost_function(X,y,w):
    m = y.size
    return np.sum( (hypothesis(X,w) - y) ** 2 )/(2*m)

def gradient_descent(X,y,w,alpha):
    #cost_history = []
    m = y.size
    flag=1;
    while flag:
        loss = (hypothesis(X,w) - y)
        gradient = np.dot(loss.T,X)/m
        prevW = w
        w = w - alpha * gradient
        #cost_history.append(cost_function(X, y, w))
        if (prevW == w).all():
            break
    return w #,np.array(cost_history)

#Adding x0 to the Feature sets
x0 = np.ones(shape=(xrow,1))
X = np.append(x0,X,1)
xrow_x0,xcol_x0 = np.shape(X)

#setting up initial w or thetas to 0
w = np.zeros(shape=(1,xcol+1))
wrow,wcol = np.shape(w)

#spliting dataset into training(70%) and test(30%) set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#defining alpha
alpha = 0.001

print(".....................................................................")
inital_cost = cost_function(X_train, y_train, w)
print("\nInitial Cost=",inital_cost,"\n")

w = gradient_descent(X_train,y_train,w,alpha)
#w,cost_history = gradient_descent(X_train,y_train,w,alpha)

print("Updated weight: ",w,"\n")
final_minimal_cost = cost_function(X_train, y_train, w)
print("Final Minimal Cost=",final_minimal_cost)

"""
#plotting the cost convergence
fig = plt.figure('Cost function convergence')
plt.plot(cost_history)
plt.grid(True)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function convergence')
plt.show()
"""

'''#.....................Model Evaluation and Model Accuracy.....................#'''

# Model Evaluation - RMSE
def rmse(y, y_pred):
    rmse = np.sqrt(sum((y - y_pred) ** 2) / len(y))
    return rmse

# Model Evaluation - R Squared
def rSquared(y, y_pred):
    mean_y = np.mean(y)
    ss_tot = sum((y - mean_y) ** 2)
    ss_res = sum((y - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

y_pred = hypothesis(X_test,w)
print("\nRoot mean squared error of the Model: ",rmse(y_test, y_pred))
print("R squared value of the model",rSquared(y_test, y_pred))
