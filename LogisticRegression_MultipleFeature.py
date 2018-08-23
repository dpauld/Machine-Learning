# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:43:16 2018

@author: idpauld
"""
import pandas as pd  
import numpy as np  
import math
from sklearn.metrics import accuracy_score  
#%matplotlib inline

path = "datasets/Social_Network_Ads_2.csv"
original_dataset = pd.read_csv(path);
print("...........................Snap of dataset...........................")
print(original_dataset.head(5))
dataset = original_dataset.copy()
dsrow,dscol = np.shape(dataset)

X = dataset.iloc[:, 0:dscol-1]
y = dataset.iloc[:, dscol-1:dscol]

'''#.........extracting categorical features and numerical features columns........#'''

x_features_labels = list(X.columns.values)
#print(features_names)
x_numerical_labels = list(X._get_numeric_data().columns.values)
#print(numerical_features)
x_categorical_labels = list(set(x_features_labels) - set(x_numerical_labels))
#print(categorical_features)
y_class_labels = list(y.columns.values)

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

def labelEncoding(dataset,y_class_labels):
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    dataset[y_class_labels[0]] = labelencoder_X.fit_transform(dataset[y_class_labels[0]])
    #dataset = labelencoder_X.fit_transform(dataset[])
    return dataset

x_scale, mean_r, std_r = normalization(X,x_numerical_labels)
X = oneHotEncoding(X,x_categorical_labels)
y = labelEncoding(y,y_class_labels)
y = y.values
y = y.astype(float)

xrow,xcol = np.shape(X)
yrow,ycol = np.shape(y)

'''#.................Finding the values of weights that makes cost minimal..........#'''

def sigmoid(z):
    #print(z,"\n")
    return (1 / (1 + math.exp(-z)))

def hypothesis(X,w):
    hyp = np.dot(X,w.T)
    #print(z)
    threshHold = 0.5
    for i in hyp:
        sigVal = sigmoid(i[0])
        #i[0] = sigVal
        if sigVal>=threshHold:
            i[0]=1
        else:
            i[0]=0
    return hyp

def cost_function(X,y,w):
    m = y.size
    hyp = hypothesis(X,w)
    return np.sum( np.dot(y.T,np.log(hyp)) + np.dot((1-y).T,np.log(1-hyp)))/(m)

def gradient_descent(X,y,w,alpha):
    m = y.size
    flag=1;
    #it = 80000
    #for i in range(it): #alternative of while loop
    while flag: 
        loss = (hypothesis(X,w) - y)
        gradient = np.dot(loss.T,X)/m
        prevW = w
        w = w - alpha * gradient
        if (prevW == w).all():
            flag=0
    return w

#Adding x0 to the Feature sets
x0 = np.ones(shape=(xrow,1))
X = np.append(x0,X,1)
xrow_x0,xcol_x0 = np.shape(X)

#setting up initial w or thetas to 0
w = np.zeros(shape=(1,xcol_x0))

#spliting dataset into training(70%) and test(30%) set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#defining alpha
alpha = 0.03
w = gradient_descent(X_train,y_train,w,alpha)

print(".....................................................................")
print("Updated weight: ",w,"\n")

'''#.....................Model Evaluation and Model Accuracy.....................#'''
y_pred = hypothesis(X_test,w)
#print(y_pred)
print("Test Accuracy ", accuracy_score(y_test, y_pred) * 100 , '%')