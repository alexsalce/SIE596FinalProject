# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:49:12 2024

@author: Alex Salce
"""

import numpy as np
import numpy.linalg as la
from abc import ABC


class gdData(ABC):
    def __init__(self, name, A, b, gd_iter, batch_size,sarah_iter,epsilon=None):
        self.name = name
        self.A = np.array(A)
        self.b = np.array(b)
        self.outer_iter = gd_iter        
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.n = self.A.shape[0]
        self.nfeat = self.A.shape[1]
        self.Ab = np.c_[A.reshape(self.n, -1), b.reshape(self.n, 1)]
        self.sarah_iter = sarah_iter
        
    def ls(cls,x, A=None, b=None):
        A = cls.A if A is None else A
        b = cls.b if b is None else b
        L = 2 * la.norm(A)**2
        eta = 1/L
        err = A@x-b
        if type(A.T) == type(err): #there has to be a better way to do this part
            grad = 2 * A.T @ err
        else:
            grad = 2 * A.T * err
        return grad, eta
    def SumGradLS(cls,x,A=None,b=None):
        A = cls.A if A is None else A
        b = cls.b if b is None else b
        grad = np.zeros(cls.nfeat)
        err = A@x-b
        for i in range(cls.nfeat):
            ls = np.multiply(A[:,i].T,err)
            grad[i] = sum(ls)/cls.n            
        return grad
    
# test = gdData("Cool", np.array([[5, 15], [25, 35], [45, 55],[6, 16], [26, 36], [46, 56],]), np.array([5, 20, 14, 32, 22, 38]), 100, 3, 1)

# Define Parameters
epsilon = 1e-2
seed = 123
outer_iter = 10
# sarah_iter = 60
batchsize = 1

np.random.seed(seed)

d = 1
n = 100
noise = np.random.rand(d)
s = np.random.rand(n,d) * 10 - 5          
t = s @ noise + np.random.randn(n)       #response (b)
S = np.column_stack((s, np.ones(n)))    #data (A)
x0 = np.zeros(d+1)

dataSet1 = gdData("Test data 1", S, t, 1000, 1, 500, epsilon)

myData = dataSet1



#SARAH algorithm

#Least Squares Loss
wold = np.copy(x0)
eta = myData.ls(wold)[1]
w_list = []
counter = 0
for _ in range (myData.outer_iter):
    vold = myData.SumGradLS(wold)
    w = wold.copy() - eta * vold.copy()
    for j in range(0,myData.sarah_iter):
        i = np.random.randint(low = 0, high = (myData.n-1)) #select random index from number of observations
        vnew = myData.ls(w,myData.A[i],myData.b[i])[0] - myData.ls(wold,myData.A[i],myData.b[i])[0] + vold
        w += -eta*np.copy(vnew)
        vold = np.copy(vnew)
    counter+=1
    w_list.append(w)
    i = np.random.randint(0,len(w_list))
    wold = np.copy(w_list[i])
    w_list = []


# standard gd loop to verify functionality
np.random.seed(123)
w = x0.copy()
while(la.norm(myData.ls(w)[0])>myData.epsilon):
    w += -myData.ls(w)[1] * myData.ls(w)[0]
    
np.random.seed(123)
w = x0.copy()
while(la.norm(myData.ls(w)[0])>myData.epsilon):
    w += -myData.ls(w)[1] * myData.SumGradLS(w)
