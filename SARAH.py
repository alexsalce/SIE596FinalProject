# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:49:12 2024

@author: Alex Salce
"""

import numpy as np
import numpy.linalg as la


class gdDataSARAH():
    def __init__(self, description, A, b, gd_iter, batch_size,sarah_iter,epsilon=None):
        self.description = description
        self.A = A
        self.b = b
        self.outer_iter = gd_iter        
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.n = self.A.shape[0]
        self.nfeat = self.A.shape[1]
        self.Ab = np.c_[A.reshape(self.n, -1), b.reshape(self.n, 1)]
        self.sarah_iter = sarah_iter
        
    def ls(self,x, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        L = 2 * la.norm(A)**2
        eta = 1/L
        err = A@x-b
        if type(A.T) == type(err): #there has to be a better way to do this part
            grad = 2 * A.T @ err
        else:
            grad = 2 * A.T * err
        return grad, eta
    def SumGradLS(self,x,A=None,b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        err = A@x-b
        ls = A.T@err
        grad = 2 * ls/self.n           
        return grad

def objective_function(x):
    return np.linalg.norm(S@x - t)**2

    
# Define Parameters
epsilon = 1e-2
seed = 123
outer_iter = 10
batchsize = 1

np.random.seed(seed)

d = 1
n = 100
noise = np.random.rand(d,1)
s = np.random.rand(n,d) * 10 - 5          
t = s @ noise + np.random.randn(n,1)       #response (b)
S = np.column_stack((s, np.ones(n)))    #data (A)
x0 = np.zeros((d+1,1))

dataSet1 = gdDataSARAH("Test data 1", S, t, 150, 1, 100, epsilon)

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
    

    for _ in range(0,myData.sarah_iter):
        i = np.random.randint(low = 0, high = (myData.n-1)) #select random index from number of observations
        randA = np.expand_dims(myData.A[i], axis=0)
        randb = np.expand_dims(myData.b[i], axis=0)
        fi_wt = myData.SumGradLS(w,randA,randb)
        fi_wt_old = myData.SumGradLS(wold,randA,randb)
        vnew = fi_wt - fi_wt_old + vold
        w += -eta*np.copy(vnew)
        vold = np.copy(vnew)
        w_list.append(w)
    i = np.random.randint(0,len(w_list))
    wold = np.copy(w_list[i])
    w_list = []

# checks
objective_function(w)
la.norm(myData.ls(w)[0])


# standard gd loop to verify functionality
# np.random.seed(123)
# w = x0.copy()
# while(la.norm(myData.ls(w)[0])>myData.epsilon):
#     w += -myData.ls(w)[1] * myData.ls(w)[0]
# objective_function(w)
  
  
