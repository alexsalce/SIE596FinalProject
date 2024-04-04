# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:49:12 2024

@author: Alex Salce
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


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
    def Lip(self, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        L = 2 * la.norm(A)**2
        eta = 1/L
        return eta    
    def GradLS(self,x, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        err = A@x-b
        ls = A.T@err
        grad = 2 * ls/self.n   #converges
        # grad = 2 * ls          #does not converge
        return grad
    def AvgGradLS(self,x,A=None,b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        err = A@x-b
        ls = A.T@err
        grad = 2 * ls/self.n           
        return grad
    def sGradLS(self,x, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        i = np.random.randint(low = 0, high = (self.n-1)) #select random index from number of observations
        randA = np.expand_dims(self.A[i], axis=0)
        randb = np.expand_dims(self.b[i], axis=0)
        grad = self.GradLS(x,randA,randb)
        return grad

def objective_function(x):
    return np.linalg.norm(S@x - t)**2

    
outer_iter = 100
inner_iter = 200

# Define Parameters
epsilon = 1e-2
seed = 123
batchsize = 1

np.random.seed(seed)

d = 5
n = 100
noise = np.random.rand(d,1)
s = np.random.rand(n,d) * 10 - 5          
t = s @ noise + np.random.randn(n,1)       #response (b)
S = np.column_stack((s, np.ones(n)))    #data (A)
x0 = np.zeros((d+1,1))
# t = s @ noise       #response (b)  NO NOISE BIAS
# S = s    #data (A) NO NOISE BIAS
# x0 = np.zeros((d,1))  #NO NOISE BIAS

dataSet1 = gdDataSARAH("Test data 1", S, t, outer_iter, 1, inner_iter, epsilon)

myData = dataSet1



#SARAH algorithm

#Least Squares Loss
wold = np.copy(x0)
eta = myData.Lip()
w_list = []
obj = []
counter = 0
for _ in range (myData.outer_iter):
    vold = myData.AvgGradLS(wold)
    w = wold.copy() - eta * vold.copy()
    obj.append(objective_function(w))
    for _ in range(0,myData.sarah_iter):
        fi_wt = myData.sGradLS(w,myData.A,myData.b)
        fi_wt_old = myData.sGradLS(wold,myData.A,myData.b)
        vnew = fi_wt - fi_wt_old + vold
        w += -eta*np.copy(vnew)
        vold = np.copy(vnew)
        w_list.append(w)
        obj.append(objective_function(w))
    i = np.random.randint(0,len(w_list))
    wold = np.copy(w_list[i])
    w_list = []

# checks
objective_function(w)
la.norm(myData.GradLS(w))


plt.plot(np.log(obj))
plt.show()

# standard gd loop to verify functionality
# np.random.seed(123)
# w = x0.copy()
# while(la.norm(myData.GradLS(w))>myData.epsilon):
#     w += -myData.Lip(myData.A, myData.b) * myData.GradLS(w)
# objective_function(w)
# la.norm(myData.GradLS(w))
  
  
