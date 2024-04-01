# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:53:59 2024

@author: Alex Salce
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


class gdDataADAM():
    def __init__(self, description, A, b, maxit, alpha, beta1, beta2, epsilon=None):
        self.description = description
        self.A = A
        self.b = b
        self.maxit = maxit        
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.maxit = maxit
        self.epsilon = epsilon
        self.n = self.A.shape[0]
        self.nfeat = self.A.shape[1]
        self.Ab = np.c_[A.reshape(self.n, -1), b.reshape(self.n, 1)]
        
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

max_iter = 30000

    
# Define Parameters

alpha = 0.1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
seed = 123
batchsize = 1

np.random.seed(seed)

d = 1
n = 100
noise = np.random.rand(d,1)
s = np.random.rand(n,d) * 10 - 5          
t = s @ noise + np.random.randn(n,1)       #response (b)
S = np.column_stack((s, np.ones(n)))    #data (A)
x0 = np.zeros((d+1,1))

dataSet1 = gdDataADAM("Test data 1", S, t,max_iter,alpha,beta1,beta2,epsilon)

myData = dataSet1


#ADAM algorithm

#Least Squares Loss
w = np.copy(x0)
m = np.zeros((d+1,1))
v = np.zeros((d+1,1))
maxit = 5000
t = 0
w_list = []
counter = 0
# alpha = myData.Lip()

while(la.norm(myData.GradLS(w))>myData.epsilon):
    t += 1
    if t > myData.maxit:
        print("max iterations exceeded")
        break
    g = myData.sGradLS(w)   #update stochastic gradient
    m = myData.beta1 * m + (1 - myData.beta1) * g    #update biased first moment est
    v = myData.beta2 * v + (1 - myData.beta2) * g**2    #update biased second raw moment est
    mhat = m/(1 - myData.beta1**t)  #bias corrected first moment est
    vhat = v/(1 - myData.beta2**t)  #bias corrected second moment est
    w = w - ( myData.alpha * mhat / (np.sqrt(vhat) + epsilon) )
    
    #stash for graphs
    w_list.append(w)


#PLOTS
# gen graph data
obj_vals = []
grad_vals = []

for w in w_list:
    obj_vals.append(objective_function(w))
    grad_vals.append(la.norm(myData.GradLS(w)))
    

from matplotlib import pyplot as plt

plt.plot(grad_vals, 'o')
plt.show()

# vec = np.linspace(0, len(w_list))
# plt.figure()
plt.plot(w_list)  # plot the function values