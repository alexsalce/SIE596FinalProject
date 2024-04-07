# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:53:59 2024

@author: Alex Salce
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt



class gdDataADAM():
    def __init__(self, A, b, maxit, alpha, beta1, beta2, batch_size, epsilon):
        self.A = A
        self.b = b
        self.maxit = maxit        
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.maxit = maxit
        self.batch = batch_size
        self.epsilon = epsilon
        self.n = self.A.shape[0]
        self.nfeat = self.A.shape[1]
        self.Ab = np.c_[A.reshape(self.n, -1), b.reshape(self.n, 1)]
        
    def LipLS(self, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        L = 2 * la.norm(A)**2
        eta = 1/L
        return eta
    def objective_function_ls(self,x,A=None,b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        return np.linalg.norm(A@x - b)**2
    def sGradLS(self, x, indices, A=None, b=None,batch_size=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        x = x
        batch = self.batch if batch_size is None else batch_size
        i = indices
        Abatch = A[i, :]
        bbatch = b[i]
        grad = self.GradLS(x,Abatch,bbatch)
        return grad
    def GradLS(self, x, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        grad = (2 * self.n / len(b)) *	A.T @ (A  @ x - b)
        return grad
    
    
def objective_function(x,A,b):
    return np.linalg.norm(A@x - b)**2

max_iter = 140

    
# Define Parameters

alpha = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
seed = 123
batchsize = 2

np.random.seed(seed)

d = 2
n = 1000
noise = np.random.rand(d,1)
s = np.random.rand(n,d) * 10 - 5          
b = s @ noise + np.random.randn(n,1)       #response (b)
A = np.column_stack((s, np.ones(n)))    #data (A)
x0 = np.zeros((d+1,1))

dataSet1 = gdDataADAM(A, b,max_iter,alpha,beta1,beta2,batchsize,epsilon)

myData = dataSet1


#ADAM algorithm

#Least Squares Loss
w = np.copy(x0)
m = np.zeros((d+1,1))
v = np.zeros((d+1,1))
t = 0
w_list = []
counter = 0
# alpha = myData.Lip()


for _ in range(myData.maxit):
    t += 1
    #DO I NEED TO MAKE A SHUFFLED MASTER LIST TO RUN THRU BATCHES? 
    i = np.random.permutation(myData.n)[:myData.batch]
    g = myData.sGradLS(w,i)   #update stochastic gradient
    m = myData.beta1 * m + (1 - myData.beta1) * g    #update biased first moment est
    v = myData.beta2 * v + (1 - myData.beta2) * g**2    #update biased second raw moment est
    mhat = m/(1 - myData.beta1**t)  #bias correction first moment est
    vhat = v/(1 - myData.beta2**t)  #bias correction second moment est
    w = w - ( myData.alpha * mhat / (np.sqrt(vhat) + epsilon) )
    
    #stash for graphs
    w_list.append(w)


#PLOTS
# gen graph data
obj_vals = []
grad_vals = []

for w in w_list:
    obj_vals.append(objective_function(w,myData.A,myData.b))
    grad_vals.append(la.norm(myData.GradLS(w)))
    

plt.plot(grad_vals)
plt.show()


gd_list = []
wgd = np.copy(x0)
eta = myData.LipLS()
while( la.norm(myData.GradLS(wgd)) > 1e-2 ):
    wgd += -eta * myData.GradLS( wgd )
    gd_list.append( myData.objective_function_ls(wgd) )
la.norm(myData.GradLS(wgd))
plt.plot(gd_list)
plt.show()