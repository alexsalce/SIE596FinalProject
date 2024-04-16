# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:29:36 2024

@author: Alex Salce
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
import time


class gdDataSARAH():
    def __init__(self, A, b, outer_loop_iter, inner_loop_iter, batch_size, epsilon=None):
        self.A = A
        self.b = b
        self.outer_iter = outer_loop_iter  
        self.inner_iter = inner_loop_iter
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
    def muLS(self, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        hess = 2*A.T @ A
        mineig = min(np.linalg.eig(hess)[0])
        return mineig
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
        # grad = (2 * self.n / len(b)) *	A.T @ (A  @ x - b)
        grad = 2 *	A.T @ (A  @ x - b)
        return grad



# Fixing seed to generate the same random number every time
np.random.seed(123)

n = 10000  # Set the size of the matrix
m = 500  # Set the size of the matrix
x = np.zeros((m, 1))  # Initial point
# x = 50*np.ones((m, 1))  # Initial point
A = np.random.randn(n, m)  # Defining matrix A
A = A / np.linalg.norm(A)
b = A@(np.random.randn(m, 1)) + np.random.randn(n, 1)  # Defining vector b
outer_loop_iter = 4
epochs = 1
inner_loop_iter = 200 # Number of iterations
L = 2 * np.linalg.norm(A) ** 2
vec = []
batch_size = 1
eta = 1/(2*L)
# eta = np.sqrt(2/L)
epsilon = 1e-3




myData = gdDataSARAH(A, b, outer_loop_iter, inner_loop_iter, batch_size)

mu = myData.muLS()
sigmam = (1/(mu * eta* (inner_loop_iter+1))) + (eta * L / (2-(eta*L)))

kappa = L/mu

# inner_loop_iter = math.ceil(2*L-1)*100
# outer_loop_iter = math.ceil(np.log(np.linalg.norm(myData.GradLS(x))**2/epsilon/np.log(9/7)))



#initialize SARAH algorith parameters
wold = np.copy(x)
w_list = []
w_plot = []

for _ in range(outer_loop_iter):
    
    w = wold
    vold = myData.GradLS(w)
    w = wold - eta * vold
    
    for i in range(inner_loop_iter):
        i = np.random.permutation(myData.n)[:myData.batch]
        vnew = myData.sGradLS(w, i) - myData.sGradLS(wold, i) + vold
        wold = np.copy(w)
        w_list.append(wold)
        # w = w - (eta/np.sqrt(i+1)) * vnew
        w = w - eta * vnew
        vold = np.copy(vnew)
        w_plot.append(myData.objective_function_ls(w))
    
    wold = w_list[ np.random.permutation(range(0,len(w_list)))[0] ]
    

plt.show()

np.linalg.norm(myData.GradLS(w))


# while myData.objective_function_ls(w)>1e-3:
#     w = w - eta * myData.GradLS(w)

start = time.time()
gd_list = []
wgd = np.copy(x)
gditer = 0
while( la.norm(myData.GradLS(wgd)) > 1e-2 ):
    gditer+=1
    wgd += -1/L * myData.GradLS( wgd )
    gd_list.append( myData.objective_function_ls(wgd) )
    
end = time.time()
length = end - start
print(length, " seconds for gd to converge to 1e-2")

la.norm(myData.GradLS(wgd))
plt.plot(w_plot)
plt.plot(gd_list)
plt.show()

np.linalg.norm(myData.GradLS(wgd))
