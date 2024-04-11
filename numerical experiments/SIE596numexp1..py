# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 06:53:26 2024

@author: Alex Salce
"""

# import SIE596FinalAlgs
from SIE596FinalAlgs import gdDataADAM
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython import display
import matplotlib.pyplot as plt
import ffmpeg


np.random.seed(123)

d = 1
n = 100
xbar = np.random.rand(d,1)
s = np.random.rand(n,1) * 10 - 5
b = s * xbar + np.random.randn(n,1)


S = np.column_stack((s, np.ones(n)))
tol = 1e-4
x = np.zeros((d + 1,1))



#ADAM algorithm

# Define Parameters

alpha = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
seed = 123
batchsize = 2
max_iter = 140

np.random.seed(seed)

dataSet1 = gdDataADAM(S, b,max_iter,alpha,beta1,beta2,batchsize,epsilon)

myData = dataSet1

#Least Squares Loss
w = np.copy(x)
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
    g = myData.sGradLS(w,i)   #update batch gradient
    m = myData.beta1 * m + (1 - myData.beta1) * g    #update biased first moment est
    v = myData.beta2 * v + (1 - myData.beta2) * g**2    #update biased second raw moment est
    mhat = m/(1 - myData.beta1**t)  #bias correction first moment est
    vhat = v/(1 - myData.beta2**t)  #bias correction second moment est
    w = w - ( myData.alpha * mhat / (np.sqrt(vhat) + epsilon) )
    
    #stash for graphs
    w_list.append(w)



#plot initialization####

Figure = plt.figure()
lines_plotted = plt.plot([])  
line_plotted = lines_plotted[0]
 
plt.xlim(-5,5) 
plt.ylim(-5,5) 
vec = np.linspace(-5, 5, n)

def AnimationFunction(frame): 
 
    # setting y according to frame
    # number and + x. It's logic
    y = vec * w[0] + w[1]
 
    # line is set with new values of x and y
    line_plotted.set_data((vec, y))

anim_created = FuncAnimation(Figure, AnimationFunction, frames=n, interval=25)


video = anim_created.to_html5_video()
html = display.HTML(video)
display.display(html)
 
# good practice to close the plt object.
plt.close()

# plt.figure()
# plt.plot(s, b, '*')


####


# for w in w_list:
#     vec = np.linspace(-5, 5, n)
#     plt.plot(vec, vec * w[0] + w[1])
#     plt.show()
