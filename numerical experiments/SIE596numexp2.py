# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:17:19 2024

@author: Alex Salce
"""
# import SIE596FinalAlgs
from SIE596FinalAlgs import gdDataADAM, gdDataSARAH, objective_function
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from IPython import display
import matplotlib.pyplot as plt
import ffmpeg
import math
import numpy.linalg as la
import time



#   __            __               
#  /\ \          /\ \__            
#  \_\ \     __  \ \ ,_\    __     
#  /'_` \  /'__`\ \ \ \/  /'__`\   
# /\ \L\ \/\ \L\.\_\ \ \_/\ \L\.\_ 
# \ \___,_\ \__/.\_\\ \__\ \__/.\_\
#  \/__,_ /\/__/\/_/ \/__/\/__/\/_/
                                 
                                 
# Fixing seed to generate the same random number every time
np.random.seed(123)

n = 10000  # Set the size of the matrix
m = 500  # Set the size of the matrix
x = np.zeros((m, 1))  # Initial point
# x = 50*np.ones((m, 1))  # Initial point
A = np.random.randn(n, m)  # Defining matrix A
A = A / np.linalg.norm(A)
b = A@(np.random.randn(m, 1)) + np.random.randn(n, 1)  # Defining vector b


#                 __                  ___                         
#  __          __/\ \__  __          /\_ \    __                  
# /\_\    ___ /\_\ \ ,_\/\_\     __  \//\ \  /\_\  ____      __   
# \/\ \ /' _ `\/\ \ \ \/\/\ \  /'__`\  \ \ \ \/\ \/\_ ,`\  /'__`\ 
#  \ \ \/\ \/\ \ \ \ \ \_\ \ \/\ \L\.\_ \_\ \_\ \ \/_/  /_/\  __/ 
#   \ \_\ \_\ \_\ \_\ \__\\ \_\ \__/.\_\/\____\\ \_\/\____\ \____\
#    \/_/\/_/\/_/\/_/\/__/ \/_/\/__/\/_/\/____/ \/_/\/____/\/____/                                                            
#  ____    ______  ____    ______  __  __     
# /\  _`\ /\  _  \/\  _`\ /\  _  \/\ \/\ \    
# \ \,\L\_\ \ \L\ \ \ \L\ \ \ \L\ \ \ \_\ \   
#  \/_\__ \\ \  __ \ \ ,  /\ \  __ \ \  _  \  
#    /\ \L\ \ \ \/\ \ \ \\ \\ \ \/\ \ \ \ \ \ 
#    \ `\____\ \_\ \_\ \_\ \_\ \_\ \_\ \_\ \_\
#     \/_____/\/_/\/_/\/_/\/ /\/_/\/_/\/_/\/_/
                                            
                                            
outer_loop_iter = 10
inner_loop_iter = 317 # Number of iterations
L = 2 * np.linalg.norm(A) ** 2
vec = []
batchsize_sarah = 1
eta_sarah = 1/(2*L)
# epochs = outer_loop_iter * inner_loop_iter / len(A)
epsilon_sarah = 1e-2


SARAHdata = gdDataSARAH(A, b, outer_loop_iter, inner_loop_iter, batchsize_sarah)

eta_sarah = 1/(2*SARAHdata.LipLS())
mu = SARAHdata.muLS()
sigmam = (1/(mu * eta_sarah* (inner_loop_iter+1))) + (eta_sarah * L / (2-(eta_sarah*L)))

kappa = L/mu
#OPTIONAL FOR STRONG CONVEX CASE EPSILON CONVERGENCE
# outer_loop_iter = math.ceil(np.log(np.linalg.norm(SARAHdata.GradLS(x))**2/epsilon_sarah/np.log(9/7)))
# inner_loop_iter = math.ceil(4.5*kappa)



#                 __                  ___                         
#  __          __/\ \__  __          /\_ \    __                  
# /\_\    ___ /\_\ \ ,_\/\_\     __  \//\ \  /\_\  ____      __   
# \/\ \ /' _ `\/\ \ \ \/\/\ \  /'__`\  \ \ \ \/\ \/\_ ,`\  /'__`\ 
#  \ \ \/\ \/\ \ \ \ \ \_\ \ \/\ \L\.\_ \_\ \_\ \ \/_/  /_/\  __/ 
#   \ \_\ \_\ \_\ \_\ \__\\ \_\ \__/.\_\/\____\\ \_\/\____\ \____\
#    \/_/\/_/\/_/\/_/\/__/ \/_/\/__/\/_/\/____/ \/_/\/____/\/____/
#  ______  ____    ______              
# /\  _  \/\  _`\ /\  _  \  /'\_/`\    
# \ \ \L\ \ \ \/\ \ \ \L\ \/\      \   
#  \ \  __ \ \ \ \ \ \  __ \ \ \__\ \  
#   \ \ \/\ \ \ \_\ \ \ \/\ \ \ \_/\ \ 
#    \ \_\ \_\ \____/\ \_\ \_\ \_\\ \_\
#     \/_/\/_/\/___/  \/_/\/_/\/_/ \/_/


alpha = 0.5
beta1 = 0.9
beta2 = 0.999
epsilon_adam = 1e-8
batchsize_adam = 100
max_iter_adam = 3170




ADAMdata = gdDataADAM(A, b,max_iter_adam,alpha,beta1,beta2,batchsize_adam,epsilon_adam)


#  ____    ______  ____    ______  __  __     
# /\  _`\ /\  _  \/\  _`\ /\  _  \/\ \/\ \    
# \ \,\L\_\ \ \L\ \ \ \L\ \ \ \L\ \ \ \_\ \   
#  \/_\__ \\ \  __ \ \ ,  /\ \  __ \ \  _  \  
#    /\ \L\ \ \ \/\ \ \ \\ \\ \ \/\ \ \ \ \ \ 
#    \ `\____\ \_\ \_\ \_\ \_\ \_\ \_\ \_\ \_\
#     \/_____/\/_/\/_/\/_/\/ /\/_/\/_/\/_/\/_/
#         ___                              __    __                     
#        /\_ \                          __/\ \__/\ \                    
#    __  \//\ \      __     ___   _ __ /\_\ \ ,_\ \ \___     ___ ___    
#  /'__`\  \ \ \   /'_ `\  / __`\/\`'__\/\ \ \ \/\ \  _ `\ /' __` __`\  
# /\ \L\.\_ \_\ \_/\ \L\ \/\ \L\ \ \ \/ \ \ \ \ \_\ \ \ \ \/\ \/\ \/\ \ 
# \ \__/.\_\/\____\ \____ \ \____/\ \_\  \ \_\ \__\\ \_\ \_\ \_\ \_\ \_\
#  \/__/\/_/\/____/\/___L\ \/___/  \/_/   \/_/\/__/ \/_/\/_/\/_/\/_/\/_/
#                    /\____/                                            
#                    \_/__/                                             

sarah_start = time.time()

#multiple outer loop SARAH
np.random.seed(123)
myData = SARAHdata
wold = np.copy(x)
w_list_sarah = []
sarah_obj_plot = []

for _ in range(outer_loop_iter):
    
    w = wold
    vold = myData.GradLS(w)
    w = wold - eta_sarah * vold
    
    for i in range(inner_loop_iter):
        i = np.random.permutation(myData.n)[:myData.batch]
        vnew = myData.sGradLS(w, i) - myData.sGradLS(wold, i) + vold
        wold = np.copy(w)
        w_list_sarah.append(wold)
        # w = w - (eta/np.sqrt(i+1)) * vnew
        w = w - eta_sarah * vnew
        vold = np.copy(vnew)
        sarah_obj_plot.append(myData.objective_function_ls(w))
    
    wold = w_list_sarah[ np.random.permutation(range(0,len(w_list_sarah)))[0] ]

sarah_end = time.time()
sarah_time = sarah_end - sarah_start
print(sarah_time, " seconds to run SARAH")    


#         _       _          _       _           _        
#        / /\    /\ \    _ / /\     /\ \        /\ \      
#       / /  \   \ \ \  /_/ / /    /  \ \      /  \ \     
#      / / /\ \__ \ \ \ \___\/    / /\ \ \    / /\ \_\    
#     / / /\ \___\/ / /  \ \ \   / / /\ \_\  / / /\/_/    
#     \ \ \ \/___/\ \ \   \_\ \ / / /_/ / / / / / ______  
#      \ \ \       \ \ \  / / // / /__\/ / / / / /\_____\ 
#  _    \ \ \       \ \ \/ / // / /_____/ / / /  \/____ / 
# /_/\__/ / /        \ \ \/ // / /\ \ \  / / /_____/ / /  
# \ \/___/ /          \ \  // / /  \ \ \/ / /______\/ /   
#  \_____\/            \_\/ \/_/    \_\/\/___________/    
                                                        
#         ___                              __    __                     
#        /\_ \                          __/\ \__/\ \                    
#    __  \//\ \      __     ___   _ __ /\_\ \ ,_\ \ \___     ___ ___    
#  /'__`\  \ \ \   /'_ `\  / __`\/\`'__\/\ \ \ \/\ \  _ `\ /' __` __`\  
# /\ \L\.\_ \_\ \_/\ \L\ \/\ \L\ \ \ \/ \ \ \ \ \_\ \ \ \ \/\ \/\ \/\ \ 
# \ \__/.\_\/\____\ \____ \ \____/\ \_\  \ \_\ \__\\ \_\ \_\ \_\ \_\ \_\
#  \/__/\/_/\/____/\/___L\ \/___/  \/_/   \/_/\/__/ \/_/\/_/\/_/\/_/\/_/
#                    /\____/                                            
#                    \_/__/                                             

svrg_start = time.time()

#multiple outer loop SARAH
np.random.seed(123)
myData = SARAHdata
wold = np.copy(x)
w_list_svrg = []
svrg_obj_plot = []

for _ in range(outer_loop_iter):
    
    w = wold
    vold = myData.GradLS(w)
    w = wold - eta_sarah * vold
    w0 = np.copy(w)
    
    for i in range(inner_loop_iter):
        i = np.random.permutation(myData.n)[:myData.batch]
        vnew = myData.sGradLS(w, i) - myData.sGradLS(w0, i) + vold
        wold = np.copy(w)
        w_list_svrg.append(wold)
        # w = w - (eta/np.sqrt(i+1)) * vnew
        w = w - eta_sarah * vnew
        svrg_obj_plot.append(myData.objective_function_ls(w))
    
    wold = w_list_svrg[ np.random.permutation(range(0,len(w_list_svrg)))[0] ]

svrg_end = time.time()
svrg_time = svrg_end - svrg_start
print(svrg_time, " seconds to run SVRG")          
        

#  ______  ____    ______              
# /\  _  \/\  _`\ /\  _  \  /'\_/`\    
# \ \ \L\ \ \ \/\ \ \ \L\ \/\      \   
#  \ \  __ \ \ \ \ \ \  __ \ \ \__\ \  
#   \ \ \/\ \ \ \_\ \ \ \/\ \ \ \_/\ \ 
#    \ \_\ \_\ \____/\ \_\ \_\ \_\\ \_\
#     \/_/\/_/\/___/  \/_/\/_/\/_/ \/_/
#         ___                              __    __                     
#        /\_ \                          __/\ \__/\ \                    
#    __  \//\ \      __     ___   _ __ /\_\ \ ,_\ \ \___     ___ ___    
#  /'__`\  \ \ \   /'_ `\  / __`\/\`'__\/\ \ \ \/\ \  _ `\ /' __` __`\  
# /\ \L\.\_ \_\ \_/\ \L\ \/\ \L\ \ \ \/ \ \ \ \ \_\ \ \ \ \/\ \/\ \/\ \ 
# \ \__/.\_\/\____\ \____ \ \____/\ \_\  \ \_\ \__\\ \_\ \_\ \_\ \_\ \_\
#  \/__/\/_/\/____/\/___L\ \/___/  \/_/   \/_/\/__/ \/_/\/_/\/_/\/_/\/_/
#                    /\____/                                            
#                    \_/__/                                             

adam_start = time.time()

np.random.seed(123)
myData = ADAMdata
w = np.copy(x)
m = np.copy(x)
v = np.copy(x)
t = 0
w_list_adam = []
adam_obj_plot = []


for _ in range(myData.maxit):
    t += 1
    i = np.random.permutation(myData.n)[:myData.batch]
    g = myData.sGradLS(w,i)   #update batch gradient
    m = myData.beta1 * m + (1 - myData.beta1) * g    #update biased first moment est
    v = myData.beta2 * v + (1 - myData.beta2) * g**2    #update biased second raw moment est
    mhat = m/(1 - myData.beta1**t)  #bias correction first moment est
    vhat = v/(1 - myData.beta2**t)  #bias correction second moment est
    w = w - ( alpha * mhat / (np.sqrt(vhat) + epsilon_adam) )
    
    #stash for graphs
    w_list_adam.append(w)
    adam_obj_plot.append(objective_function(w,A,b))

adam_end = time.time()
adam_time = adam_end - adam_start
print(adam_time, " seconds to run ADAM")          
    
#  __  __  ______  __  __  ______   __       __       ______         
# /\ \/\ \/\  _  \/\ \/\ \/\__  _\ /\ \     /\ \     /\  _  \        
# \ \ \ \ \ \ \L\ \ \ `\\ \/_/\ \/ \ \ \    \ \ \    \ \ \L\ \       
#  \ \ \ \ \ \  __ \ \ , ` \ \ \ \  \ \ \  __\ \ \  __\ \  __ \      
#   \ \ \_/ \ \ \/\ \ \ \`\ \ \_\ \__\ \ \L\ \\ \ \L\ \\ \ \/\ \     
#    \ `\___/\ \_\ \_\ \_\ \_\/\_____\\ \____/ \ \____/ \ \_\ \_\    
#     `\/__/  \/_/\/_/\/_/\/_/\/_____/ \/___/   \/___/   \/_/\/_/    
                                                                   
                                                                   
#  ____    ____    ______  ____    ______   ____    __  __  ______   
# /\  _`\ /\  _`\ /\  _  \/\  _`\ /\__  _\ /\  _`\ /\ \/\ \/\__  _\  
# \ \ \L\_\ \ \L\ \ \ \L\ \ \ \/\ \/_/\ \/ \ \ \L\_\ \ `\\ \/_/\ \/  
#  \ \ \L_L\ \ ,  /\ \  __ \ \ \ \ \ \ \ \  \ \  _\L\ \ , ` \ \ \ \  
#   \ \ \/, \ \ \\ \\ \ \/\ \ \ \_\ \ \_\ \__\ \ \L\ \ \ \`\ \ \ \ \ 
#    \ \____/\ \_\ \_\ \_\ \_\ \____/ /\_____\\ \____/\ \_\ \_\ \ \_\
#     \/___/  \/_/\/ /\/_/\/_/\/___/  \/_____/ \/___/  \/_/\/_/  \/_/
                                                                   
                                                                   
#  ____    ____    ____    ____     ____    __  __  ______           
# /\  _`\ /\  _`\ /\  _`\ /\  _`\  /\  _`\ /\ \/\ \/\__  _\          
# \ \ \/\ \ \ \L\_\ \,\L\_\ \ \/\_\\ \ \L\_\ \ `\\ \/_/\ \/          
#  \ \ \ \ \ \  _\L\/_\__ \\ \ \/_/_\ \  _\L\ \ , ` \ \ \ \          
#   \ \ \_\ \ \ \L\ \/\ \L\ \ \ \L\ \\ \ \L\ \ \ \`\ \ \ \ \         
#    \ \____/\ \____/\ `\____\ \____/ \ \____/\ \_\ \_\ \ \_\        
#     \/___/  \/___/  \/_____/\/___/   \/___/  \/_/\/_/  \/_/        
   
                        
gd_start = time.time()

gd_obj_plot = []
wgd = np.copy(x)
gditer = 0
while( la.norm(myData.GradLS(wgd)) > 1e-2 ):
    gditer+=1
    wgd += -1/L * myData.GradLS( wgd )
    gd_obj_plot.append( np.linalg.norm(A@wgd - b)**2 )
    
gd_end = time.time()
gd_time = gd_end - gd_start
print(gd_time, " seconds for gd to converge to 1e-2")


#  ____    __       _____   ______  ____       
# /\  _`\ /\ \     /\  __`\/\__  _\/\  _`\     
# \ \ \L\ \ \ \    \ \ \/\ \/_/\ \/\ \,\L\_\   
#  \ \ ,__/\ \ \  __\ \ \ \ \ \ \ \ \/_\__ \   
#   \ \ \/  \ \ \L\ \\ \ \_\ \ \ \ \  /\ \L\ \ 
#    \ \_\   \ \____/ \ \_____\ \ \_\ \ `\____\
#     \/_/    \/___/   \/_____/  \/_/  \/_____/
#       __                          __                          
#      /\ \       __               /\ \__  __                   
#   ___\ \ \____ /\_\     __    ___\ \ ,_\/\_\  __  __     __   
#  / __`\ \ '__`\\/\ \  /'__`\ /'___\ \ \/\/\ \/\ \/\ \  /'__`\ 
# /\ \L\ \ \ \L\ \\ \ \/\  __//\ \__/\ \ \_\ \ \ \ \_/ |/\  __/ 
# \ \____/\ \_,__/_\ \ \ \____\ \____\\ \__\\ \_\ \___/ \ \____\
#  \/___/  \/___//\ \_\ \/____/\/____/ \/__/ \/_/\/__/   \/____/
#                \ \____/                                       
#                 \/___/                                                                                  
                                             

plt.plot(sarah_obj_plot)
plt.plot(svrg_obj_plot)
plt.plot(adam_obj_plot)
plt.plot(gd_obj_plot)
plt.show()

#svrg and sarah delta
svrg_saraa_delta = np.array(svrg_obj_plot) - np.array(sarah_obj_plot)
plt.plot(svrg_saraa_delta)
plt.show()