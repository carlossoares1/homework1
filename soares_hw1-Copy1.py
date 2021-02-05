#!/usr/bin/env python
# coding: utf-8

# In[64]:


from scipy.stats import multivariate_normal
from numpy import random
from numpy import array
from numpy import zeros
import numpy as np
import numpy.matlib

# covariance = np.diag(sigma**2)
# z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
# # Reshape back to a (30, 30) grid.
# z = z.reshape(x.shape)

#a) Generate 25 3-d points...
mu1 = np.array([0, 0, 0])
mu2 = np.array([4,4,4])
#sigma = np.matlib.identity(1, dtype=int)
sigma = np.array([1,1,1])
print(sigma, mu1, mu2)
#set1 = multivariate_normal(mu1,sigma,(25,3))
#set2 = multivariate_normal(mu2,sigma,(25,3))
set1 = np.random.normal(mu1, sigma, (25, 3))
set2 = np.random.normal(mu2,sigma,size=(25,3))

print(f"Length of set 1: {len(set1)}", set1)
print(f"Length of set 2: {len(set2)}", set2)


# In[65]:


#zero = np.zeros((25,1))
one = np.ones((25,1))
set1c = np.concatenate((set1,one),axis=1)
#print(set1c)
one *= -1
set2c = np.concatenate((set2,one),axis=1)
#print(set2c)
set3 = np.concatenate((set1c,set2c),axis=0)
print(set3)


# In[3]:


#b) Compute beta**2
beta_square = 0
for i in set3:
    alfa = (np.linalg.norm(i))**2
    #print(alfa)
    if beta_square < alfa:
        beta_square = alfa
print("The max beta square is ", beta_square)


# In[63]:


#c) Find a hyperplane using batch algorithm

w = np.array([0,1,0])
lrate = 0.1
criterion = 0.05
p = 6
Lrate_step = 10
sum1 = np.array([0,0,0])
k = 0
while (p > criterion and k < 10000):
    for i in set3:
        x_array = np.delete(i,3)
        wtrp = np.transpose(w)
        #print(w, wtrp)
        if (i[-1]*(wtrp.dot(x_array))) < 0:
            sum1 = sum1 + x_array*i[-1]
            w = w + lrate * sum1
    p = np.linalg.norm(lrate * sum1)
    k += 1
    if Lrate_step % k == 0:
        lrate = lrate / 2
    print(f"iteration: {k} and p {p}")
    
print(f"This is w: {w} and final lrate {lrate}")


# In[ ]:


#initial lrate = 0.1
#iteration: 232 and p 0.0482035530655326
#This is w:  [ -8.7534995  -13.24464171  -7.36689506] and this is final lrate 0.00625
#initial lrate = 0.1
#iteration: 3 and p 0.04515762275977162
#This is w:  [-0.75193972 -1.89641177 -0.46393424]


# In[59]:


#d)choose an initial weight vactor w0 constrained to the surface of a 3D sphere of radius 0.1 
# (e.g. |wo| = 0.1; choose w0 = [0,0,0.1]
wzero = np.array([0,0.1,0])


# In[60]:


# Stochastic Perceptron

w = wzero
wtrp = np.transpose(w)
misclass = True
k = 0
m_count = 0
while (misclass and k < 100000):
    misclass = False
    for i in set3:      
        x_array = np.delete(i,3)
        #wtrp = np.transpose(w)
        #print(w, wtrp)
        if (i[-1]*(wtrp.dot(x_array))) < 0:
            w = w + i[-1]*x_array
            wtrp = np.transpose(w)
            misclass = True
            m_count += 1
    k+=1
print(f"iteration: {k} and misclassified hit: {m_count}")
print(w)


# In[ ]:


#iteration: 10000 and misclassified hit: 127144
#[-0.02748196 -4.77103709  1.09085898]
#iteration: 50000 and misclassified hit: 635773
#[-2.62969572 -7.50221965 -3.98329163]


# In[ ]:





# In[ ]:





# In[111]:





# In[112]:





# In[52]:


#f) compute gama
gama = 100
wtilt = w
#x_array = np.resize(set3,(50,3))
for i in set3:
    x_array = np.delete(i,3)
    wtrp = np.transpose(wtilt)
    #print(w_tilt, wtrp)
    #gama_test = (i.dot(wtrp))
    gama_test = (wtrp.dot(x_array))*i[-1]
    #print(gama_test)
    if (gama_test < gama and gama_test > 0):
        gama = gama_test
print("The gama value is ", gama)


# In[53]:


alfa = beta_square/gama
print(alfa)


# In[54]:


#wzero = [0,0,0.1]
#w_tilt = np.delete(weights,0)
kzero = ((np.linalg.norm(wzero - alfa*wtilt))**2 ) / beta_square
print(kzero)


# In[45]:


#Experimentin with w
#weight = [0.1,1,2,3] -> kzero = 8.5 and epoch(4)
#weight = [0.1,0,0,0] -> kzero = 8.5 and epoch(4)


# In[ ]:




