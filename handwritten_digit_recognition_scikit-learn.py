#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn import datasets
handwritten_digits = datasets.load_digits()
type(handwritten_digits)


# In[3]:


print(len(handwritten_digits.images))
print("\n")
print(type(handwritten_digits.images))
print("\n")
print(len(handwritten_digits.target))
print("\n")
print(type(handwritten_digits.target))


# In[4]:


print(handwritten_digits.images.shape)
print("\n")
print(handwritten_digits.target.shape)


# In[5]:


images = handwritten_digits.images
labels_assign = handwritten_digits.target
images = images.reshape((images.shape[0], -1))
images.shape


# In[6]:


handwritten_digits.images[7]


# In[7]:


import matplotlib.pyplot as plt
plt.gray() 
imgplot = plt.imshow(handwritten_digits.images[7])
print("labels_assign: ",handwritten_digits.target[7])

plt.show()


# In[8]:


from sklearn import svm
classifier = svm.SVC(gamma=0.007)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels_assign, test_size=0.25, random_state=1001)


# In[9]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[10]:


classifier.fit(X_train, y_train)


# In[11]:


score = classifier.score(X_test,y_test)
score


# In[12]:


plt.gray() 
imgplot = plt.imshow(handwritten_digits.images[5])
print("label: ",handwritten_digits.target[5])
plt.show()
print("\n")
imgplot = plt.imshow(handwritten_digits.images[7])
print("label: ",handwritten_digits.target[7])
plt.show() 


# In[ ]:




