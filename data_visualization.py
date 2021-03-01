#!/usr/bin/env python
# coding: utf-8

# In[7]:


#import all the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import math


# In[8]:


DATA_PATH = "Data/HTRU_2.csv"

data = pd.read_csv(DATA_PATH)


# In[9]:


# make the correlation heatmaps on imbalanced scaled data

corr_matrix = data.corr()

corr_matrix.style.background_gradient(cmap='YlGn').set_precision(2)


# In[10]:


x = data.iloc[:, 0:7]

y = data.iloc[:, -1]


# In[12]:


df_class_0 = x[data['CLASS'] == 0]
df_class_1 = x[data['CLASS'] == 1]

def plot_graph_attributes(attri):
    for i in attri:
        for j in attri:
            if i!=j:
                plt.figure(figsize=(3.5,3))
                plt.scatter(df_class_0[i], df_class_0[j], color="#76ead7", alpha=0.1, s=4)
                plt.scatter(df_class_1[i], df_class_1[j], color="#2814ba", alpha=0.1, s=4)
                plt.xlabel(i)
                plt.ylabel(j)

                plt.show()


# In[14]:


attri=['MIP', 'SDIP', 'KIP', 'SIP', 'MDM']

plot_graph_attributes(attri)


# In[15]:


def plot_histogram(attri):
    for i in attri:
        plt.hist(x=data[i], bins=50, color='orange',alpha=0.7, rwidth=0.85)
        plt.xlabel(i)
        plt.ylabel("frequency")
        plt.show()


# In[16]:


plot_histogram(attri)


# In[17]:


def plot_heat_map(attri):
    for i in attri:
        for j in attri:
            if i!=j:
                plt.hist2d(data[i], data[j], bins =40) 
                plt.xlabel(i)
                plt.ylabel(j)

                plt.show()
    


# In[18]:


plot_heat_map(attri)


# In[ ]:




