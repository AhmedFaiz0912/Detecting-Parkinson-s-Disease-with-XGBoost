#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Make necessary imports

import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


#Import the dataset

df=pd.read_csv(r'F:\Projects\Detecting Parkinsons disease using XGBoost\parkinsons.data')
df.head()


# In[3]:


#Get the features and labels
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values


# In[4]:


#Get the count of each label (0 and 1) in labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])


# In[5]:


#Scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels


# In[6]:


#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)


# In[7]:


#Train the model
model=XGBClassifier()
model.fit(x_train,y_train)


# In[8]:


#Calculate the accuracy
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)

