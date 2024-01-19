#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder  #for train test splitting
from sklearn.model_selection import train_test_split #for decision tree object
from sklearn.tree import DecisionTreeClassifier #for checking testing results


# In[2]:


# Pandas is used for data manipulation
import pandas as pd
# Read in data and display first 5 rows
features = pd.read_csv('E:/Python docs/company_Data.csv')
features.head(5)


# In[3]:


#getting information of dataset
features.info()


# In[4]:


print('The shape of our features is:', features.shape)


# In[5]:


features.isnull().any()


# In[6]:


# let's plot pair plot to visualise the attributes all at once
sns.pairplot(data=features, hue = 'ShelveLoc')


# In[7]:


#Creating dummy vairables dropping first dummy variable
df=pd.get_dummies(features,columns=['Urban','US'], drop_first=True)


# In[8]:


print(df.head())


# In[9]:


from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[10]:


df['ShelveLoc']=df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})


# In[11]:


print(df.head())


# In[12]:


x=df.iloc[:,0:6]
y=df['ShelveLoc']
x


# In[13]:


y


# In[14]:


df['ShelveLoc'].unique()


# In[15]:


df.ShelveLoc.value_counts()


# In[23]:


colnames = list(df.columns)
colnames


# In[24]:


# Descriptive statistics for each column
df.describe()


# In[25]:


df.head()


# In[26]:


# labels is the values we want to predict
labels = np.array(df['Income'])
# Remove the labels from the features
# axis 1 refers to the columns
features = df.drop('Income', axis = 1)
# Saving feature names for later use
features_list = list(df.columns)
# Convert to numpy array
features = np.array(df)


# In[27]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[28]:


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# # Establish Baseline

# In[34]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)


# In[35]:


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# Determine Performance Metrics

# In[33]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:





# In[ ]:




