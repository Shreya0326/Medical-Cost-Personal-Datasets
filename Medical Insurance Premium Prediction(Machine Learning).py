#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("insurance.csv")


# ## Data exploration

# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.select_dtypes(include='object').columns


# In[7]:


len(data.select_dtypes(include='object').columns)


# In[8]:


data.select_dtypes(include = ['int64','float64']).columns


# In[9]:


len(data.select_dtypes(include=['int64', 'float64']).columns)


# In[10]:


# Stastical Summary
data.describe()


# In[11]:


# Group Dataset into 'sex', 'smoker', 'region'


# In[12]:


data.groupby('sex').mean()


# In[13]:


data.groupby('smoker').mean()


# In[14]:


data.groupby('region').mean()


# ## Dealing with missing value

# In[15]:


data.isnull().values.any()


# In[16]:


data.isnull().values.sum()


# ## Working with categorical data

# In[17]:


data.select_dtypes(include='object').columns


# In[18]:


data['sex'].unique()


# In[19]:


data['smoker'].unique()


# In[20]:


data['region'].unique()


# In[21]:


data.head()


# In[22]:


# One hot encoading
data = pd.get_dummies(data=data,drop_first=True)


# In[23]:


data.head()


# In[24]:


data.shape


# ## Correlation Matrix

# In[25]:


data_2 =data.drop(columns='charges')


# In[26]:


data_2.corrwith(data['charges']).plot.bar(
figsize=(16,9),title='correlation with charges',rot=45,grid=True
)


# In[27]:


corr =data.corr()


# In[28]:


# Heatmap
plt.figure(figsize= (16,9))
sns.heatmap(corr,annot = True)


# ## Splitting the data

# In[29]:


data.head()


# In[30]:


# Matrix of features/ independent variables
x = data.drop(columns='charges')


# In[31]:


# Target / dependent variable
y = data['charges']


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[33]:


x_train.shape


# In[34]:


y_train.shape


# In[35]:


x_test.shape


# In[36]:


y_test.shape


# ## Feature Scaling

# In[37]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[38]:


x_train


# In[39]:


x_test


# ## Building the model

# ## Multiple Linear Regression

# In[40]:


from sklearn.linear_model import LinearRegression
regressor_lr = LinearRegression()
regressor_lr.fit(x_train,y_train)


# In[41]:


y_pred = regressor_lr.predict(x_test)


# In[42]:


from sklearn.metrics import r2_score


# In[43]:


r2_score(y_test,y_pred)


# ## Random Forest Regression

# In[44]:


from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor()
regressor_rf.fit(x_train,y_train)


# In[45]:


y_pred = regressor_rf.predict(x_test)


# In[46]:


r2_score(y_test,y_pred)


# ## XGBoost regression

# In[48]:


from xgboost import XGBRFRegressor
regressor_xgb = XGBRFRegressor()
regressor_xgb.fit(x_train,y_train)


# In[49]:


y_pred = regressor_xgb.predict(x_test)


# In[50]:


r2_score(y_test,y_pred)


# ## Predict charges for a new customer

# In[51]:


data.head()


# In[59]:


import warnings
warnings.filterwarnings('ignore')


# 1) Name: Titu, age: 40, sex: 1, bmi:45.50, children:4, smoker:1, region:northeast
# 

# In[60]:


titu_obs = [[40, 45.5, 4, 1, 1, 0, 0, 0]]


# In[61]:


regressor_xgb.predict(sc.transform(titu_obs))


# 2) Name: Sara, age:19, bmi:27.9, children:0, sex:female, smoker:no, region: northwest
# 

# In[62]:


sara_obs = [[19,27.9,0,0,0,1,0,0]]


# In[63]:


regressor_xgb.predict(sc.transform(sara_obs))


# In[ ]:




