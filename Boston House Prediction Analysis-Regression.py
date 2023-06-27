#!/usr/bin/env python
# coding: utf-8

# ## Dataset Information

# Boston House Prices Dataset was collected in 1978 and has 506 entries with 14 attributes (or) features for homes from various suburbs in Boston.
# 
# Attribute Information:
#         - CRIM     	per capita crime rate by town
# 
#         - ZN       	proportion of residential land zoned for lots over 25,000 sq.ft.
# 
#         - INDUS   	proportion of non-retail business acres per town
# 
#         - CHAS   	Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 
#         - NOX        nitric oxides concentration (parts per 10 million)
# 
#         - RM       	average number of rooms per dwelling
# 
#         - AGE     	proportion of owner-occupied units built prior to 1940
# 
#         - DIS      	weighted distances to five Boston employment centers
# 
#         - RAD      	index of accessibility to radial highways
# 
#         - TAX      	full-value property-tax rate per $10,000
# 
#         - PTRATIO pupil-teacher ratio by town
# 
#         - B        	1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 
#         - LSTAT    % lower status of the population
# 
#         - MEDV     Median value of owner-occupied homes in $1000's

# ## Import Modulues

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ## Loading the dataset

# In[2]:


df = pd.read_csv("Boston Dataset.csv")
df.drop(columns=['Unnamed: 0'], axis=0, inplace=True)
df.head()


# In[3]:


# statistical info
df.describe()


# In[4]:


# datatype info
df.info()


# ## Preprocessing the dataset

# In[5]:


# check for null values
df.isnull().sum()


# ## Exploratory Data Analysis

# In[6]:


# create box plots
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[7]:


# create dist plot
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.distplot(value, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# ## Min-Max Normalization

# In[8]:


cols = ['crim', 'zn', 'tax', 'black']
for col in cols:
    # find minimum and maximum of that column
    minimum = min(df[col])
    maximum = max(df[col])
    df[col] = (df[col] - minimum) / (maximum - minimum)


# In[9]:


fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.distplot(value, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[10]:


# standardization
from sklearn import preprocessing
scalar = preprocessing.StandardScaler()

# fit our data
scaled_cols = scalar.fit_transform(df[cols])
scaled_cols = pd.DataFrame(scaled_cols, columns=cols)
scaled_cols.head()


# In[11]:


for col in cols:
    df[col] = scaled_cols[col]


# In[12]:


fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.distplot(value, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# ## Correlation Matrix

# In[13]:


corr = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# ## Relation between Target and Correlated Variables

# In[14]:


sns.regplot(y=df['medv'], x=df['lstat'])


# In[15]:


sns.regplot(y=df['medv'], x=df['rm'])


# ## Input Split

# In[16]:


X = df.drop(columns=['medv', 'rad'], axis=1)
y = df['medv']


# ## Model Training

# In[17]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
def train(model, X, y):
    # train the model
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model.fit(x_train, y_train)
    
    # predict the training set
    pred = model.predict(x_test)
    
    # perform cross-validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    print("Model Report")
    print("MSE:",mean_squared_error(y_test, pred))
    print('CV Score:', cv_score)


# ### Linear Regression:

# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Assuming X and y are your input features and target variable, respectively

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the LinearRegression model
model = LinearRegression()
train(model, X, y)

# Get and sort the coefficients
coef = pd.Series(model.coef_, X.columns).sort_values()

# Plot the coefficients
coef.plot(kind='bar', title='Model Coefficients')


# ### Decision Tree:

# In[22]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Feature Importance')


# ### Random Forest:

# In[25]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Feature Importance')


# ### Extra Trees:

# In[27]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Feature Importance')


# ### XGBoost:

# In[29]:


import xgboost as xgb
model = xgb.XGBRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Feature Importance')

