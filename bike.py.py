#!/usr/bin/env python
# coding: utf-8

# In[121]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# Reading and understandig dataset

# In[122]:


df_raw = pd.read_csv('day.csv')


# In[123]:


display(df_raw.head(10))


# In[124]:


df_raw = df_raw.drop(['casual','registered'],axis=1)


# In[125]:


df_raw.shape


# In[126]:


#Replace all numerical data with meaningful values
df_raw['season'] = df_raw['season'].replace({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
df_raw['weathersit'] = df_raw['weathersit'].replace({1: 'Clear, Few clouds, Partly cloudy, Partly cloudy', 2: 'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist', 3: 'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds', 4: 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog'})
df_raw['yr'] = df_raw['yr'].replace({0: '2018', 1: '2019'})


# In[127]:


df_raw.info


# In[128]:


df_raw.describe()


# In[129]:


display(df_raw.isnull().sum())


# Step 2: Visualizing dataset

# In[130]:


num_cols = ["temp","atemp","hum","windspeed","cnt"]

for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set(style="whitegrid")
    
    
    # Histplot on the first subplot
    sns.histplot(x=df_raw[col], ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')
    axes[0].tick_params(axis='x', rotation=45)

    # Boxplot on the second subplot
    sns.boxplot(x=df_raw[col], ax=axes[1])
    axes[1].set_title(f'Boxplot of {col}')
    axes[1].tick_params(axis='x', rotation=45)
    
    
    plt.show()


# In[131]:


# Correlation matrix
corr_matrix = df_raw[num_cols].corr()

# Heatmap
plt.figure(figsize=(25, 25))
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Heatmap')
plt.show()


# In[132]:


#visualizing categorical variables
plt.figure(figsize=(20,20))
plt.subplot(331)
sns.boxplot(x='season', y='cnt',data=df_raw)
plt.subplot(332)
sns.boxplot(x='yr', y='cnt',data=df_raw)
plt.subplot(333)
sns.boxplot(x='mnth', y='cnt',data=df_raw)
plt.subplot(334)
sns.boxplot(x='holiday', y='cnt',data=df_raw)
plt.subplot(335)
sns.boxplot(x='weekday', y='cnt',data=df_raw)
plt.subplot(336)
sns.boxplot(x='weathersit', y='cnt',data=df_raw)
plt.show();


# In[ ]:





# In[133]:


sns.pairplot(df_raw[["temp","atemp","hum","windspeed","cnt"]])
plt.show()


# Here we can see that temp vs atemp and registered vs cnt is showing a positive linear relationship so, we may have to drop these coloumns while building our model.
# 

# In[134]:


df_raw.drop('atemp',axis=1, inplace=True)
df_raw.info()


# Preparing the Data for Modeling

# In[135]:


# dropping instant and dteday column for modeling.
df_raw=df_raw.drop(('instant'), axis=1)
df_raw.head()


# In[136]:


df_raw=df_raw.drop(('dteday'),axis=1)
df_raw.head()


# Getting dummies for our categorical variables.
# 

# In[137]:


df_raw['season']=df_raw['season'].astype('category')
df_raw['weathersit']=df_raw['weathersit'].astype('category')
df_raw['mnth']=df_raw['mnth'].astype('category')
df_raw['weekday']=df_raw['weekday'].astype('category')


# In[138]:


df_raw = pd.get_dummies(df_raw, drop_first=True)


# In[139]:


df_raw.head()


# Splitting the Data into Train and Test

# In[140]:


from sklearn.model_selection import train_test_split


# In[141]:


np.random.seed(0)
df_train, df_test = train_test_split(df_raw, train_size = 0.70, test_size = 0.30, random_state = 333)


# In[142]:


print(df_train.shape)
print(df_test.shape)


# In[143]:


#Rescalling
scaler= MinMaxScaler()
num_vars=['temp','hum','windspeed','cnt']
df_train[num_vars]=scaler.fit_transform(df_train[num_vars])
df_train.head()


# In[144]:


y_train= df_train.pop('cnt')
X_train=df_train


# In[145]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[146]:


rfe = RFE(lm)             # running RFE
rfe = rfe.fit(X_train, y_train)


# In[147]:


col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
X_train_rfe = X_train[col]


# In[148]:


print(X_train_rfe.head())
print(X_train_rfe.info())


# In[149]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[150]:


vif = pd.DataFrame()
vif['Parameters'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[151]:


import statsmodels.api as sm

# Add a constant from train set
X_train_lm1 = sm.add_constant(X_train_rfe)

# Create a first fitted Linear Regression model using VIF check
lr1 = sm.OLS(y_train, X_train_lm1).fit()
print(lr1.params)
print (lr1.summary(), '\n')


# In[152]:


X_train_rfe = X_train[col]


# In[153]:


# Adding a constant variable   
X_train_rfe = sm.add_constant(X_train_rfe)


# In[154]:


lm = sm.OLS(y_train,X_train_rfe).fit()   #


# In[155]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[156]:


# dropping the varible const.
X_train_new = X_train_rfe.drop(["const"], axis = 1)


# In[157]:


# rebuilding the model without const.
X_train_lm = sm.add_constant(X_train_new)


# In[158]:


lm = sm.OLS(y_train,X_train_lm).fit()


# In[159]:


#Let's see the summary of our linear model
print(lm.summary())


# In[160]:


# calculating the vif for our new model.
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[161]:


# dropping the varible hum.
X_train_1 = X_train_new.drop(["hum"], axis = 1)


# In[162]:


# rebuilding the model without const.
X_train_lm = sm.add_constant(X_train_1)


# In[163]:


lm = sm.OLS(y_train,X_train_lm).fit()


# In[164]:


#Let's see the summary of our linear model
print(lm.summary())


# In[165]:


# calculating vif for our new model.
vif = pd.DataFrame()
X = X_train_1
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[166]:


# dropping the varible temp.
X_train_2 = X_train_1.drop(["temp"], axis = 1)

# rebuilding the model without const.
X_train_lm = sm.add_constant(X_train_2)

lm = sm.OLS(y_train,X_train_lm).fit()

#Let's see the summary of our linear model
print(lm.summary())


# In[167]:


# calculating vif for our new model.
vif = pd.DataFrame()
X = X_train_2
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# This model looks good, as there seems to be VERY LOW Multicollinearity between the predictors and the p-values for all the predictors seems to be significant. For now, we will consider this as our final model (unless the Test data metrics are not significantly close to this number).

# Residual analysis
# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the displot of the error terms and see what it looks like.

# In[168]:


y_train_cnt = lm.predict(X_train_lm)


# In[169]:


# Plot the Graph of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins = 20)
fig.suptitle('Error Terms', fontsize = 10)                  # Plot heading 
plt.xlabel('Errors', fontsize = 10) 


# Here we can see our error terms terms are normally distributed.
# 
# Making predictions
# scaling the test data.

# In[170]:


df_test[num_vars] = scaler.transform(df_test[num_vars])


# In[171]:


y_test = df_test.pop('cnt')
X_test = df_test


# In[172]:


# Now let's use our model to make predictions.

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_2.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)


# In[173]:


# Making predictions
y_pred = lm.predict(X_test_new)


# Model Evaluation.

# In[174]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)     


# In[175]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# Company can continue with the stratergy used in 2019.
# They can offer a special discount on holidays for public to use bikes over other sevices.
# windspeed is a factor which can't be controlled by the comapny, so they can think another possible way to overcome this issue.
# They can offer coupons in spring season to encourage public to use bikes.
# Rainy weather is the most affecting weather for the comapny as people avoid going out on bikes in rainy season.

# In[ ]:




