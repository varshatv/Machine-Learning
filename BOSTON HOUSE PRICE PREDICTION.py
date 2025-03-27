#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


bh=pd.read_csv(r"C:\Users\Uer\Downloads\boston.csv")


# In[5]:


bh.info()


# In[6]:


bh.head()


# In[7]:


bh.isna().sum()


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X=bh.drop(columns=['MEDV'])
y=bh['MEDV']


# In[10]:


X


# In[11]:


y


# In[12]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=.85,random_state=50)


# In[13]:


X_train.shape


# In[14]:


X_test.shape


# In[15]:


X_train.head()


# In[16]:


y_train.head()


# In[130]:


#Linear Regression


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


lrmodel=LinearRegression()
lrmodel


# In[19]:


lrmodel.fit(X_train,y_train)


# In[20]:


lrmodel.score(X_train,y_train)


# In[84]:


y_pred_lr=lrmodel.predict(X_test)


# In[85]:


X_test


# In[86]:


y_test


# In[87]:


y_pred_lr


# In[26]:


from sklearn.metrics import mean_squared_error


# In[88]:


print ("Linear Regression:",mean_squared_error(y_test,y_pred_lr,squared=False))


# In[89]:


plt.plot(l,y_test)
plt.plot(l,y_pred_lr,'red')


# In[131]:


#Decision Tree


# In[28]:


from sklearn.tree import DecisionTreeRegressor


# In[29]:


dtmodel=DecisionTreeRegressor()
dtmodel.fit(X_train,y_train)
y_pred_dt=dtmodel.predict(X_test)


# In[30]:


print ("DecisionTree Regressor:",mean_squared_error(y_test,y_pred_dt,squared=False))


# In[81]:


plt.plot(l,y_test)
plt.plot(l,y_pred_dt,'red')


# In[ ]:


#SVR


# In[31]:


from sklearn.svm import SVR


# In[32]:


from sklearn.preprocessing import StandardScaler


# In[33]:


Scale=StandardScaler()


# In[45]:


Scale_X_train=Scale.fit_transform(X_train)
Scale_X_test=Scale.fit_transform(X_test)


# In[35]:


svrmodel=SVR()
svrmodel


# In[36]:


svrmodel.fit(Scale_X_train,y_train)


# In[37]:


y_pred_svr=svrmodel.predict(Scale_X_test)


# In[72]:


len(y_test)


# In[73]:


l=[i for i in range(76)]
len(l)


# In[82]:


plt.plot(l,y_test)
plt.plot(l,y_pred_svr,'red')


# In[38]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[39]:


mse=mean_squared_error(y_test,y_pred_svr)
print ("Mean Squared Error:",mse)
mae=mean_absolute_error(y_test,y_pred_svr)
print("Mean Absolute Error:",mae)


# In[40]:


from sklearn.model_selection import GridSearchCV


# In[54]:


svr=SVR()


# In[59]:


param_grids={'C':[0.01,0.5,0.1,1],
            'kernel':['linear','rbf','sigmoid','poly'],
            'gamma':['auto','scale']}


# In[61]:


gsc=GridSearchCV(svr,param_grids)


# In[62]:


gsc.fit(Scale_X_train,y_train)


# In[63]:


gsc.best_params_


# In[70]:


y_pred_svr1=gsc.predict(Scale_X_test)


# In[71]:


svr_mae=mean_absolute_error(y_pred_svr1,y_test)
print("SVR Mean Absolute Error:",svr_mae)
svr_mse=mean_squared_error(y_pred_svr1,y_test)
print("SVR Mean Squared Error:",svr_mse)


# In[78]:


plt.plot(l,y_test)
plt.plot(l,y_pred_svr1,'red')


# In[91]:


#random forest regressor


# In[94]:


from sklearn.ensemble import RandomForestRegressor


# In[95]:


rf_model=RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train,y_train)


# In[97]:


y_pred_rf=rf_model.predict(X_test)
y_pred_rf


# In[102]:


plt.plot(l,y_test)
plt.plot(l,y_pred_rf)


# In[100]:


predictions = {
    "Linear Regression": y_pred_lr,
    "Decision Tree": y_pred_dt,
    "SVR": y_pred_svr,
    "Random Forest Regressor":y_pred_rf}


# In[101]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, color='black', label="Actual Prices", alpha=0.6)

for name, y_pred in predictions.items():
    plt.scatter(y_test, y_pred, label=name, alpha=0.6)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Comparison of Regression Model Predictions")
plt.legend()
plt.show()


# In[ ]:




