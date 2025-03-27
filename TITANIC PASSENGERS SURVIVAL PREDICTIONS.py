#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[66]:


td=pd.read_csv(r"C:\Users\Uer\Downloads\Titanic-Dataset.csv")


# In[67]:


td.info()


# In[68]:


td.isna().sum()


# In[69]:


td['Age']=td['Age'].fillna(td['Age'].mean())
td['Embarked']=td['Embarked'].fillna(td['Embarked'].mode()[0])


# In[70]:


td.isna().sum()


# In[72]:


td=pd.get_dummies(td,columns=['Sex','Embarked'],drop_first=True)


# In[73]:


td.info()


# In[75]:


td['Sex_male']=td['Sex_male'].astype(int)
td['Embarked_Q']=td['Embarked_Q'].astype(int)
td['Embarked_S']=td['Embarked_S'].astype(int)


# In[76]:


td=td.drop(columns=['Name','Ticket','Cabin'])


# In[77]:


td.head()


# In[78]:


X=td.drop(columns=['Survived'])
y=td['Survived']


# In[79]:


X


# In[80]:


y


# In[81]:


from sklearn.model_selection import train_test_split


# In[82]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=.80,random_state=42)


# In[83]:


X_train.shape


# In[84]:


X_test.shape


# In[85]:


X_train.head()


# In[86]:


X_test.head()


# In[38]:


#logistic regression


# In[87]:


from sklearn.linear_model import LogisticRegression


# In[94]:


lg_model=LogisticRegression(max_iter=1000)


# In[95]:


lg_model.fit(X_train,y_train)


# In[98]:


y_pred_lg=lg_model.predict(X_test)


# In[97]:


from sklearn.metrics import confusion_matrix,classification_report


# In[117]:


cm_lg=confusion_matrix(y_test,y_pred_lg)
cm_lg


# In[101]:


sns.heatmap(confusion_matrix(y_test,y_pred_lg),annot=True)


# In[102]:


print(classification_report(y_test,y_pred_lg))


# In[104]:


lg_model.score(X_train,y_train)


# In[106]:


#random forest 


# In[107]:


from sklearn.ensemble import RandomForestClassifier


# In[109]:


rf_model=RandomForestClassifier(n_estimators=50,max_features='sqrt',random_state=100)
rf_model


# In[110]:


rf_model.fit(X_train,y_train)


# In[111]:


y_pred_rf=rf_model.predict(X_test)


# In[115]:


cm_rf=confusion_matrix(y_test,y_pred_rf)
cm_rf


# In[118]:


sns.heatmap(confusion_matrix(y_test,y_pred_rf),annot=True)


# In[119]:


rf_model.score(X_train,y_train)


# In[121]:


print(classification_report(y_test,y_pred_rf))


# In[122]:


from sklearn.model_selection import GridSearchCV


# In[123]:


n_estimators=[64,100,120,200]
bootstrap=[True,False]
oob_score=[True,False]


# In[124]:


rf=RandomForestClassifier()


# In[125]:


max_features=['sqrt','auto']
param_grid={'n_estimators':n_estimators,'max_features':max_features,'bootstrap':bootstrap,'oob_score':oob_score}


# In[128]:


gcv=GridSearchCV(rf,param_grid)
gcv


# In[129]:


gcv.fit(X_train,y_train)


# In[130]:


gcv.best_params_


# In[131]:


y_pred_rf1=gcv.predict(X_test)


# In[132]:


print(classification_report(y_test,y_pred_rf1))


# In[133]:


confusion_matrix(y_test,y_pred_rf1)


# In[134]:


#svm


# In[135]:


from sklearn.svm import SVC


# In[137]:


svc_model=SVC(kernel='linear',C=100)
svc_model


# In[138]:


svc_model.fit(X_train,y_train)


# In[144]:


y_pred_svc=svc_model.predict(X_test)


# In[145]:


confusion_matrix(y_test,y_pred_svc)


# In[146]:


print(classification_report(y_test,y_pre_svc))


# In[ ]:




