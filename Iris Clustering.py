#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


# In[19]:


df=sns.load_dataset('iris')


# In[20]:


df.head()


# In[21]:


df.info()


# In[22]:


df.isna().sum()


# In[23]:


df['species'].unique()


# In[24]:


df=pd.get_dummies(df,dtype=int,drop_first=True)


# In[25]:


df.head()


# In[26]:


from sklearn.preprocessing import MinMaxScaler


# In[27]:


Scaler=MinMaxScaler()


# In[28]:


Scaled_df=Scaler.fit_transform(df)


# In[29]:


Scaled_df


# In[30]:


pd.DataFrame(Scaled_df)


# In[31]:


hr_df=pd.DataFrame(Scaled_df,columns=df.columns)


# In[32]:


hr_df.head()


# In[33]:


sns.heatmap(hr_df)


# In[34]:


sns.clustermap(hr_df)


# In[35]:


from sklearn.cluster import AgglomerativeClustering


# In[36]:


ag_model=AgglomerativeClustering(n_clusters=5)
ag_model


# In[37]:


clusters=ag_model.fit_predict(hr_df)


# In[38]:


df.head()


# In[39]:


hr_df.head()


# In[40]:


len(hr_df.columns)


# In[41]:


from scipy.cluster.hierarchy import dendrogram


# In[42]:


from scipy.cluster.hierarchy import linkage


# In[46]:


link_matric=linkage(ag_model.children_)


# In[47]:


len(link_matric)


# In[48]:


dendrogram(link_matric,truncate_mode='level')
plt.show()


# DBSCAN

# In[50]:


from sklearn.cluster import DBSCAN


# In[51]:


dbs_model=DBSCAN()


# In[52]:


def display_category (dbs_model,data1):
    labels=dbs_model.fit_predict(data1)
    sns.scatterplot(data=data1,x='sepal_length',y='sepal_width',hue=labels)


# In[53]:


dbs_model.fit_predict(df)


# In[54]:


display_category(dbs_model,df)


# In[55]:


np.sum(dbs_model.labels_==1)


# In[56]:


np.linspace(1,5,10)


# In[58]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[ ]:




