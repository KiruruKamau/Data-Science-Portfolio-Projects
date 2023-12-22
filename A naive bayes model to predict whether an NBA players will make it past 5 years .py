
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[2]:


extracted_data = pd.read_csv('extracted_nba_players_data.csv')


# In[3]:


extracted_data.head(10)


# In[4]:


#Define the y (target) variable
y = extracted_data['target_5yrs']
#Define the X (predictor) variables
X = extracted_data.copy()
X = X.drop('target_5yrs', axis=1)


# In[5]:


y.head(10)


# In[6]:


X.head(10)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


# In[8]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[9]:


nb = GaussianNB()

nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)


# In[10]:


# Print your accuracy score.
print('Accuracy:', accuracy_score(y_test, y_pred))
# Print your precision score.
print('Precision:', precision_score(y_test, y_pred))
# Print your recall score.
print('Recall:', recall_score(y_test, y_pred))

# Print your f1 score.

print('F1 score:', f1_score(y_test, y_pred))


# In[11]:


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb.classes_)
disp.plot()


# In[ ]:




