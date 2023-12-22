
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


os.chdir(r'C:\Users\nyanc\Data-Science-Portfolio-Projects')


# In[4]:


score = pd.read_csv('score.csv')


# In[5]:


score.head()


# In[6]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[7]:


score.isna().sum()


# In[8]:


score.shape


# In[10]:


score['Hours'].describe()


# In[11]:


fig = sns.histplot(score['Scores'])
fig.set_title('Distribution of scores')


# In[12]:


ols_formula = 'Scores ~ Hours'

OLS = ols(formula=ols_formula, data=score)

model = OLS.fit()


# In[14]:


model_results = model.summary()
model_results


# In[19]:


#Linearity
fig = sns.scatterplot(x=score['Hours'], y=score['Scores'])


# In[20]:


fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)

# Set the x-axis label.
fig.set_xlabel("Fitted Values")

# Set the y-axis label.
fig.set_ylabel("Residuals")

# Set the title.
fig.set_title("Fitted Values v. Residuals")

# Add a line at y = 0 to visualize the variance of residuals above and below 0.

fig.axhline(0)

# Show the plot.
plt.show()


# In[ ]:


# Since the linearity and Homoscedasticity are assumptions are not met this implies that the model does not truly explain the relationships between this variables.

