#!/usr/bin/env python
# coding: utf-8

# # Libraries 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Text analysis
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist as fdist
import re

# Remove warnings
import warnings
warnings.filterwarnings('ignore')


# # IMPORTING THE 1st DATASET

# In[2]:


df1 = pd.read_csv("QVI_transaction_data.csv")
df1.head()


# #  High Level Summaries For Transaction Data

# In[3]:


print("The dimension for transaction data is: \n",df1.shape,"\n")
print("The column names for transaction data are: \n",df1.columns)


# In[4]:


print("The variable types for transaction data are:")
df1.info()


# In[5]:


import datetime
def int_date(csvdata):
    csvdate = datetime.datetime(1900, 1, 1)
    if(csvdata<60):
        delta_in_days = datetime.timedelta(days = (csvdata - 1))
    else:
        delta_in_days = datetime.timedelta(days = (csvdata - 2))
    converted_date = csvdate + delta_in_days
    return converted_date


# In[6]:


df1['DATE'] =df1['DATE'].apply(int_date)
df1['DATE'].head()


# In[7]:


df1.head()


# In[8]:


print("Checking the null values for transaction data are:")
df1.isnull().sum()


# In[9]:


print("Checking for duplicate values")
df1[df1.duplicated(['TXN_ID'])].head()


# In[10]:


# Select the first duplicated TXN_ID
df1.loc[df1['TXN_ID'] == 48887, :]


# In[11]:


print("After emoving the duplicate values: ")
#removing the duplicate values
df1.drop(df1[df1['TXN_ID'].duplicated()].index, axis=0, inplace=True)
print("The dimension for transaction data is: \n",df1.shape,"\n")


# In[12]:


df1.describe()


# In[13]:


df1['DATE'].describe()


# In[14]:


# We need to analyse the product name. 
df1['PROD_NAME'].head(10)


# In[15]:


# create a new column called packet size
df1['PACK_SIZE'] = df1['PROD_NAME'].str.extract("(\d+)")
df1['PACK_SIZE'] = pd.to_numeric(df1['PACK_SIZE'])
df1.head()


# In[16]:


# remove he special characters from the productname and also the weight
def clean_prdname(text):
    text =  re.sub('[&/]', ' ', text) 
    text =  re.sub('\d\w*', ' ', text) 
    return text

df1['PROD_NAME'] = df1['PROD_NAME'].apply(clean_prdname)
df1.head()


# In[17]:


#Separating the words
cleanname = df1['PROD_NAME']
string = "".join(cleanname)
productname = word_tokenize(string)
df1.head()


# In[18]:


#counting
fdist(productname)


# In[19]:


df1['PROD_NAME'] = df1['PROD_NAME'].apply(lambda x: x.lower())
df1.head()


# In[20]:


df1 = df1[~df1['PROD_NAME'].str.contains("salsa")]
df1['PROD_NAME'] = df1['PROD_NAME'].apply(lambda x: x.title())


# # OUTLIERS

# In[21]:


# creating a dataframe with only numberical data type
df1_num = df1.select_dtypes(['float','int'])
df1_num.head()


# In[22]:


for column in df1_num:
    plt.figure()
    df1_num.boxplot([column])


# In[23]:


sns.distplot(x = df1_num['TOT_SALES'],kde = True)


# In[24]:


#removing the outlier from tot_sales
a = df1_num[df1_num['TOT_SALES']<8.00]
a.head()


# In[25]:


sns.distplot(a.TOT_SALES,kde = True)


# In[26]:


sns.boxplot(a['TOT_SALES'])


# # IMPORTING THE 2nd DATASET

# In[27]:


df2 = pd.read_csv("QVI_purchase_behaviour.csv")
df2.head()


# # High Level Summaries For Purchase Behaviour Data

# In[28]:


print("The dimension for purchase behaviour data is: \n",df2.shape,"\n")
print("The column names for purchase behaviour data are: \n",df2.columns)


# In[29]:


print("The variable types for transaction data are:")
df2.info()


# In[30]:


print("Checking for null/missing values:")
df2.isnull().sum()





