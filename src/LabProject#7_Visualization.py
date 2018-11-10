
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


# This creates a pandas dataframe and assigns it to the titanic variable.
dfTitanic = pd.read_csv("C:\\Users\\Sony\\Desktop\\train.csv")
# Print the first 5 rows of the dataframe.
dfTitanic.head(10)


# In[4]:


# This creates a pandas dataframe and assigns it to the titanic variable.
dfTitanic_test = pd.read_csv("C:\\Users\\Sony\\Desktop\\test.csv")
# Print the first 5 rows of the dataframe.
dfTitanic_test.head().T


# In[7]:


#shape command will give number of rows/samples/examples and number of columns/features/predictors in dataset
#(rows,columns)
dfTitanic.shape


# In[8]:


#Describe gives statistical information about numerical columns in the dataset
dfTitanic.describe()
#you can check from count if there are missing vales in columns, here age has got missing values


# In[9]:


#To see if there are any more columns with missing values 
null_columns=dfTitanic.columns[dfTitanic.isnull().any()]
dfTitanic.isnull().sum()


# In[10]:


dfTitanic_test.isnull().sum()


# In[11]:


#Age, Fare and cabin has missing values. we will see how to fill missing values next.


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)


labels = []
values = []
for col in null_columns:
    labels.append(col)
    values.append(dfTitanic[col].isnull().sum())
ind = np.arange(len(labels))
width=0.6
fig, ax = plt.subplots(figsize=(6,5))
rects = ax.barh(ind, np.array(values), color='red')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_ylabel("Column Names")
ax.set_title("Variables with missing values");


# In[18]:


dfTitanic.hist(bins=10,figsize=(9,7),grid=False);


# In[19]:


#we can see that Age and Fare are measured on very different scaling. So we need to do feature scaling before predictions.


# In[21]:


g = sns.FacetGrid(dfTitanic, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="green");


# In[23]:


g = sns.FacetGrid(dfTitanic, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"orange", 0:"blue"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();


# In[24]:


g = sns.FacetGrid(dfTitanic, hue="Survived", col="Sex", margin_titles=True,
                palette="Set1",hue_kws=dict(marker=["^", "v"]))
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Survival by Gender , Age and Fare');


# In[25]:


dfTitanic.Embarked.value_counts().plot(kind='bar', alpha=0.55)
plt.title("Passengers per boarding location");


# In[26]:


sns.set(font_scale=1)
g = sns.factorplot(x="Sex", y="Survived", col="Pclass",
                    data=dfTitanic, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('How many Men and Women Survived by Passenger Class');


# In[33]:


ax = sns.boxplot(x="Survived", y="Age", 
                data=dfTitanic)
ax = sns.stripplot(x="Survived", y="Age",
                   data=dfTitanic, jitter=True,
                   edgecolor="gray")


# In[34]:


dfTitanic.Age[dfTitanic.Pclass == 1].plot(kind='kde')    
dfTitanic.Age[dfTitanic.Pclass == 2].plot(kind='kde')
dfTitanic.Age[dfTitanic.Pclass == 3].plot(kind='kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') ;


# In[35]:


corr=dfTitanic.corr()#["Survived"]
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


# In[37]:


dfTitanic["Deck"]=dfTitanic.Cabin.str[0]
dfTitanic_test["Deck"]=dfTitanic.Cabin.str[0]
dfTitanic["Deck"].unique() # 0 is for null values


# In[38]:


g = sns.factorplot("Survived", col="Deck", col_wrap=4,
                    data=dfTitanic[dfTitanic.Deck.notnull()],
                    kind="count", size=2.5, aspect=.8);

