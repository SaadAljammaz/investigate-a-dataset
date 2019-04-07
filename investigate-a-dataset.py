
# coding: utf-8

# # Project : Investigate a Dataset
# # Dataset Used : TMDb movie data
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue.

# ### Setting up the import statements for all of the packages that I need

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="ticks", color_codes=True)


# <a id='Questions'></a>
# ### Describing the questions
# <ul>
#     <li><a href="#q1">1- What type of movies that made a huge profits, popularity and high votes?</a></li>
#     <li><a href="#q2">2- What kinds of properties are associated with movies that have high revenues?</a></li>
#     <li><a href="#q3">3- what is the best month to release the movie?</a></li>
#     <li><a href="#q4">4- Who is the most profitable director?</a></li>
# </ul>

# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# 
# ### General Properties

# ### Loading the data and printing a few lines

# In[2]:


df = pd.read_csv('tmdb-movies.csv')
df.head(1)


# In[3]:


df.info()


# In[4]:


df.nunique()


# ### Data Cleaning

# ### Removing the columns that we don't need

# In[5]:


df.drop(columns='id',inplace=True)
df.drop(columns='imdb_id',inplace=True)
#df.drop(columns='original_title',inplace=True)
df.drop(columns='homepage',inplace=True)
df.drop(columns='overview',inplace=True)
df.drop(columns='production_companies',inplace=True)
df.drop(columns='tagline',inplace=True)
df.drop(columns='keywords',inplace=True)


# In[6]:


df.info()


# ### Removing duplicated rows

# In[7]:


df.duplicated().sum()


# In[8]:


df.drop_duplicates(inplace=True)


# In[9]:


df.duplicated().sum()


# ### Handling less than 100 values in budget and revenue

# In[10]:


df.describe()


# In[11]:


# I will fill values less than 100 with the mean value for the column
df['budget'].values[df['budget'] < 100] = df.budget.mean()


# In[12]:


# I will fill values less than 100 with the mean value for the column
df['revenue'].values[df['revenue'] < 100] = df.budget.mean()


# In[13]:


df.describe()


# <a id='eda'></a>
# ## Exploratory Data Analysis

# First of all I will show the histogram for each column to see the shape, distribution and understand the skewness of the data.

# In[14]:


df.hist(figsize=(15,8));


# <a id='q1'></a>
# ## Question 1 : What type of movies that made a huge profits, popularity and high votes?
# <a href="#Questions">Get back to the Questions</a>

# ### Loading the only columns that we need

# In[15]:


columns = ['release_year','budget', 'revenue','genres','vote_average','popularity']
q1 = df[columns]
q1.head()


# ### Adding profits column by subtracting budget from revenue

# In[16]:


revenue = np.subtract(q1['revenue'],q1['budget'])
q1['Profits'] = revenue
q1.head()


# ### Removing the columns that we don't need

# In[17]:


q1 = q1.drop(columns=['revenue','budget'])
q1.head()


# In[18]:


q1.groupby('genres').sum().sort_values(by=['Profits'], ascending=False).head()


# In[19]:


ax = q1.groupby('genres').sum().sort_values(by=['Profits'], ascending=False).head().plot(y='Profits',title='Profits for each Genre',figsize=(12,8));
ax.set_xlabel("genres");
ax.set_ylabel("Profits");


# In[20]:


ax = q1.groupby('genres').sum().sort_values(by=['vote_average'], ascending=False).head().plot(y='vote_average',title='vote average for each Genre',figsize=(12,8));
ax.set_xlabel("genres");
ax.set_ylabel("vote average");


# In[21]:


ax = q1.groupby('genres').sum().sort_values(by=['popularity'], ascending=False).head().plot(y='popularity',title='popularity for each Genre',figsize=(12,8));
ax.set_xlabel("genres");
ax.set_ylabel("popularity");


# ## We can see that the two genres Comedy and Drama are the best type of movies
# <hr style="height:1px;border:none;color:#333;background-color:#333;" />

# <a id='q2'></a>
# ### Question 2 : What kinds of properties are associated with movies that have high revenues?
# <a href="#Questions">Get back to the Questions</a>

# ### Loading the only columns that we need

# In[22]:


# In order test all properties that are associated with movies, we will consider
# only the properties that we can handle during the process of producing the 
# movie such as budget, genres and runtime. properties like popularity and vote_average
# are came as a result for the movie and doesn't came until releasing the movie.
columns = ['budget','revenue', 'runtime','genres']
q2 = df[columns]
q2.head()


# In[23]:


q2.plot(x='budget',y='revenue',kind='scatter',title='The scatter between budget and revenue',figsize=(12,8));


# In[24]:


q2.plot(x='runtime',y='revenue',kind='scatter',title='The scatter between run time and revenue',figsize=(12,8));


# In[25]:


ax = q2.groupby('genres').sum().sort_values(by=['revenue'], ascending=False).head().plot(y='revenue',title='The revenue for each genres',figsize=(12,8));
ax.set_xlabel("genres");
ax.set_ylabel("revenue");


# ### considering the genre of the movie, we can say that comedy is the best way to achieve high revenue
# <hr style="height:1px;border:none;color:#333;background-color:#333;" />

# <a id='q3'></a>
# ### Research Question 3 : what is the best month to release the movie?
# <a href="#Questions">Get back to the Questions</a>

# ### Loading the only columns that we need

# In[26]:


columns = ['popularity','release_date']
q3 = df[columns]
q3.head()


# ### Extracting the month

# In[27]:


q3['date'] = pd.to_datetime(q3['release_date'])
q3['month'] = q3['date'].dt.month


# In[28]:


q3.head()


# In[29]:


q3.plot(x='month',y='popularity',kind='scatter',title='The scatter between run time and revenue',figsize=(12,8));


# <a id='q4'></a>
# ### Research Question 4 : Who is the most profitable director?

# ### Loading the only columns that we need

# In[30]:


columns = ['director','budget','revenue']
q4 = df[columns]
q4.head()


# ### Adding profits column by subtracting budget from revenue

# In[31]:


revenue = np.subtract(q4['revenue'],q4['budget'])
q4['Profits'] = revenue
q4.head()


# ### Removing the columns that we don't nee

# In[32]:


q4.drop(columns=['revenue','budget'],inplace=True)
q4.head()


# In[33]:


q4.groupby(['director']).sum().sort_values(by=['Profits'],ascending=False).head()


# In[34]:


ax = q4.groupby(['director']).sum().sort_values(by=['Profits'],ascending=False).head().plot(title='the Profits for each director',figsize=(12,8));
ax.set_xlabel("director");
ax.set_ylabel("Profits");


# ### From the visualization we can see that Steven Spielberg is the director who made the highest profits from the movies
# <hr style="height:1px;border:none;color:#333;background-color:#333;" />

# ## Limitations
# During this project, I faced many limitation that hindered me.
# First of all, the duplication of rows, I could handle it by deleting the duplicated rows.
# Second, the less than 100 values in budget and revenue columns, I assumed that, 100$ as a budget for movie doesn't make sense so I filled them with the mean of them.

# <a id='conclusions'></a>
# ## Conclusions
# We can see that the two genres Comedy and Drama are the best type of movies. From the visualization we can see that Steven Spielberg is the director who made the highest profits from the movies. Considering the genre of the movie, we can say that Comedy is the best way to achieve high revenue

# ## References

# <center>N/A</center>
