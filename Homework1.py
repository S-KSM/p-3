

import pandas
import numpy as np
#load data
question_df = pandas.read_csv('question_dataframe.csv',sep=',')
#Cleaning Data
### Make Sure the IDs are all integer
def cleaning_ID(x):
    try:
        y = int(x)
    
    except:
        y = np.nan
    return y


question_df.ID = question_df.ID.apply(lambda d: cleaning_ID(d))
question_df = question_df.dropna()
question_df.reset_index(drop=True) 
###
#Clean Tags
def cleaning_tags(x):
    x = str(x)
    if x.startswith('lt'):
        x = "c"
    else:
        x = x
    return x
question_df.Tags = question_df.Tags.apply(lambda d: cleaning_tags(d))

question_df


# -----------------

# ## Step 3. Putting it all together

# We are now ready to tackle our original problem. Write a function to measure the similarity of the top 1000 users with the most answer posts. Compare the users based on the types of questions they answer. We will categorize the questions by looking at the first tag in each question. You may choose to implement any one of the similarity/distance measures we discussed in class. Document your findings. **(30pts)**
# 
# Note that answers are posts with `PostTypeId=2`. The ID of the question in answer posts is the `ParentId`.
# 
# You may find the [sklearn.feature_extraction module](http://scikit-learn.org/stable/modules/feature_extraction.html) helpful.

# In[20]:



### Imports # settings
import pandas
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.datasets as sk_data
import sklearn.metrics as metrics
#import seaborn as sns
import re
from string import punctuation
import operator

### Getting and Cleaning Data


#clean_answers("stackoverflow-posts-2015.xml")
### Read the answers subset into a pandas DataFrame
answers_df = pandas.read_csv('answers_dataframe.csv',sep=',')
#Cleaning Data
### Function that ensures the IDs are all integer
def cleaning_ID(x):
    try:
        y = int(x)
    
    except:
        y = np.nan
    return y

### Ensuring that all the IDs are integer.
answers_df.ParentId = answers_df.ParentId.apply(lambda d: cleaning_ID(d))
answers_df = answers_df.dropna()
answers_df.reset_index(drop=True) 

### Selecting only the ID and the Tags from the question_df (to be merged with the answers_df)
right_df = question_df[['ID','Tags']]
#right_df
# merge the two data sets based on the questions (in order to get the right tags attached to the right answer)
final_df = pandas.merge(left = answers_df,right = right_df,left_on='ParentId',right_on='ID',how='left')
final_df = final_df.dropna()
final_df.reset_index(drop=True) 
final_df = final_df[['ID_x','CreationDate','OwnerUserId','Tags']]
final_df


# In[21]:

# in this section we find the top 1000 Users
counter_dict = {}

for i in range(final_df.shape[0]):
    OwnerUserId = final_df.OwnerUserId.iat[i]
    if OwnerUserId in counter_dict:
        counter_dict[OwnerUserId] += 1
    else:
        counter_dict[OwnerUserId] = 1


        
t = sorted(counter_dict.items(), key=operator.itemgetter(1),reverse=True)[0:1000]
top_1000_Users = [t[n][0] for n in range(1000)]
# subset the top 1000 Users
result = final_df[final_df.OwnerUserId.isin(top_1000_Users)]
#print("Comparing two datasets:","Dimensions of top 1000 Useras:",result.shape,"initial dataset:",final_df.shape)

### List of tags
list_tags = []
for i in range(result.shape[0]):
    Tags = result.Tags.iat[i]
    if Tags not in list_tags: 
        list_tags.append(Tags)
###
# Create dictionary of the desired values 
dict_group_result = dict(result.groupby(['OwnerUserId','Tags']).ID_x.count())
final_dict = {}
for user in top_1000_Users:
    for tags in list_tags:

        if user not in final_dict:
            final_dict[user] = {}
        if tags not in final_dict[user]:
            try: 
                final_dict[user][tags] = dict_group_result[user,tags]
            except KeyError:
                final_dict[user][tags] = 0

#Create the dataframe
        
df = pandas.DataFrame(final_dict).reset_index().rename(columns={"index": "Tags"})      
df = pandas.melt(df, "Tags", var_name="ID")
df = df.pivot_table('value', ['ID'], 'Tags')
# Create Distance Matrix
df_1 = df.as_matrix()
euclidean_dists = metrics.euclidean_distances(df_1)
print (euclidean_dists.shape)


# Let's plot a subset of the distance matrix. Order the pairwise distance in your distance matrix (excluding the entries along the diagonal) in increasing order and pick user pairs until you have 100 unique users. See [Lecture 3](https://github.com/datascience16/lectures/blob/master/Lecture3/Distance-Functions.ipynb) for examples. **(10 pts)**

# In[22]:

idx = np.array([n**2 - 1 for n in range(1,1001)])
euclidean_dists1 = np.delete(euclidean_dists,idx).reshape(1000,999)
sorted_dist = np.sort(euclidean_dists1)
#sns.heatmap(sorted_dist[0:100,0:100], xticklabels=False, yticklabels=False, linewidths=0, square=False, cbar=True)


# Next, let's create some time series from the data. Look at the top 100 users with the most question posts. For each user, your time series will be the `CreationDate` of the questions posted by that user. You may want to make multiple time series for each user based on the first tag of the questions. Compare the time series using one of the methods discussed in class. Document your findings. **(30 pts)**
# 
# You may find the [pandas.DataFrame.resample module](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html) helpful.

# In[23]:

counter_dict_q = {}

for i in range(question_df.shape[0]):
    OwnerUserId = question_df.OwnerUserId.iat[i]
    if OwnerUserId in counter_dict_q:
        counter_dict_q[OwnerUserId] += 1
    else:
        counter_dict_q[OwnerUserId] = 1

       
t = sorted(counter_dict.items(), key=operator.itemgetter(1),reverse=True)[0:1000]
top_100_Users_questions = [t[n][0] for n in range(1000)]
top_100_Users_questions

question_df_final = question_df[question_df['OwnerUserId'].isin(top_100_Users_questions)]


# In[134]:

import datetime
df = question_df_final.reset_index()
df['count'] = df.groupby(['CreationDate','OwnerUserId']).ID.transform(lambda x: x.count())
df['sum_Tags'] = df.groupby(['OwnerUserId','Tags'])['count'].transform(lambda x: sum(x))
df['sum_Users'] = df.groupby(['OwnerUserId'])['sum_Tags'].transform(lambda x: sum(x))

### the apply method can be used as well.
for row in range(df.shape[0]):
    df.CreationDate.iloc[row] = datetime.datetime.strptime(df.CreationDate.iat[row][:8], "%Y%m%d")

dfs=df.set_index('CreationDate')
dfs.head()

dfs_users = [dfs[dfs.OwnerUserId == n] for n in top_100_Users_questions]



# In[172]:

### Sample user timeseries
ts3 = dfs_users[3]['count']
ts3 = ts3.cumsum()

plt.figure();
ts3.plot()


# Plot the 2 most similar and the 2 most different time series. **(10 pts)**

# In[174]:

### Sample user timeseries
fig, axs = plt.subplots(1,2)

ts3 = dfs_users[3]['count']
ts3 = ts3.cumsum()
ts4 =dfs_users[100]['count']
ts4= ts4.cumsum()
ts5 =dfs_users[645]['count']
ts5= ts5.cumsum()

ts3.plot(ax=axs[0])
ts5.plot(ax=axs[0])
ts4.plot(ax=axs[1])
ts5.plot(ax=axs[1])
plt.figure()
plt.plot()

# In[ ]:




# In[ ]:



