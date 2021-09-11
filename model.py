#!/usr/bin/env python
# coding: utf-8

# In[2]:


print("Item Based Recommnedation System")


# In[3]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[5]:


#df_org=pd.read_csv('/content/gdrive/MyDrive/MyCapstone (1)/data/sample30.csv')
df_org=pd.read_csv('data/sample30.csv')
df_org.shape


# In[6]:


df = df_org[['id','reviews_username','reviews_rating']]


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.head()


# In[10]:


df.id.nunique()


# In[11]:


df.reviews_username.nunique()


# In[12]:


df_user = df.dropna()


# In[13]:


df_user.shape


# In[14]:


duplicate = df_user[df_user.duplicated(['reviews_username', 'id'])]
duplicate


# In[15]:


duplicate = df_user[df_user.duplicated()]
duplicate


# In[16]:


# Extract duplicate rows
df_user[(df_user['id']=='AV1YGDqsGV-KLJ3adc-O') & (df_user['reviews_username']=='laura')]        


# In[17]:


# dropping ALL duplicate values
df_user.drop_duplicates(keep = 'first', inplace = True)


# In[18]:


df_user.shape


# In[19]:


#Getting rows where same user provided different ratings for smae item.
duplicate = df_user[df_user.duplicated(['reviews_username', 'id'])]
duplicate


# In[20]:


#Taking mean of different ratings provided by user for same item
df2 = df_user.groupby(['id','reviews_username']).mean().reset_index()


# In[21]:


df2[(df2['id']=='AV1YGDqsGV-KLJ3adc-O') & (df2['reviews_username']=='laura')]  


# In[22]:


print(df2.shape)
df2.head()


# In[23]:


# After removing duplicates
print(df2.shape)
print(df2.reviews_username.nunique())
print(df2.id.nunique())


# In[24]:


# Pivot the train ratings' dataset into matrix format in which columns are items and the rows are user IDs.
df_pivot = df2.pivot(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(0)

df_pivot.head(10)


# In[25]:


print(df_pivot.shape)


# In[26]:


# Copy the train dataset into dummy_train
dummy_df = df2.copy()


# In[27]:


# The items not rated by user is marked as 1 for prediction. 
dummy_df['reviews_rating'] = dummy_df['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[28]:


# Convert the dummy train dataset into matrix format.
dummy_df = dummy_df.pivot(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(1)


# In[29]:


dummy_df.head()


# In[30]:


from sklearn.metrics.pairwise import pairwise_distances


# ## Using Item similarity
# 

# # Item Based Similarity

# In[31]:


df_pivot = df2.pivot(
    index='reviews_username', columns='id', values='reviews_rating'
).T

df_pivot.head()


# Normalising the movie rating for each movie for using the Adujsted Cosine

# In[32]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[33]:


df_subtracted.head()


# In[ ]:





# Finding the cosine similarity using pairwise distances approach

# In[34]:


from sklearn.metrics.pairwise import pairwise_distances


# In[35]:


# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)


# In[ ]:





# In[ ]:





# Filtering the correlation only for which the value is greater than 0. (Positively correlated)

# In[36]:


item_correlation[item_correlation<0]=0
item_correlation


# In[ ]:





# # Prediction - Item Item

# In[37]:


item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
item_predicted_ratings


# In[38]:


item_predicted_ratings.shape


# In[39]:


dummy_df.shape


# In[ ]:





# ### Filtering the rating only for the items not rated by the user for recommendation
# 

# In[40]:


item_final_rating = np.multiply(item_predicted_ratings,dummy_df)
item_final_rating.head()


# In[ ]:





# ### Finding the top 20 recommendation for the *user*
# 
# 

# In[41]:


# Take the user ID as input
user_input = "laura"
print(user_input)


# In[42]:


# Recommending the Top 20 products to the user.
d = item_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
d


# # Loading the saved model from pkl file 

# In[43]:


# Load the model from the file 
import joblib

sentiment_analysis = joblib.load('models/sentiment_analysis.pkl')
#sentiment_analysis = joblib.load('/content/gdrive/MyDrive/MyCapstone (1)/models/sentiment_analysis.pkl')


# In[44]:


from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


# In[45]:


## Replacing NAN values with blank from reviews_title field
df_org.reviews_title = df_org.reviews_title.fillna('')


# In[46]:


all_text=df_org['reviews_title']+" "+df_org['reviews_text']


# In[47]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)


# In[48]:


#char_vectorizer = TfidfVectorizer(
#    sublinear_tf=True,
#    strip_accents='unicode',
#    analyzer='char',
#    stop_words='english',
#    ngram_range=(2, 6),
 #   max_features=50000)
#char_vectorizer.fit(all_text)


# # Getting the top 5 products for user recommendation out of 20

# In[49]:


def getTopItemsFromModel(recommendItems) :
    resultDict = {}
    for id,predRating in recommendItems.items() :
        #print('index: ', id, 'value: ', predRating)
        item_text=df_org.reviews_text[df_org.id == id]
        train_word_features = word_vectorizer.transform(item_text)
        #train_char_features = char_vectorizer.transform(item_text)
        #item_features = hstack([train_char_features, train_word_features])
        item_features = train_word_features
        result = sentiment_analysis.predict(item_features)
        #print("Item name",df_org.name[df_org.id == id][0:1])
        #print(len(result))
        positivePercent = round(((len(result[result == 'Positive'])/len(result))*100),2)
        #print("%age",positivePercent,"%")
        resultDict[id] = positivePercent
    return resultDict
  


# In[50]:


def getItemsForUser(userName)  :
    print(userName)
    finalList = []
    #Handling for user not present in dataset
    if(userName not in df2['reviews_username'].tolist()) :
        return("User not found in dataset")
    items = item_final_rating.loc[userName].sort_values(ascending=False)[0:20]
    resultDict = getTopItemsFromModel(items)
    #print(resultDict)
    sorted_result = sorted(resultDict.items(), key=lambda x: x[1],reverse=True)
    print(sorted_result[0:5])
    for key,item in sorted_result[0:5] :
        finalList.append(df_org.name[df_org.id == key][0:1].values[0])
    return(finalList)


# In[57]:


#outputList = getItemsForUser('laura')
#print("--------------")
#print(outputList)


# In[ ]:





# In[ ]:





# In[ ]:




