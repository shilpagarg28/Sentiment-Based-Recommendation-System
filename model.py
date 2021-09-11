#!/usr/bin/env python
# coding: utf-8

# In[58]:


print("Item Based Recommnedation System")


# In[59]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[60]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[61]:


#df_org=pd.read_csv('/content/gdrive/MyDrive/MyCapstone (1)/data/sample30.csv')
df_org=pd.read_csv('data/sample30.csv')
df_org.shape


# In[62]:


df = df_org[['id','reviews_username','reviews_rating']]


# In[63]:


df.shape


# In[64]:


df.info()


# In[65]:


df.head()


# In[66]:


df.id.nunique()


# In[67]:


df.reviews_username.nunique()


# In[68]:


df_user = df.dropna()


# In[69]:


df_user.shape


# In[70]:


duplicate = df_user[df_user.duplicated(['reviews_username', 'id'])]
duplicate


# In[71]:


duplicate = df_user[df_user.duplicated()]
duplicate


# In[72]:


# Extract duplicate rows
df_user[(df_user['id']=='AV1YGDqsGV-KLJ3adc-O') & (df_user['reviews_username']=='laura')]        


# In[73]:


# dropping ALL duplicate values
df_user.drop_duplicates(keep = 'first', inplace = True)


# In[74]:


df_user.shape


# In[75]:


#Getting rows where same user provided different ratings for smae item.
duplicate = df_user[df_user.duplicated(['reviews_username', 'id'])]
duplicate


# In[76]:


#Taking mean of different ratings provided by user for same item
df2 = df_user.groupby(['id','reviews_username']).mean().reset_index()


# In[77]:


df2[(df2['id']=='AV1YGDqsGV-KLJ3adc-O') & (df2['reviews_username']=='laura')]  


# In[78]:


print(df2.shape)
df2.head()


# In[79]:


# After removing duplicates
print(df2.shape)
print(df2.reviews_username.nunique())
print(df2.id.nunique())


# In[80]:


# Pivot the train ratings' dataset into matrix format in which columns are items and the rows are user IDs.
df_pivot = df2.pivot(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(0)

df_pivot.head(10)


# In[81]:


print(df_pivot.shape)


# In[82]:


# Copy the train dataset into dummy_train
dummy_df = df2.copy()


# In[83]:


# The items not rated by user is marked as 1 for prediction. 
dummy_df['reviews_rating'] = dummy_df['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[84]:


# Convert the dummy train dataset into matrix format.
dummy_df = dummy_df.pivot(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(1)


# In[85]:


dummy_df.head()


# In[86]:


from sklearn.metrics.pairwise import pairwise_distances


# ## Using Item similarity
# 

# # Item Based Similarity

# In[87]:


df_pivot = df2.pivot(
    index='reviews_username', columns='id', values='reviews_rating'
).T

df_pivot.head()


# Normalising the movie rating for each movie for using the Adujsted Cosine

# In[88]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[89]:


df_subtracted.head()


# In[ ]:





# Finding the cosine similarity using pairwise distances approach

# In[90]:


from sklearn.metrics.pairwise import pairwise_distances


# In[91]:


# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)


# In[ ]:





# In[ ]:





# Filtering the correlation only for which the value is greater than 0. (Positively correlated)

# In[92]:


item_correlation[item_correlation<0]=0
item_correlation


# In[ ]:





# # Prediction - Item Item

# In[93]:


item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
item_predicted_ratings


# In[94]:


item_predicted_ratings.shape


# In[95]:


dummy_df.shape


# In[ ]:





# ### Filtering the rating only for the items not rated by the user for recommendation
# 

# In[96]:


item_final_rating = np.multiply(item_predicted_ratings,dummy_df)
item_final_rating.head()


# In[ ]:





# ### Finding the top 20 recommendation for the *user*
# 
# 

# In[97]:


# Take the user ID as input
user_input = "laura"
print(user_input)


# In[98]:


# Recommending the Top 20 products to the user.
d = item_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
d


# # Loading the saved model from pkl file 

# In[99]:


# Load the model from the file 
import joblib

sentiment_analysis = joblib.load('models/sentiment_analysis.pkl')
#sentiment_analysis = joblib.load('/content/gdrive/MyDrive/MyCapstone (1)/models/sentiment_analysis.pkl')


# In[100]:


from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


# In[101]:


## Replacing NAN values with blank from reviews_title field
df_org.reviews_title = df_org.reviews_title.fillna('')


# In[102]:


all_text=df_org['reviews_title']+" "+df_org['reviews_text']


# In[103]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)


# In[104]:


#char_vectorizer = TfidfVectorizer(
#    sublinear_tf=True,
#    strip_accents='unicode',
#    analyzer='char',
#    stop_words='english',
#    ngram_range=(2, 6),
#   max_features=50000)
#char_vectorizer.fit(all_text)


# # Getting the top 5 products for user recommendation out of 20

# In[105]:


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
  


# In[106]:


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


# In[107]:


#outputList = getItemsForUser('laura')
#print("--------------")
#print(outputList)


# In[ ]:





# In[ ]:





# In[ ]:




