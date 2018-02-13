
# coding: utf-8

# In[10]:

import sys
import time
import csv
import unicodecsv as csv
from TwitterAPI import TwitterAPI
import pickle
import numpy as np
from pathlib import Path


# In[11]:

consumer_key = 'tb9zMiyUJ9UeX6blluI4Wlws8'
consumer_secret = 'XlGNdXHj6Rd2LJ2t9LodnCLydvO2kZ4KLoq95VnQRXfuKXDgZn'
access_token = '3231404389-56xTceJ255ar5Sj2sQtWuDj8z0grnNedlqOt5AC'
access_token_secret = 'qwPAoWezVnfgeQCCaJy03rOQUJG2Z9ZACzVPIjcqu7IKD'


# In[12]:

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.     
    """
    consumer_key = 'tb9zMiyUJ9UeX6blluI4Wlws8'
    consumer_secret = 'XlGNdXHj6Rd2LJ2t9LodnCLydvO2kZ4KLoq95VnQRXfuKXDgZn'
    access_token = '3231404389-56xTceJ255ar5Sj2sQtWuDj8z0grnNedlqOt5AC'
    access_token_secret = 'qwPAoWezVnfgeQCCaJy03rOQUJG2Z9ZACzVPIjcqu7IKD'
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


# In[13]:

def get_tweets(twitter,hashtags,tweets,since_id):
    
    while True:
        try:

            request = twitter.request('search/tweets',{'q':hashtags,'count':1000,'lang':'en','max_id':since_id})
            for response in request:
                tweets.append(response)
            print (len(tweets))
            return tweets
        except:
            print("Unexpected error:", sys.exc_info()[0])
            break     


# In[14]:

def get_friends(screen_name): 
    return sorted([friend_id for friend_id in twitter.request('friends/ids',{'screen_name' : screen_name, 'count' :5000})])


# In[15]:

def select_required(tweet):
    data = []
    
    for each_tweet in tweet:
        tweet_data = {}
        tweet_data['screen_name']=each_tweet['user']['screen_name']
        tweet_data['userid']= each_tweet['user']['id']
        tweet_data['description']=each_tweet['user']['description']
        tweet_data['tweet']=each_tweet['text']
        tweet_data['username']=each_tweet['user']['name']
        tweet_data['since_id']=each_tweet['id']
        data.append(tweet_data)
    return data


# In[16]:

def open_file(tweets,filename):
    fname= filename+'.pkl'
    p = Path(fname)
    if p.is_file():
        with open(fname, "rb") as file:
            try:
                tweets = pickle.load(file)
            except EOFError:
                return tweets
    return tweets


# In[17]:


def write_file(tweets,filename):
    fname=filename+'.pkl'
    print("file length after writing->",len(tweets))
    pickle.dump(tweets, open(fname, 'wb'))
        


# In[18]:

def main():
    twitter = get_twitter()
    print('Established Twitter connection.')
    print(' started collecting tweets From Twitter Based on location')
    hashtags=['#DubNation','#WeTheNorth','#GrindCity','#DetroitBasketball','#thunderup','#LakeShow','#DefendTheLand']
    s_id=0
    for tags in hashtags:
        tweets = []
        tweets_from_file =[]
        since_id=[]
        tweets_from_file =open_file(tweets_from_file,tags)
        if tweets_from_file:    
            for i in tweets_from_file:
                since_id.append(i['since_id'])
            s_id=min(since_id)
            print (s_id)
        print("stating")
        print ("length before search ",len(tweets_from_file))
        tweets = get_tweets(twitter,tags,tweets,s_id)
        tweets= select_required(tweets)
        for i in tweets:
            tweets_from_file.append(i)
        write_file(tweets_from_file,tags)
    print('tweets saved to a file called tweets \n')

if __name__ == '__main__':
    main()


# In[ ]:



