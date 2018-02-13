
# coding: utf-8

# In[16]:

#import all required packages
import sys
import time
import csv
import unicodecsv as csv
from TwitterAPI import TwitterAPI
import pickle
import numpy as np
from pathlib import Path
import requests


# In[17]:

#twitter API keys for collecting data
consumer_key = 'tb9zMiyUJ9UeX6blluI4Wlws8'
consumer_secret = 'XlGNdXHj6Rd2LJ2t9LodnCLydvO2kZ4KLoq95VnQRXfuKXDgZn'
access_token = '3231404389-56xTceJ255ar5Sj2sQtWuDj8z0grnNedlqOt5AC'
access_token_secret = 'qwPAoWezVnfgeQCCaJy03rOQUJG2Z9ZACzVPIjcqu7IKD'


# In[38]:

#connecting twitter using  API keys for collecting data
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


# In[62]:

#roburst request block collecting data from Twitter rest API
def get_tweets(twitter,hashtags,tweets,since_id):
    while True:
        try:
            #parameters for request method
            request = twitter.request('search/tweets',{'q':hashtags,'count':100,'lang':'en','max_id':since_id})
            for response in request:
                tweets.append(response)
            print (len(tweets))
            return tweets
        except:
            print("Unexpected error:", sys.exc_info()[0])
            break     


# In[54]:

#get friends based on the screen name of a user
def get_friends(screen_name): 
    return sorted([friend_id for friend_id in twitter.request('friends/ids',{'screen_name' : screen_name, 'count' :5000})])


# In[55]:

#select the fields required for future processing the Json response of the Twitter

def select_required(tweet,M,F):
    data = []
    for each_tweet in tweet:
        name = each_tweet['user']['name']
        if name:
            tweet_data = {}
            tweet_data['screen_name']=each_tweet['user']['screen_name']
            tweet_data['userid']= each_tweet['user']['id']
            tweet_data['description']=each_tweet['user']['description']
            tweet_data['tweet']=each_tweet['text']
            tweet_data['username']=each_tweet['user']['name']
            tweet_data['since_id']=each_tweet['id']
            tweet_data['location']= each_tweet['user']['location']
            tweet_data['gender']=gender_by_name(name, M, F)
            if 'retweeted_status' in each_tweet.keys():
                tweet_data['fcount']=each_tweet['retweeted_status']['favorite_count']
            else:
                tweet_data['fcount']= 1
            data.append(tweet_data)
    return data


# In[56]:

#get name list from the census data and start identifying use name to be a male or female
def names():
    males_url = 'http://www2.census.gov/topics/genealogy/' +                 '1990surnames/dist.male.first'
    females_url = 'http://www2.census.gov/topics/genealogy/' +                   '1990surnames/dist.female.first'
    males = requests.get(males_url).text.split('\n')
    females = requests.get(females_url).text.split('\n')
    male_names = set([m.split()[0].lower() for m in males if m])
    female_names = set([f.split()[0].lower() for f in females if f])


    # Keep names that are more frequent in one gender than the other.
    males_pct = dict([(m.split()[0].lower(), float(m.split()[1]))
                      for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1]))
                        for f in females if f])

    male_names = set([m for m in male_names if m not in female_names or
                  males_pct[m] > females_pct[m]])
    female_names = set([f for f in female_names if f not in male_names or
                  females_pct[f] > males_pct[f]])
    return male_names,female_names


# In[57]:

#gender classification
def gender_by_name(name, male_names, female_names):
    if name:
        first = name.split()[0].lower()
        if first in male_names:
            return ('male')
        elif first in female_names:
            return ('female')
        else:
            return ('male')


# In[58]:

#open the pickle and get the content of the file for appending new data
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


# In[59]:

#finally write all the collected data back in pickle file
def write_file(tweets,filename):
    fname=filename+'.pkl'
    print("file length after writing->",len(tweets))
    pickle.dump(tweets, open(fname, 'wb'))
        


# In[64]:

def main():
    #establish twitter object
    twitter = get_twitter()
    print('Established Twitter connection.')
    print(' started collecting tweets From Twitter Based on location')
    #team names that are used for analysis
    #hashtags=['#mufc','#basvafc','#fcbvafc','#arsenal','#championsleague','#ucl','#FCBarcelona','#RealMadrid',
    #         '#CristianoRonaldo','#cr7','#LeoMessi','#messi','#mancity','#chelsea','#mcfcvcfc','#Celtic']
    hashtags=['#DubNation','#WeTheNorth','#GrindCity','#DetroitBasketball','#thunderup','#LakeShow','#DefendTheLand']
   
    s_id=0
    #get list of male and females names
    M,F=names()
    for tags in hashtags:
        tweets = []
        tweets_from_file =[]
        since_id=[]
        #opening file
        tweets_from_file =open_file(tweets_from_file,tags)
        # get the minimum Tweet ID to make the collection to be unique
        if tweets_from_file:    
            for i in tweets_from_file:
                since_id.append(i['since_id'])
            s_id=min(since_id)
            print (s_id)
        print("stating")
        print ("length before search ",len(tweets_from_file))
        #get the tweets
        tweets = get_tweets(twitter,tags,tweets,s_id)
        #filtering the Json response
        tweets= select_required(tweets,M,F)
        for i in tweets:
            tweets_from_file.append(i)
        #write the contents to file    
        write_file(tweets_from_file,tags)
    print('tweets saved to a file called tweets \n')

if __name__ == '__main__':
    main()


# In[ ]:



