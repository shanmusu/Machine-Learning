
# coding: utf-8

# In[46]:

import json
import csv
import numpy as np
from pathlib import Path
import sys
import time
import pickle


# In[47]:

def open_file(tweets,fname):
    fname=fname+'.pkl'
    p = Path(fname)
    if p.is_file():
        with open(fname, "rb") as file:
            try:
                tweets = pickle.load(file)
            except EOFError:
                return tweets
    return tweets


# In[48]:

def get_tweets(file,tweets_list):
    for i in file:
        t=[]
        tweets_list.append(i['tweet'])
    return tweets_list
    


# In[49]:

def read_tarining_data(filename):
    tweets = []
    labels1 = []
    with open(filename, 'r',encoding='utf-8') as x:
        filereader = csv.reader(x)
        for  row in filereader:
            labels1.append(row[0])
            tweets.append(row[1])
    return tweets,np.array(labels1)


# In[50]:

def open_clusture(fname):
    p = Path(fname)
    if p.is_file():
        with open(fname, "rb") as file:
            try:
                tweets = pickle.load(file)
            except EOFError:
                return tweets
    return tweets


# In[56]:

def main():
    hashtags=['#WeTheNorth','#GrindCity','#DubNation','#DefendTheLand','#DetroitBasketball','#thunderup','#LakeShow']
    alltweet=[]
    alllable=[]
    clusname='Cluster.pkl'
    c_details=open_clusture(clusname)
    clus_count=[]
    sampletweet=[]
    sampleable=[]
    for tags in hashtags:
        tweets=[]
        tweets_list=[]
        #get tweets for each team 
        tweets= open_file(tweets,tags)
        tweets_list= get_tweets(tweets,tweets_list)
        labeled_tweets,labels = read_tarining_data('tweet_classified.csv')
        for i in labeled_tweets:
            alltweet.append(i)
        for i in labels:
            alllable.append(i)
        fname=tags+ '.csv'
        labeled_tweets,labels=read_tarining_data(fname)
        sampletweet.append(labeled_tweets)
        sampleable.append(labels[0])

    for i  in range(1,5):
        clus_count.append(len(c_details[i]))

    numberofuser= c_details['usercount']


    outf = open('summerize.txt', 'wt')
    json.dump({
               'Number of messages collected:': (len(alltweet)),
               'Number of users collected:': numberofuser,
               'Number of communities discovered:': len(clus_count),
               'average number of nodes in each cluster':clus_count, 
                'sample tweets':sampletweet[0][:5],
                'sample tweet sentiment':sampleable[:5]


              },
              outf, indent=2)
    outf.close()
if __name__ == '__main__':
    main()


# In[ ]:




# In[ ]:



