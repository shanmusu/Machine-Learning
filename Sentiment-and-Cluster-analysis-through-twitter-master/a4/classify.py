
# coding: utf-8

# In[49]:

import pickle
from pathlib import Path
import sys
import re
import numpy as np
import time
from collections import Counter
from TwitterAPI import TwitterAPI
from collections import Counter, defaultdict, deque
import csv
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


# In[50]:

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


# In[51]:

def get_tweets(file,tweets_list):
    for i in file:
        t=[]
        tweets_list.append(i['tweet'])
    return tweets_list
    


# In[52]:

def process_Data(allTweet):
    data = []
    for json in allTweet:
        tweet_data = []
        tweet_data.append(json['username'])
        tweet_data.append(json['userid'])
        tweet_data.append(json['description'])
        tweet_data.append(json['tweet'])
       
        data.append(tweet_data)
    return data


# In[53]:

def write_tweets_csv(data):
    with open('twitter_data.csv', 'w',encoding='utf-8') as fp:
        a = csv.writer(fp)
        a.writerows(data)
   


# In[55]:

def affin():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')
    afinn = dict()
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1]) 


# In[56]:

def afinn_sentiment2(terms, afinn, verbose=False):
    pos = 0
    neg = 0
    val=0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
                
            else:
                neg += -1 * afinn[t]
                
    val=pos+neg
    
    if val < 0 :
        return  -1
    else:
        return 1
        


# In[57]:

def create_for_manual(tweet_test,afinn):
    tweet_manual_labelling = []
    for i in tweet_test:
        i = re.sub('http\S+', 'THIS_IS_A_URL', i)
        i = re.sub('@\S+', 'THIS_IS_A_MENTION', i)
        if i.split()[0] != 'RT':
            score= afinn_sentiment2(i, afinn, verbose=False)
            
            tweet_manual_labelling.append((score,i))
    with open('tweet_manual_labelling.csv', 'w',encoding='utf-8',newline='') as fp1:
        filewriter = csv.writer(fp1)
        filewriter.writerows(tweet_manual_labelling)


# In[58]:

def write_classification(f,logistic_prediction,tweets_list):
    
    tweet_classified_labelling = []
    for i in range(len(tweets_list)):
        tweet_classified_labelling.append((logistic_prediction[i],tweets_list[i]))
   
    with open(f, 'w',encoding='utf-8',newline='') as fp1:
        filewriter = csv.writer(fp1)
        filewriter.writerows(tweet_classified_labelling)


# In[59]:

def read_tarining_data(filename):
    tweets = []
    labels1 = []
    with open(filename, 'r',encoding='utf-8') as x:
        filereader = csv.reader(x)
        for  row in filereader:
            labels1.append(row[0])
            tweets.append(row[1])
    return tweets,np.array(labels1)


# In[60]:

def tokenize(text):
    
    tokens = re.findall(r"\w+|\S", text.lower(),flags = re.L)
    tokens1 = []
    for i in tokens:
        i = re.sub('http\S+', 'THIS_IS_A_URL', i)
        i = re.sub('@\S+', 'THIS_IS_A_MENTION', i)
        x = re.findall(r"\w+|\S", i,flags = re.U)
        for j in x:
            tokens1.append(j)            
    return tokens1


# In[61]:

def do_vectorize(tokenizer_fn=tokenize, min_df=1,
                 max_df=1., binary=False, ngram_range=(1,1)):

    
    vectorizer = CountVectorizer(input = 'content', tokenizer = tokenizer_fn, min_df=min_df, 
                                     max_df=max_df, binary=True, ngram_range=ngram_range,
                                 dtype = 'int',analyzer='word',token_pattern='(?u)\b\w\w+\b',encoding='utf-8' )
    return vectorizer


# In[62]:

def get_clf():
    return LogisticRegression()


# In[63]:

def prediction(CLF,trained_CSR,trained_label,untrained_tweets_CSR):
    CLF.fit(trained_CSR,trained_label)
    predicted = CLF.predict(untrained_tweets_CSR)
    return predicted


# In[64]:

def do_cross_validation(X, y,clf, n_folds=5):
    cv = KFold(len(y), n_folds)
    accuracies = []
    for train_ind, test_ind in cv: 
        clf.fit(X[train_ind], y[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(y[test_ind], predictions))
    avg = np.mean(accuracies)
    return avg



# In[66]:

def main():
    hashtags=['#WeTheNorth','#GrindCity','#DubNation','#DefendTheLand','#DetroitBasketball','#thunderup','#LakeShow']
    #hashtags=['#DubNation']
    tweets_list=[]

    labeled_tweets,labels = read_tarining_data('tweet_manual_labelling.csv')
    clf_logistic = get_clf() 
    #get the vectorizer object
    vectorizer = do_vectorize() 
    X = vectorizer.fit_transform(tweet for tweet in labeled_tweets)
    y = np.array(labels) 
    logistic_regression_accuracy = (do_cross_validation(X, y,clf_logistic))*100
    print('Average cross validation accuracy for Logistic Regression=%.1f percentage' % (logistic_regression_accuracy))
    for tags in hashtags:
        tweets=[]
        #get tweets for each team 
        tweets= open_file(tweets,tags)
        #to store in csv process the data
        data=process_Data(tweets)
        write_tweets_csv(data)    
        #data to be tested
        tweets_list= get_tweets(tweets,tweets_list)
        #create_for_manual(tweets_list,afinn) only one -- done for you
        #Prediction for unlabelled Tweets
        test_tweet_vector = vectorizer.transform(t for t in tweets_list)
        logistic_prediction =prediction(clf_logistic,X,y,test_tweet_vector)
        fname=tags+ '.csv'
        write_classification(fname,logistic_prediction,tweets_list)

        result = dict(Counter(logistic_prediction))
   


        print ("Logistic Regression Results for team",tags)
        for i in result:
            if i == '-1':
                print ("\t Tweets aganist team\t\t\t\t",result[i])

            elif i == '0':
                print ("\t Number of advertising tweets on team \t\t%d" %result[i])

            elif i == '1':
                print ("\t Number of Tweets supporting team \t\t%d" %result[i])

        labeled_tweets,labels=read_tarining_data(fname)
        
if __name__ == '__main__':
    main()


# In[ ]:




# In[ ]:




# In[31]:




# In[ ]:



