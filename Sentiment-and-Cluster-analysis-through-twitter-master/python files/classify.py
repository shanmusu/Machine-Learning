
# coding: utf-8

# In[170]:


#import all needed libraries
import pickle
from pathlib import Path
import sys
import re
import numpy as np
import time
import collections
from collections import Counter
from TwitterAPI import TwitterAPI
from collections import Counter, defaultdict, deque
import csv
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from io import BytesIO
from zipfile import ZipFile
import matplotlib.pyplot as plt
import pylab as pl
from urllib.request import urlopen
from sklearn.metrics import precision_recall_fscore_support


# In[92]:

#open the pickle and get the content of the file
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


# In[93]:

#get the tweets from pickle file to a list
def get_tweets(file,tweets_list):
    for i in file:
        t=[]
        tweets_list.append(i['tweet'])
    return tweets_list
    


# In[94]:

#get back the data that which is collected and stored in pickle file
def process_Data(allTweet):
    data = []
    for json in allTweet:
        
        tweet_data = []
        tweet_data.append(json['username'])
        tweet_data.append(json['userid'])
        tweet_data.append(json['description'])
        tweet_data.append(json['tweet'])
        tweet_data.append(json['gender'])
        tweet_data.append(json['location'])
        data.append(tweet_data)
        
    return data


# In[95]:

#write the tweets to the file twitter_data.csv
def write_tweets_csv(data):
    with open('twitter_data.csv', 'w',encoding='utf-8') as fp:
        a = csv.writer(fp)
        a.writerows(data)


# In[96]:

#collect the afinn data set which has scores for each word
def affin():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')
    afinn = dict()
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1]) 
    return afinn


# In[97]:

#calculate score for tweets which is used for manual labelling
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


# In[98]:

#create a csv file which has all tweets and corresponding label
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


# In[99]:

#store the newly labled tweets to file 
#classify number of tweets based on MALE and FEMALE with positive and negative count
def write_classification(f,logistic_prediction,tweets_list,data):
    
    tweet_classified_labelling = []
    for i in range(len(data)):
        tweet_classified_labelling.append((logistic_prediction[i],data[i][3],data[i][4],data[i][5]))
    count=dict()
    for i in range(len(tweet_classified_labelling)):
        
        if tweet_classified_labelling[i][0] == '1' and tweet_classified_labelling[i][2]== 'male' :
            
            if('male+' not in count.keys()):
                 count['male+'] =1
            else:
                 count['male+'] +=1 
        if tweet_classified_labelling[i][0] == '-1' and tweet_classified_labelling[i][2]== 'male' :
            if('male-' not in count.keys()):
                 count['male-'] =1
            else:
                count['male-'] +=1 
        if tweet_classified_labelling[i][0] == '-1' and tweet_classified_labelling[i][2]== 'female' :
            if('female-' not in count.keys()):
                count['female-'] =1
            else:
                count['female-'] +=1 
        if tweet_classified_labelling[i][0] == '1' and tweet_classified_labelling[i][2]== 'female' :
            if('female+' not in count.keys()):
                count['female+'] =1
            else:
                count['female+'] +=1 
    
    with open(f, 'w',encoding='utf-8',newline='') as fp1:
        filewriter = csv.writer(fp1)
        filewriter.writerows(tweet_classified_labelling)
    return count


# In[167]:

#read the tweets from  file back to list
def read_tarining_data(filename):
    tweets = []
    labels1 = []
    with open(filename, 'r',encoding='utf-8') as x:
        filereader = csv.reader(x)
        for  row in filereader:
            labels1.append(int(row[0]))
            tweets.append(row[1])
    return tweets,np.array(labels1)


# In[147]:

#tokenize the tweets based on Alpha numerics with no special characters allowed in it
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


# In[148]:

#create a sparse matrix based on the tweets file by selecting the features for the matrix from tweets
def do_vectorize(tokenizer_fn=tokenize, min_df=1,
                 max_df=1, binary=False, ngram_range=(1,1)):

    
    vectorizer = CountVectorizer(input = 'content', tokenizer = tokenizer_fn, min_df=min_df, 
                                     max_df=max_df, binary=True, ngram_range=ngram_range,
                                 dtype = 'int',analyzer='word',token_pattern='(?u)\b\w\w+\b',encoding='utf-8' )
    return vectorizer


# In[149]:

#Logistic regression 
def get_clf():
    return LogisticRegression()


# In[150]:

#support vector machine
def SVC():
    L=LinearSVC(C=1.0, penalty='l2',random_state=42)
    return L


# In[151]:

#linear regression
def linear():
    L=LinearRegression()
    return L


# In[152]:

#predicted values for the unlablled tweets
def prediction(CLF,trained_CSR,trained_label,untrained_tweets_CSR):
    CLF.fit(trained_CSR,trained_label)
    predicted = CLF.predict(untrained_tweets_CSR)
    return predicted


# In[153]:

#identify accuracy by using the K fold method
def do_cross_validation(X, y,clf, n_folds=5):
    cv = KFold(len(y), n_folds)
    accuracies = []
    for train_ind, test_ind in cv: 
        #print (train_ind, test_ind)
        clf.fit(X[train_ind], y[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(y[test_ind], predictions))
    avg = np.mean(accuracies)
    return avg



# In[154]:

#Identify the precision for the prediction model
def precision(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    len1 = len(y_pred)
    if len1 == 0:
        return 0
    else:
        return len(i) / len1


# In[175]:

def main():
    #team names that are used for analysis
    '''hashtags=['#mufc','#basvafc','#fcbvafc','#arsenal','#championsleague','#ucl','#FCBarcelona','#RealMadrid',
              '#CristianoRonaldo','#cr7','#LeoMessi','#messi','#mancity','#chelsea','#mcfcvcfc','#Celtic']
    '''
     hashtags=['#WeTheNorth','#GrindCity','#DubNation','#DefendTheLand','#DetroitBasketball','#thunderup','#LakeShow']
  
    tweets_list=[]
    afinn=affin()
    #get the labeled tweets and labels
    labeled_tweets,labels = read_tarining_data('tweet_manual_labelling.csv')
    """
    create object for logistic regression
    create object for support vector Machine regression
    create object for linear regression
    create object for naive bayes
    """
    clf_logistic = get_clf() 
    c_SVC = SVC()
    l_reg=linear()
    gnb = GaussianNB()
    
    #get the vectorizer object
    vectorizer = do_vectorize() 
    #tweets are vectorised and stored in X
    X = vectorizer.fit_transform(tweet for tweet in labeled_tweets)
    #corresponding values for the tweets
    y = np.array(labels)
    #fit the model with the train data
    gnb.fit(X.toarray(), y)
    y_predict = gnb.predict(X.toarray())
    nb= (do_cross_validation(X.toarray(), y,gnb))
    
    #Result of naive Bayes
    print('Average cross validation accuracy for naive Bayes Regression=%.1f percentage' % (nb*100))
    res= precision_recall_fscore_support(y,y_predict ,average='macro') 
    print ("precision",res[0])
    print ("recall",res[1])
    print ("F1-Score",res[2])
    
    clf_logistic.fit(X, y)
    p1=clf_logistic.predict(X)
    logistic_regression_accuracy= (do_cross_validation(X, y,clf_logistic))*100
    
    res= precision_recall_fscore_support(y, p1,average='macro') 
    #Result of Logistic Regression
    print('Average cross validation accuracy for Logistic Regression=%.1f percentage' % (logistic_regression_accuracy))
    print ("precision",res[0])
    print ("recall",res[1])
    print ("F1-Score",res[2])
    
    cv1= do_cross_validation(X, y,c_SVC)
    c_SVC.fit(X, y)
    p2=c_SVC.predict(X)
    svc_accuracy=(cv1*100)
    res= precision_recall_fscore_support(y, p2,average='macro') 
    #Result of Linear SVC
    print('Average cross validation accuracy for LinearSVC=%.1f percentage' % (svc_accuracy))
    print ("precision",res[0])
    print ("recall",res[1])
    print ("F1-Score",res[2])
   
    if logistic_regression_accuracy > svc_accuracy:
        clf=clf_logistic
        print('We use Logistic Regression for Classification because of Its High accuracy')
    else:
        clf=c_SVC
        print('We use LinearSVC for Classification because of Its High accuracy')
        
    location = dict() 
    
    for tags in hashtags:
        tweets=[]
        #get tweets for each team 
        tweets= open_file(tweets,tags)
        #to store in csv process the data
        
        #location are collected from the tweets 
        for t in tweets:
            if(t['location']!='' and t['location']!='.'):
                if(t['location'] not in location.keys()):
                    location[t['location']] =1
                else:
                    location[t['location']] +=1 
        tweet=dict()
        for t in tweets:
            if(t['tweet'].split()[0] == 'RT'):
                if(t['tweet'] not in tweet.keys()):
                    tweet[t['tweet']] =1
                else:
                    tweet[t['tweet']] +=1 
        f=0
        for t in tweets:
            if 'fcount' in t.keys():
                if(t['fcount']>f):
                    f=t['fcount']
                    ft=t['tweet']
        #process the data 
        data=process_Data(tweets)
        
        write_tweets_csv(data)            
        #data to be tested
        tweets_list= get_tweets(tweets,tweets_list) 
       
        #Prediction for unlabelled Tweets 
        test_tweet_vector = vectorizer.transform(t for t in tweets_list)
        logistic_prediction =prediction(clf,X,y,test_tweet_vector)
        fname=tags+ '.csv'
        c = write_classification(fname,logistic_prediction,tweets_list,data)
        class style:
            BOLD = '\033[1m'
            END = '\033[0m'
        print (style.BOLD + tags + style.END)
        
        if 'male+' in c.keys():
            print ("number of positive tweets by MALE",c['male+'])
        if 'female+' in c.keys():
            print ("number of positive tweets by FEMALE",c['female+'])
        if 'male-' in c.keys():
            print ("number of negative tweets by MALE",c['male-'])
        if 'female-' in c.keys():
            print ("number of negative tweets by FEMALE",c['female-'])
        
        result = dict(Counter(logistic_prediction)) 
        #print the results of analysis
        
        print ("Most favourite Tweet count ",f)
        print ("TWEET: ", ft)
        
        print ("Regression Results for team",tags)
        for i in result:
            if i == '-1':
                print ("\t Tweets aganist team\t\t\t\t",result[i])

            elif i == '1':
                print ("\t Number of Tweets supporting team \t\t%d" %result[i])

        labeled_tweets,labels=read_tarining_data(fname)
    print ("TOP LOCATIONS WHICH HAS MORE FOLLOWERS IN OUR ANALYSIS")     
    values = [] #in same order as traversing keys
    keys = [] #also needed to preserve order
    d = collections.Counter(location)
    for k, v in d.most_common(10):
        keys.append(k)
        values.append(v)
        print ('%s: %i' % (k, v))
    a=range(len(keys))
    plt.bar(a,values,align = 'center' )
    plt.xticks(a,keys)
    plt.xticks(a,rotation='45')
    plt.show()
    
    print ("TOP TWEETS WHICH HAS BEEN RETWEETED MOSTLY ")   
    t = collections.Counter(tweet)
    for k, v in t.most_common(10):
        print ('%s: %i' % (k, v))
if __name__ == '__main__':
    main()

