# SENTIMENT AND CLUSTER ANALYSIS THROUGH TWITTER API	

```
Developed an application in python where given the hashtags that are trending in twitter, it identifies the users who has used 
those hashtags and builds a graph between them. Using Girvan Newman cluster analysis algorithm, clusters among them are identified. 
Tweets that are collected are classified using Machine learning models like Support Vector Machine, naïve based and logistic regression
for predicting the sentiment of the tweets and accuracy comparision is made.
```

### PYTHON FILES AND ITS ACTIONS

```

 `collect.py`: It is used to collect data whihc is used for analysis. Running this script will create a file or files containing the
 data that is need for the subsequent phases of analysis.

 `cluster.py`:  It is used to read the data collected in the previous file and using  community detection algorithm to cluster users
 into communities. 

 `classify.py`:  It is used to classify data along any dimension of your choosing (e.g., sentiment, gender, spam, etc.). You may write
 any files you need to save the results.

 `summarize.py`:  It is used to read the output of the previous methods to write a textfile called `summary.txt` containing the
 following entries:

```


### 1.DATA COLLECTION

It is based on the data collected from the Twitter based on the NBA teams and its hash tags.


```
• '#DubNation'
• '#WeTheNorth'
• '#GrindCity'
• '#DetroitBasketball'
• '#thunderup'
• '#LakeShow'
• '#DefendTheLand'
```

# 1.1 Collecting unique data 

```
•	My source of data will be from twitter. 
•	We can collect data using thier API which provides data with many restriction, usually we will be able to get only data which 
are pusblished within 15 days. 
•	For making my dataset unique I need to collect the data in regular interval during game season. 
•	Each tweet has a id, each time I run my collect.py I store the recent Tweet ID and later when I re-run my code i will be able to 
get the unqiue tweets. 
•	I am collecting tweets for each hashtag and storing the tweets in of each hashtag in a pickle file with the name as hastag.
	example: ubNation.pkl
		WeTheNorth'.pkl

```

# 1.2 Data Report

```
 Number of Unique Tweets collected : 54082
 Number of unique users collected : 5985
```

# 2. cluster.py

```
•	In this process I pass my previously collected data from collect to do the cluster process. To make the clustering more 	
	sensible I choose only users who have focused and tweeted about atleast more than one team to identify clusters among them 	
	and made a graph for them.
•	I removed all user who follows or tweeted only for a single team.
•	Later the graph is sent to grivan newman to identify cluster.
•	All the teets and cluster information is stored in a file Cluster.pkl
```

# 2.1 OUTPUT FOR WHILE CLUSTER

```
	Mutual number of followers for the team #thunderup #WeTheNorth		 ----> 6
	Mutual number of followers for the team #thunderup #DefendTheLand 	 ----> 9
	Mutual number of followers for the team #thunderup #DetroitBasketball    ----> 12
	Mutual number of followers for the team #thunderup #GrindCity 		 ----> 9
	Mutual number of followers for the team #thunderup #LakeShow 		 ----> 5
	Mutual number of followers for the team #thunderup #DubNation 		 ----> 7
	Mutual number of followers for the team #WeTheNorth #DefendTheLand 	 ----> 6
	Mutual number of followers for the team #WeTheNorth #DetroitBasketball   ----> 11
	Mutual number of followers for the team #WeTheNorth #GrindCity 		 ----> 16
	Mutual number of followers for the team #WeTheNorth #LakeShow		 ----> 3
	Mutual number of followers for the team #WeTheNorth #DubNation     	 ----> 14
	Mutual number of followers for the team #DefendTheLand #DetroitBasketball----> 24
	Mutual number of followers for the team #DefendTheLand #GrindCity        ----> 8
	Mutual number of followers for the team #DefendTheLand #LakeShow 	 ----> 3
	Mutual number of followers for the team #DefendTheLand #DubNation 	 ----> 7
	Mutual number of followers for the team #DetroitBasketball #GrindCity    ----> 15
	Mutual number of followers for the team #DetroitBasketball #LakeShow     ----> 11
	Mutual number of followers for the team #DetroitBasketball #DubNation    ----> 15
	Mutual number of followers for the team #GrindCity #LakeShow 		 ----> 4
	Mutual number of followers for the team #GrindCity #DubNation 		 ----> 7
	Mutual number of followers for the team #LakeShow #DubNation  		 ----> 7

Graph has 109 nodes and 240 edges:

	Cluster 1  Number of nodes/followers: 18
	Cluster 2  Number of nodes/followers: 20
	Cluster 3  Number of nodes/followers: 20
	Cluster 4  Number of nodes/followers: 19

Cluster Information stored in file Cluster.pkl 
```

### BEFORE CLUSTURING
![image](https://github.com/mpradeep1994/Sentiment-and-Cluster-analysis-through-twitter/blob/master/cluster%20images/before_clusture.png "Optional title")
### AFTER CLUSTURING - 1
![image](https://github.com/mpradeep1994/Sentiment-and-Cluster-analysis-through-twitter/blob/master/cluster%20images/Cluster1.png "Optional title")
### AFTER CLUSTURING - 2
![image](https://github.com/mpradeep1994/Sentiment-and-Cluster-analysis-through-twitter/blob/master/cluster%20images/Cluster2.png "Optional title")
### AFTER CLUSTURING - 3
![image](https://github.com/mpradeep1994/Sentiment-and-Cluster-analysis-through-twitter/blob/master/cluster%20images/Cluster3.png "Optional title")
### AFTER CLUSTURING - 4
![image](https://github.com/mpradeep1994/Sentiment-and-Cluster-analysis-through-twitter/blob/master/cluster%20images/Cluster4.png "Optional title")


# 3. classify.py

```
•	It makes use of the data piped out of previous process and out of which a sample of tweets is labled manually by reading the 
tweets and identified as positive or negative tweet, which is then used as the training dataset to train the models which predicts the 
output lable for the test or any future data.

•	We have files with same name as the hashtag in data lake folder which has tweets corresponding to each team. once the tweets are 
read and fit into the model. The predicted lable is attached to the CSV file and placed inside the data lake. 

•	Computed average testing accuracy for each model over k-fold of Cross validation  and accuracy comparison analysis 		
is made and documented.
```
### files having tweets and its label (Sentiment)
```
	DubNation.csv
	WeTheNorth'.csv
	GrindCity.csv
	DetroitBasketball.csv
	thunderup.csv
	LakeShow.csv
	DefendTheLand.csv

output:
	Accuracy for Logistic Regression=84.4 %
	Accuracy for SVM =85.50 %
	Accuracy for Decision Tree =89.65 %

Logistic Regression Results for team #WeTheNorth
	 Number of (-Ve Tweets) aganist team          87
	 Number of (+Ve Tweets) supporting team       1259
Logistic Regression Results for team #GrindCity
	 Number of (-Ve Tweets) aganist team 		  800
	 Number of (+Ve Tweets) supporting team	      1637
Logistic Regression Results for team #DubNation
	 Number of (-Ve Tweets) aganist team 		  836
	 Number of (+Ve Tweets) supporting team	      2907
Logistic Regression Results for team #DefendTheLand
	 Number of (-Ve Tweets) aganist team 		  1288
	 Number of (+Ve Tweets) supporting team		  3684
Logistic Regression Results for team #DetroitBasketball
	 Number of (-Ve Tweets) aganist team 		  1304
	 Number of (+Ve Tweets) supporting team       3895
Logistic Regression Results for team #thunderup
	 Number of (-Ve Tweets) aganist team 		  3079
	 Number of (+Ve Tweets) supporting team		  4358
Logistic Regression Results for team #LakeShow
	 Number of (-Ve Tweets) aganist team 		  3669
	 Number of (+Ve Tweets) supporting team 	  4617

```
 
# 4. summerize.py

Information from other process is gathered and printed as result file

### Insight from the analysis:
```
•	Dubnation is a team which has minimum negative tweets Which means that the team and the team where really doing well for the season.
•	lakeshow has equal number of positive and negative tweets which makes the team to have equal number of followers and haters.
•	Like based on the count from the above result we can make conclusions from it. 

PROBLEM: since my train data has tweets mostly from the Dubnation team, the prediction for other teams 
was influenzed by the tweets of that team which made the count of negative tweets for other teams to go far high.
```
