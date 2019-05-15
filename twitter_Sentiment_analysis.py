# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:01:15 2019

@author: Soham Wani
"""

import pandas as pd
import numpy as np
import re
import tweepy
from textblob import TextBlob as tb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
from textblob.classifiers import NaiveBayesClassifier
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
import pickle

#Credentials for Twitter API:
consumer_key = '<user_credentials>'
consumer_secret = '<user_credentials>'
access_token = '<user_credentials>'
access_token_secret = '<user_credentials>'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

c1 = 1
df = pd.DataFrame(columns=['TimeStamp','Tweet'])
try:
	for tweet in tweepy.Cursor(api.search,q="#narendramodi",tweet_mode='extended',
	                           count=100,lang="en", since="2019-01-01").items():
		df = df.append({'TimeStamp':tweet.created_at,'Tweet':tweet.full_text}, ignore_index=True)
		print(c1)
		c1 += 1
		
except:	
	print('Twitter limit reached')
	
finally:
	df.to_excel("narendramodi.xlsx", index=False)
	
	
# Function To clean individual Tweet:
def tidy(x):
	x = re.sub("@[\w]*", " ", x)
	x = x.lower()
	x = re.sub("(https?://[\w./]*)", " ", x)
	x = re.sub("[^a-z#]", " ", x)
	x = re.sub("#[\w]*", " ", x)
	x = ' '.join([w for w in x.split()])
	return x

# Function to assign polarity to individual Tweet:
def polarity_bin(x):
	if x > 0.05:
		return 'Positive'
	elif x < -0.05:
		return 'Negative'
	else:
		return 'Neutral'
	

# Function to assign polarity to individual Tweet:
def clean(df, name):
	stemmer = PorterStemmer()
	words = stopwords.words("english")
	analyser = SentimentIntensityAnalyzer()
	df['tidy_tweet'] = df['Tweet'].apply(lambda x: tidy(str(x)))
	df['polarity1'] = df['tidy_tweet'].apply(lambda x: polarity_bin(tb(x).sentiment.polarity))
	df['polarity2'] = df['tidy_tweet'].apply(lambda x: polarity_bin(analyser.polarity_scores(x)['compound']))
	df['cleaned'] = df['tidy_tweet'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", str(x)).split() if i not in words]))
	df['Name'] = name
	return df


def piechart(df1,df2,p=1):
	labels = ['Negative', 'Neutral', 'Positive']
	explode = [0,0,0.1]
	fig, axs = plt.subplots(1, 2)
	fig.suptitle('Twitter Sentiments', fontsize=24)
	axs[0].pie(df1.Tweet.groupby(df1['polarity{}'.format(p)]).count(), labels=labels, explode=explode, 
	   textprops={'fontsize': 14}, autopct='%1.1F%%')
	axs[0].set_title(df1['Name'][0],fontsize = 20)
	axs[0].set_xlabel('Total Tweets: {}'.format(df1['polarity{}'.format(p)].count()),fontsize = 14)
	axs[1].pie(df2.Tweet.groupby(df2['polarity{}'.format(p)]).count(), labels=labels, explode=explode, 
	   textprops={'fontsize': 14}, autopct='%1.1F%%')
	axs[1].set_title(df2['Name'][0],fontsize = 20)
	axs[1].set_xlabel('Total Tweets: {}'.format(df2['polarity{}'.format(p)].count()),fontsize = 14)
	return plt.show()


def hashcloud(df,stop=None):
	n = []
	for i in df['Tweet']:
	    h = re.findall('#[A-Za-z0-9]*',str(i))
	    n.extend(h)
	all_words = ' '.join([text for text in n])
	wordcloud = WordCloud(width=700, height=500, random_state=21, max_font_size=80,
					   max_words=100, collocations=False,background_color='white',
					   stopwords=stop).generate(all_words)
	plt.figure(figsize=(10, 7))
	plt.imshow(wordcloud)
	plt.axis('off')
	return plt.show()

def isRT(x):
	if re.search('^RT', str(x)) == None:
		return 'No'
	else:
		return 'Yes'


def classify(x, model):
	return model.classify(x)
	
	
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    

def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def precision_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows


def recall_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns


def f1score(confusion_matrix):
	r = recall_average(confusion_matrix)
	p = precision_average(confusion_matrix)
	return (2*r*p)/(r+p)


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


# Loading Datasets into DataFrame:
modi = pd.read_excel('Namo.xlsx')
name1 = 'Narendra Modi'
modi = clean(modi, name1)

rahul = pd.read_excel('Raga.xlsx')
name2 = 'Rahul Gandhi'
rahul = clean(rahul, name2)

# Combining DataFrames:
df = pd.concat([modi,rahul], ignore_index=True, sort=False)

### Visualization:
# Pie Chart of Sentiments:
# Change 
piechart(modi, rahul, p=2)

# HashTags Word Cloud:
hashcloud(df)

# HashTags Word Cloud by restricting some words:
stop = ['RahulGandhi','NarendraModi']
hashcloud(df,stop)

# Pivot Table of Sentiments:
# For Tweets Labaled using TextBlob sentiment polarity:
c = pd.crosstab(index=df['polarity1'],columns=df['Name'],rownames=['Sentiment'])
print(c)

# For Tweets Labaled using VaderSentiment:
d = pd.crosstab(index=df['polarity2'],columns=df['Name'],rownames=['Sentiment'])
print(d)

## Preparing dataframe for Model Building:
df['RT'] = df['Tweet'].apply(lambda x: isRT(x))

df_c = df.loc[df['RT']=='No']


df_train, df_test = train_test_split(df_c, test_size=0.2, random_state = 42)

#Slicing only required columns for model building:
df_train = df_train[['tidy_tweet','polarity2']]
df_test = df_test[['tidy_tweet','polarity2']]

train = []
for row in df_train.itertuples(index=False, name=None):
	train.append(row)
	
# # Naive Bayes Model:
model = NaiveBayesClassifier(train)

# Classification:
classify('considered most serving politician india being honoured with', model)

# Classifing Test Set:
df_test['Predicted'] = df_test['tidy_tweet'].apply(lambda x: classify(x, model))

# Evaluating Results:
# Confusion Matrix:
con_mat = cm(df_test['polarity2'], df_test['Predicted'])

from pandas_ml import ConfusionMatrix
confusion_matrix = ConfusionMatrix(df_test['polarity2'], df_test['Predicted'])
print("Confusion matrix for Naive Bayes:\n%s" % confusion_matrix)

print('For Naive Bayes model: \nPrecision: {0:.3f} \nRecall: {1:.3f} \nf1score: {2:.3f} \nAccuracy: {3:.3f}'.format(precision_average(con_mat),recall_average(con_mat),f1score(con_mat),accuracy(con_mat)))

acc = accuracy(con_mat)
confusion_matrix.plot(normalized=True)
plt.title('Naive Bayes \nAccuracy:{0:.3f}'.format(acc))
plt.show()


# Spliting our dataset into testing and training set:
X_train, X_test, y_train, y_test = train_test_split(df_c['cleaned'], df_c['polarity2'], 
													test_size=0.20, random_state = 42)

vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))

final_features = vectorizer.fit_transform(df_c['cleaned']).toarray()
final_features.shape


# Random Forest:
# Instead of doing these steps one at a time, we can use a pipeline to complete them all at once
randomforest = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', RandomForestClassifier(random_state=42))])
	
# fitting our model and save it in a pickle for later use
model = randomforest.fit(X_train, y_train)
with open('RandomForest.pickle', 'wb') as f:
    pickle.dump(model, f)
ytest = np.array(y_test)

# Evaluating Results:
# Confusion Matrix:
con_mat = cm(ytest, model.predict(X_test))

confusion_matrix = ConfusionMatrix(ytest, model.predict(X_test))
print("Confusion matrix for Random Forest:\n%s" % confusion_matrix)

print('For Random Forest model: \nPrecision: {0:.3f} \nRecall: {1:.3f} \nf1score: {2:.3f} \nAccuracy: {3:.3f}'.format(precision_average(con_mat),recall_average(con_mat),f1score(con_mat),accuracy(con_mat)))

# Confusion matrix plot:
acc = accuracy(con_mat)
confusion_matrix.plot(normalized=True)
plt.title('Random Forest \nAccuracy:{0:.3f}'.format(acc))
plt.show()


# SVM:
supportvector = Pipeline([('vect', vectorizer),
                     ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5, random_state=42))])
model = supportvector.fit(X_train, y_train)    

# fitting our model and save it in a pickle for later use
with open('svm.pickle', 'wb') as f:
    pickle.dump(model, f)
ytest = np.array(y_test)

# Evaluating Results:
# Confusion Matrix:
con_mat = cm(ytest, model.predict(X_test))

confusion_matrix = ConfusionMatrix(ytest, model.predict(X_test))
print("Confusion matrix for SVM:\n%s" % confusion_matrix)

print('For SVM model: \nPrecision: {0:.3f} \nRecall: {1:.3f} \nf1score: {2:.3f} \nAccuracy: {3:.3f}'.format(precision_average(con_mat),recall_average(con_mat),f1score(con_mat),accuracy(con_mat)))

# Confusion matrix plot:
acc = accuracy(con_mat)
confusion_matrix.plot(normalized=True)
plt.title('SVM \nAccuracy:{0:.3f}'.format(acc))
plt.show()