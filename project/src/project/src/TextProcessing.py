import pymongo
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from itertools import islice
import numpy as np

import pandas as pa

#nltk.download('stopwords')#download the list with stopwords if not exist
client = pymongo.MongoClient('localhost', 27017)
db = client['db']

#Remove punctuation from a column of a dataframe and put the results to a new column
def remove_punctions(df,column,newColumn):
    df[newColumn]=df[column].str.replace('[^\w\s]', '')
    return df;

#Steemming a column of a dataframe and put the results to a new column
def stemming(df,column,newColumn):
    porter_stemmer = PorterStemmer()
    df["tokenized column"] =df[column].apply(lambda x: filter(None, x.split(" ")))
    df['stemmed column'] = df["tokenized column"].apply(lambda x: [porter_stemmer.stem(y) for y in x])
    df[newColumn]=df['stemmed column'].apply(lambda x : " ".join(x))
    return df;

#Remove stopwords from a column of a dataframe and put the results to a new column
def remove_stopword(df,column,newColumn):
    stop = stopwords.words('english')
    df[newColumn] = df[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return df;

#Transform letter to lower from a column of a dataframe and put the results to a new column
def upper_to_lower(df,column,newColumn):
    df[newColumn] = df[[column]].apply(lambda name: name.str.lower())
    return df;

def textTFIDF(df):
    tvec = TfidfVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2))#initialize TFIDF VECTORIZER
    tvec_weights = tvec.fit_transform(df.finalReviews.dropna())#Fit
    weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
    weights_df = pa.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
    print(weights_df.sort_values(by='weight', ascending=False).head(20))
    return tvec_weights


