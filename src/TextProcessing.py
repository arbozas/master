import pymongo
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from itertools import islice
from sklearn import preprocessing
import numpy as np
import pandas as pa
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

client = pymongo.MongoClient('localhost', 27017)
db = client['db']

#Remove punctuation from a column of a dataframe and put the results to a new column
def remove_punctions(df,column,newColumn):
    df[newColumn]=df[column].str.replace('[^\w\s]', '')#remove punctuation
    df.drop(column, axis=1, inplace=True)
    df[newColumn] = df[newColumn].str.replace('\d+', '')#remove numbers
    df[newColumn] = df[newColumn].str.replace(' +', ' ')#remove multiple empty spaces
    df[newColumn] = df[newColumn].str.strip()
    return df;

#Steemming a column of a dataframe and put the results to a new column
def stemming(df,column,newColumn):
    porter_stemmer = PorterStemmer()
    df["tokenized column"] =df[column].apply(lambda x: filter(None, x.split(" ")))
    df.drop(column, axis=1, inplace=True)
    df['stemmed column'] = df["tokenized column"].apply(lambda x: [porter_stemmer.stem(y) for y in x])
    df.drop("tokenized column", axis=1, inplace=True)
    df[newColumn]=df['stemmed column'].apply(lambda x : " ".join(x))
    df.drop("stemmed column", axis=1, inplace=True)
    return df;

#Remove stopwords from a column of a dataframe and put the results to a new column
def remove_stopword(df,column,newColumn):
    stop = stopwords.words('english')
    df[newColumn] = df[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df.drop(column, axis=1, inplace=True)
    return df;

#Transform letter to lower from a column of a dataframe and put the results to a new column
def upper_to_lower(df,column,newColumn):
    df[newColumn] = df[[column]].apply(lambda name: name.str.lower())
    df.drop(column, axis=1, inplace=True)
    return df;

def textTFIDF(df):
    tvec = TfidfVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2))#initialize TFIDF VECTORIZER
    tvec_weights = tvec.fit_transform(df.finalReviews.dropna()).toarray()#Fit

    #if YOU WANT TO ADD MORE FEATURE LIKE THIS
    #review_count=df.review_Count.values.reshape((len(df.review_Count.values), 1))
    #data = np.concatenate((tvec_weights,review_count), axis=1)
    #print(data)

    X_normalized = preprocessing.normalize(tvec_weights, norm='l2')
    return X_normalized

def textCountVec(df):
    cvec = CountVectorizer(min_df=.0025, max_df=.1, ngram_range=(1,2))
    cvec.fit(df.finalReviews)
    print(len(cvec.vocabulary_.items()))
    cvec_counts = cvec.transform(df.finalReviews).todense()

    #print(cvec_counts.shape)
    transformer = TfidfTransformer()
    transformed_weights = transformer.fit_transform(cvec_counts)
    #print(transformed_weights)
    #print(transformed_weights.shape)
    #X_normalized = preprocessing.normalize(cvec_counts, norm='l2')
    #print(X_normalized)
    return transformed_weights

