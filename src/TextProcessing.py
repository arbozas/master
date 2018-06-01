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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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
    df["tokenized column"] =df[column].apply(lambda x: filter(None, x.split(" ")))#tokenize column
    df.drop(column, axis=1, inplace=True)
    df['stemmed column'] = df["tokenized column"].apply(lambda x: [porter_stemmer.stem(y) for y in x])#stem column
    df.drop("tokenized column", axis=1, inplace=True)
    df[newColumn]=df['stemmed column'].apply(lambda x : " ".join(x))
    df.drop("stemmed column", axis=1, inplace=True)
    return df;

#Remove stopwords from a column of a dataframe and put the results to a new column
def remove_stopword(df,column,newColumn):
    stop = stopwords.words('english')
    df[newColumn] = df[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))#remove stopwords
    df.drop(column, axis=1, inplace=True)
    return df;

#Transform letter to lower from a column of a dataframe and put the results to a new column
def upper_to_lower(df,column,newColumn):
    df[newColumn] = df[[column]].apply(lambda name: name.str.lower())#upper to lower s
    df.drop(column, axis=1, inplace=True)
    return df;

#Transform text to tfIDF
def textTFIDF(df):
    tvec = TfidfVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2))#initialize TFIDF VECTORIZER
    tvec_weights = tvec.fit_transform(df.finalReviews.dropna())#Fit
    X_normalized = preprocessing.normalize(tvec_weights, norm='l2')#normalize

    return X_normalized

#Create features for CNN
def FeatureForCNN(df):
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(df.finalReviews)
    sequences = tokenizer.texts_to_sequences(df.finalReviews.values)
    data = pad_sequences(sequences, maxlen=1000)

    return data