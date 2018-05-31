import pandas as pa
import pymongo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn import tree
from itertools import islice
from sklearn.metrics import accuracy_score
import project.src.TextProcessing as tx
import project.src.loadData as data
import project.src.dataAnalysis as dataAnalysis
import project.src.features as feat
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import time
from bson import json_util
import sys
import pickle


class ListStream:
    def __init__(self):
        self.data = []

    def write(self, s):
        self.data.append(s)


client = pymongo.MongoClient('localhost', 27017)
db = client['db']
db_diagrams = client['diagrams']

#nltk.download('stopwords')#download the list with stopwords if not exist
RANDOM_STATE = 0
number = 1000# amount of data you want to load from reviews number=0 brings all data

#Predict and print results
def print_results(clf, label='DEFAULT'):
    print('********* ' + label + " *********")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    start = time.time()
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    stop = time.time()
    print("%20s accuracy: %0.3f (+/- %0.3f), time:%.3f" % (label, scores.mean(), scores.std() * 2, stop - start))

#fit models
def fit_clf(*args):
        pipeline = make_pipeline(*args)
        start = time.time()
        pipeline.fit(X_train, y_train)
        return pipeline

def loadFilesIfEmptyDB(file1, file2, file3):
    response = open(file1,"r")
    data = json_util.loads(response.read())
    connection = MongoClient('mongodb+srv://dimitris:p3si0s1983@flaskwebapp-jtxjt.mongodb.net/test?retryWrites=true', 27017)
    connection.database_names()
    db = connection.database
    restaurants = db.restaurants
    restaurants.insert_many(data).inserted_id


    response = open(file2, "r")
    data = json_util.loads(response.read())
    connection = MongoClient('mongodb+srv://dimitris:p3si0s1983@flaskwebapp-jtxjt.mongodb.net/test?retryWrites=true',
                             27017)
    connection.database_names()
    db = connection.database
    reviews = db.reviews
    reviews.insert_many(data).inserted_id

    response = open(file3, "r")
    data = json_util.loads(response.read())
    connection = MongoClient('mongodb+srv://dimitris:p3si0s1983@flaskwebapp-jtxjt.mongodb.net/test?retryWrites=true',
                             27017)
    connection.database_names()
    db = connection.database
    users = db.users
    users.insert_many(data).inserted_id

def ifDBisEmpty():
    connection = MongoClient('mongodb+srv://dimitris:p3si0s1983@flaskwebapp-jtxjt.mongodb.net/test?retryWrites=true',27017)
    db = connection['db']
    collection1 = db['users']
    collection2 = db['restaurants']
    collection3 = db['reviews']

    return collection1.count() == 0 or collection2.count() == 0 or collection3.count() == 0  # Check if any collections is empty

def storeFitObjectIntoDB(serialized_object, name):
    db_diagrams.results.insert({name: serialized_object})

if __name__ == '__' \
               'main__':
    #if ifDBisEmpty():
    #    loadFilesIfEmptyDB('restaurants.json','reviews.json','users.json')

    df = data.loadReviews(number)#load review data
    #data.starCount()
    #print(df)

    df.info()
    #stars = df.groupby('stars').mean()
    #print(stars)
    #ax=sns.heatmap(data=stars.corr(), annot=True)
    #plt.show()

    df=tx.stemming(df,"text","stemReviews")#calling method to stem column1 and put result  to column2
    df=tx.remove_stopword(df,"stemReviews","stopWordReviews")#calling method to stopword column1 and put result  to column2
    df=tx.upper_to_lower(df,"stopWordReviews","lowerReviews")#calling method to transform upper to lower column1 and put result  to column2
    df=tx.remove_punctions(df,"lowerReviews","finalReviews")#calling method to remove punctions column1 and put result  to column2
    #print(df)
    print("---------------Final Dataframe:-----------------")
    df.drop(df.columns[[5,6,7,8,9]], axis=1, inplace=True)#drops the colums we want
    print(df)

    # The categories of stars 1,2,3,4,5 reduced to 0,1
    df=feat.MulticlassToBinary(df)
    #print(df)

    #dataAnalysis.codenotused(df)


    #Tranform reviews to tf-idf vectors
    tvec_weights=tx.textTFIDF(df)
    #print(tvec_weights)
    #tvec_weights=tx.textCountVec(df)
    #----------Classifications--------------------
    X_train, X_test, y_train, y_test = train_test_split(tvec_weights,df["targets"], test_size=0.33, random_state=RANDOM_STATE)


    #models
    clfs = [
       {'name': 'RandomForestClassifier', 'obj': RandomForestClassifier(random_state=RANDOM_STATE)},
        {'name': 'LinearSVC', 'obj': LinearSVC(random_state=RANDOM_STATE)},
        {'name': 'MultinomialNB', 'obj':  MultinomialNB()},
        {'name': 'AdaBoost', 'obj':AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=300)},
        {'name': 'Logistic Regresion',"obj":LogisticRegression(random_state=RANDOM_STATE)},
        {'name': 'Neural MLP',"obj":MLPClassifier(random_state=RANDOM_STATE)}
    ]

    #Run models
    for c in  clfs:
        sys.stdout = x = ListStream()  # capture the stdout stream
        print_results(fit_clf(c['obj']), c['name'])
        sys.stdout = sys.__stdout__
        storeFitObjectIntoDB(x.data,  c['name'])