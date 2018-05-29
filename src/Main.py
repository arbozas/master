import pandas as pa
import pymongo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import TextProcessing as tx
import loadData as data
import dataAnalysis as dataAnalysis
#import modelCNN as modelCNN
import features as feat
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import nltk
import time
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler


client = pymongo.MongoClient('localhost', 27017)
db = client['db']

#nltk.download('stopwords')#download the list with stopwords if not exist
RANDOM_STATE = 0
number =1000# amount of data you want to load from reviews number=0 brings all data

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

if __name__ == '__' \
               'main__':

    df = data.loadReviews(number)#load review data
    df = df[df.stars != 3]#remove 3 stars reviews
    df=feat.MulticlassToBinary(df)
    print(df.head())
    print('Information of the original data set: \n {}'.format(Counter(df.targets)))
    data.plot_pie(df.targets)
    ratio = 'auto'
    #X,Y = RandomUnderSampler(ratio=ratio, random_state=0).fit_sample(df["text"],df["targets"])
    df_targets=df["targets"]
    df_data=df.loc[:, ['text', 'review_count']]#put more features here
    print(df_data)
    features=df_data.values
    balanced_x, balanced_y = feat.balance_classes(features,df_targets)
    df=pa.DataFrame(balanced_x,columns=["text","review_Count"])
    df_targets=pa.DataFrame(balanced_y,columns=["targets"])
    print('Information of  set after balancing using "auto"'
        ' mode:\n ratio={} \n y: {}'.format(ratio, Counter(df_targets.targets)))

    #df.info()
    #stars = df.groupby('stars').mean()
    #print(stars)
   # ax=sns.heatmap(data=stars.corr(), annot=True)
    #plt.show()
    data.plot2(1)

    df=tx.stemming(df,"text","stemReviews")#calling method to stem column1 and put result  to column2
    df=tx.remove_stopword(df,"stemReviews","stopWordReviews")#calling method to stopword column1 and put result  to column2
    df=tx.upper_to_lower(df,"stopWordReviews","lowerReviews")#calling method to transform upper to lower column1 and put result  to column2
    df=tx.remove_punctions(df,"lowerReviews","finalReviews")#calling method to remove punctions column1 and put result  to column2

    print("---------------Final Dataframe:-----------------")
    print(df.head())

    # The categories of stars 1,2,3,4,5 reduced to 0,1

    #dataAnalysis.codenotused(df)


    #Tranform reviews to tf-idf vectors
    features=tx.textTFIDF(df)

    #print(tvec_weights)
    #print(df.review_Count.values)


    #print(df_targets)
    #print(tvec_weights)
    #tvec_weights=tx.textCountVec(df)
    #----------Classifications--------------------

    X_train, X_test, y_train, y_test = train_test_split(features,df_targets.values.ravel(), test_size=0.33, random_state=RANDOM_STATE)

    #models
    clfs = [
       {'name': 'RandomForestClassifier', 'obj': RandomForestClassifier(random_state=RANDOM_STATE)},
        {'name': 'LinearSVC', 'obj': LinearSVC(random_state=RANDOM_STATE)},
        {'name': 'MultinomialNB', 'obj':  MultinomialNB()},
        #{'name': 'AdaBoost', 'obj':AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=300)},
        {'name': 'Logistic Regresion',"obj":LogisticRegression(random_state=RANDOM_STATE)},
        #{'name': 'Neural MLP',"obj":MLPClassifier(random_state=RANDOM_STATE)}
    ]
    #Run models
    for  c in  clfs:
        print_results(fit_clf(c['obj']), c['name'])#print results

    #modelCNN.modelCNN(X_train, X_test, y_train, y_test)