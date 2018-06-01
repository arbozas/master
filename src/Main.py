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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn import tree
from itertools import islice
from sklearn.metrics import accuracy_score
import project.src.TextProcessing as tx
import project.src.loadData as data
import project.src.Results as results
import project.src.diagram as diagram
import project.src.features as feat
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import nltk
import time
from collections import Counter
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential


client = pymongo.MongoClient('localhost', 27017)
db = client['db']

nltk.download('stopwords')#download the list with stopwords if not exist
RANDOM_STATE = 0
number =1000# amount of data you want to load from reviews number=0 brings all data

#Predict and print results
def print_results(clf, label='DEFAULT'):
    print('********* ' + label + " *********")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    df[label]=y_pred.tolist()#add precitions to df in a column named as the models name
    start = time.time()
    scores = cross_val_score(clf, features,df_targets.values.ravel(), cv=10)
    stop = time.time()
    print("%20s accuracy: %0.3f (+/- %0.3f), time:%.3f" % (label, scores.mean(), scores.std() * 2, stop - start))

#fit models
def fit_clf(*args):
        pipeline = make_pipeline(*args)
        pipeline.fit(X_train, y_train)
        return pipeline

if __name__ == '__' \
               'main__':

    df = data.loadReviews(number)#load review data
    df = df[df.stars != 3]#remove 3 stars reviews
    df=feat.MulticlassToBinary(df)#Makes the classes 1-2 to 0(negative) and classes 4-5 to 1(positive)
    print('Information of the original data set: \n {}'.format(Counter(df.targets)))
    #diagram.plot_pie(df.targets)
    df_targets=df["targets"]
    features=df.values

    #balance the classes by underesampling the most common class
    balanced_x, balanced_y = feat.balance_classes(features,df_targets)
    df=pa.DataFrame(balanced_x,columns=["text","review_Count"])
    df_targets=pa.DataFrame(balanced_y,columns=["targets"])

    df=tx.stemming(df,"text","stemReviews")#calling method to stem column1 and put result  to column2
    df=tx.remove_stopword(df,"stemReviews","stopWordReviews")#calling method to stopword column1 and put result  to column2
    df=tx.upper_to_lower(df,"stopWordReviews","lowerReviews")#calling method to transform upper to lower column1 and put result  to column2
    df=tx.remove_punctions(df,"lowerReviews","finalReviews")#calling method to remove punctions column1 and put result  to column2

    print("---------------Final Dataframe:-----------------")
    print(df.head())


    data_cnn=tx.FeatureForCNN(df)#Final data for CNN
    features=tx.textTFIDF(df)#Final Data for other models
    df = pa.DataFrame()

    #----------Classifications--------------------

    X_train, X_test, y_train, y_test = train_test_split(features,df_targets.values.ravel(), test_size=0.33, random_state=RANDOM_STATE)
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(data_cnn, df_targets.values.ravel(), test_size=0.33,
                                                        random_state=RANDOM_STATE)
    #models
    clfs = [
       {'name': 'RandomForestClassifier', 'obj': RandomForestClassifier(random_state=RANDOM_STATE)},
        {'name': 'LinearSVC', 'obj': LinearSVC(random_state=RANDOM_STATE)},
        {'name': 'MultinomialNB', 'obj':  MultinomialNB(alpha=0.7)},
        {'name': 'Logistic Regresion',"obj":LogisticRegression(random_state=RANDOM_STATE)},

    ]

    #Run models
    for  c in  clfs:
        print_results(fit_clf(c['obj']), c['name'])#print results

    #Convolutional model
    model = Sequential()
    model.add(Embedding(20000, 128, input_length=1000))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=3, batch_size=32, verbose=2)
    y_pred_cnn=model.predict(X_test_cnn,batch_size=64)
    df["CNN"] = y_pred_cnn#add precitions to df in a column named CNN

    #Plot ROC results
    results.printRoc(y_test,y_test_cnn,df)