import pymongo
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import pandas as pa
#nltk.download('stopwords')#download the list with stopwords if not exist
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['yelp_db']

def find_all_restaurants():
    return db['restaurants'].find({})

def insert_to_db(json_obj, collection_name):
    db[collection_name].insert(json_obj)

#Remove punctuation from a column of a dataframe and put the results to a new column
def remove_punctions(df,column,newColumn):
    df[newColumn]=df[column].str.replace('[^\w\s]', '')

#Steemming a column of a dataframe and put the results to a new column
def stemming(df,column,newColumn):
    porter_stemmer = PorterStemmer()
    df["tokenized column"] =df[column].apply(lambda x: filter(None, x.split(" ")))
    df['stemmed column'] = df["tokenized column"].apply(lambda x: [porter_stemmer.stem(y) for y in x])
    df[newColumn]=df['stemmed column'].apply(lambda x : " ".join(x))

#Remove stopwords from a column of a dataframe and put the results to a new column
def remove_stopword(df,column,newColumn):
    stop = stopwords.words('english')
    df[newColumn] = df[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    #df["tokenized column"] =df[column].apply(lambda x: filter(None, x.split(" ")))
    #df[newColumn]=df["tokenized column"].apply(lambda x: [item for item in x if item not in stop])

#Transform letter to lower from a column of a dataframe and put the results to a new column
def upper_to_lower(df,column,newColumn):
    df[newColumn] = df[[column]].apply(lambda name: name.str.lower())

if __name__ == '__main__':

    #print(df)
    df = pa.DataFrame(list(db['users'].find_one()))
    #print(df)
    df = pa.DataFrame(list(db['reviews'].find_one()))
    # print(df)

    # Find the coordinates for each restaurant and
    # save them to an external collection
    # all_restaurants = find_all_restaurants()
    # for restaurant in all_restaurants:
    #     json_obj = {
    #         'name': restaurant['name'],
    #         'business_id': restaurant['business_id'],
    #         'longitude': restaurant['longitude'],
    #         'latitude': restaurant['latitude']
    #     }
    #     insert_to_db(json_obj, 'restaurants_coordinates')

    print("Reviews")
    #Number of reviews for each star category
    #for i in range(6):
        #starcount=db['reviews'].find({"stars": i}, {'text':1,"_id":0,"stars":1 }).count()
        #print("Number of reviews with "+ str(i)+" stars")
        #print(starcount)

    #df = pa.DataFrame(list(db['reviews'].find({"stars": 5}, {'text':1,"_id":0,"stars":1 })))
    #print(df)

    df = pa.DataFrame(list(db['reviews'].find({"stars": 5}, {'text': 1, "_id": 0, "stars": 1}).limit(5)))
    stemming(df,"text","stemReviews")#calling method to stem column1 and put result  to column2
    remove_stopword(df,"stemReviews","stopWordReviews")#calling method to stopword column1 and put result  to column2
    upper_to_lower(df,"stopWordReviews","lowerReviews")#calling method to transform upper to lower column1 and put result  to column2
    remove_punctions(df,"lowerReviews","finalReviews")#calling method to remove punctions column1 and put result  to column2
    print(df)
    print("---------------Final Dataframe:-----------------")
    df.drop(df.columns[[1,2,3,4,5,6]], axis=1, inplace=True)#drops the colums we want
    print(df)
