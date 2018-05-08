import pymongo
import pandas as pa

client = pymongo.MongoClient('localhost', 27017)
db = client['db']

def find_all_restaurants():
    return db['restaurants'].find({})

def insert_to_db(json_obj, collection_name):
    db[collection_name].insert(json_obj)
#Load reviews
def loadReviews(datalimit):
    if datalimit is 0:
        df = pa.DataFrame(list(db['reviews'].find({}, {'text': 1, "_id": 0, "stars": 1, "funny": 1, "useful": 1, "cool": 1})))
    else:
        df = pa.DataFrame(list(db['reviews'].find({}, {'text': 1, "_id": 0, "stars": 1,"funny":1,"useful":1,"cool":1}).limit(datalimit)))
    return df


    #df = pa.DataFrame(list(db['restaurants'].find_one()))
    #print(df)
    #df = pa.DataFrame(list(db['users'].find_one()))
    #print(df)
    #df = pa.DataFrame(list(db['reviews'].find_one()))
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

    #print("Reviews")
    #Number of reviews for each star category
    #for i in range(6):
        #starcount=db['reviews'].find({"stars": i}, {'text':1,"_id":0,"stars":1 }).count()
        #print("Number of reviews with "+ str(i)+" stars")
        #print(starcount)

    #df = pa.DataFrame(list(db['reviews'].find({"stars": 5}, {'text':1,"_id":0,"stars":1 })))
    #print(df)