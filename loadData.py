import pymongo
import pandas as pa
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from dateutil.parser import parse
import gridfs
import logging
import sys

import numpy as np
client = pymongo.MongoClient('localhost', 27017)
db = client['db']
db_diagrams = client['diagrams']

def find_all_restaurants():
    return db['restaurants'].find({})

def insert_to_db(json_obj, collection_name):
    db[collection_name].insert(json_obj)

#Load reviews
def plot_pie(y):
    target_stats = Counter(y)
    labels = target_stats.keys()
    sizes = target_stats.values()
    explode = tuple([0.1] * len(target_stats))
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True,
           autopct='%1.1f%%')
    ax.axis('equal')
    # storing image into the database for later use
    storeImagePlotInDB(plt, db_diagrams, "third")
    plt.show()

def plot2(s):
    df_rev = pa.DataFrame(list(db['reviews'].find({}, {"_id":0,"stars":1,"business_id":1,"user_id":1})))
    df_rest= pa.DataFrame(list(db['restaurants'].find({}, { "_id": 0, "categories":1,"business_id":1,"review_count":1,"neighborhood":1})))
    df_user=pa.DataFrame(list(db['users'].find({}, { "_id": 0, "user_id":1,"review_count":1})))
    df_rev_rest=pa.merge(df_rev, df_rest, on='business_id', how='inner')
    #print(df_rev_rest)
    df_rev_users = pa.merge(df_rev, df_user, on='user_id', how='inner')
   # print(len(df_rest.categories.values))

    print(df_rest.describe(include='all'))
    #df_rest["categories"] = df_rest["categories"].apply(lambda x: filter(None, x.split(",")))
    #print(df_categories)

    # plot stars numbers
    plt.hist(df_rev_rest.stars)

def storeImagePlotInDB(plot, db_to_store, name):
    #tmp local image, to be delete
    path_to_store = "./"+name+".png"
    #print(path_to_store)
    plot.savefig(path_to_store)

    #open the file
    datafile = open(path_to_store, "rb")
    thedata = datafile.read()
    
    # create a new gridfs object.
    fs = gridfs.GridFS(db_to_store)

    # store the data in the database
    fs.put(thedata, filename=name)

def loadReviews(datalimit):
    if datalimit is 0:
        df = pa.DataFrame(list(db['reviews'].find({}, {'text': 1, "_id": 0,"stars": 1})))
    else:
        #df = pa.DataFrame(list(db['reviews'].find({}, {'text': 1, "_id": 0, "stars": 1}).limit(datalimit)))
        df_rev = pa.DataFrame(list(db['reviews'].find({}, {"_id": 0,'text': 1, "stars": 1, "business_id": 1, "user_id": 1}).limit(datalimit)))
        df_rest = pa.DataFrame(list(db['restaurants'].find({}, {"_id": 0, "categories": 1, "business_id": 1,
                                                                "neighborhood": 1, "review_count": 1})))
        df= pa.merge(df_rev, df_rest, on='business_id', how='inner')
        #print(df.head())
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

def starCount():
    #for i in range(6):
       # starcount=db['reviews'].find({"stars": i}, {'text':1,"_id":0,"stars":1 }).count()
        #print("Number of reviews with "+ str(i)+" stars")
       # print(starcount)

    df_rev = pa.DataFrame(list(db['reviews'].find({}, {"_id":0,"stars":1,"business_id":1,"user_id":1,"date":1})))
    df_rest= pa.DataFrame(list(db['restaurants'].find({}, { "_id": 0, "categories":1,"business_id":1,"neighborhood":1,"review_count":1,"stars":1})))
    df_user=pa.DataFrame(list(db['users'].find({}, { "_id": 0, "user_id":1,"review_count":1})))
    df_rev_rest=pa.merge(df_rev, df_rest, on='business_id', how='inner')
    #print(df_rev_rest)
    df_rev_users = pa.merge(df_rev, df_user, on='user_id', how='inner')
    print(df_rev_users)

    df_rest['newstar'] = df_rest['stars']
    df_rest.loc[df_rest['stars'] ==1.5, 'newstar'] = 1
    df_rest.loc[df_rest['stars'] ==2.5, 'newstar'] = 2
    df_rest.loc[df_rest['stars'] == 3.5, 'newstar'] = 3
    df_rest.loc[df_rest['stars'] == 4.5, 'newstar'] = 4

    #plot stars numbers
    #plt.hist(df_rest.review_count)
    #plt.show()
    #g = sns.FacetGrid(data=df_rest, col='newstar')
    #g.map(plt.hist, 'review_count', bins=10)
   # plt.show()

    # plots per star restaurants vs review_count
    #g = sns.FacetGrid(data=df_rest, col='newstar')
    #g.map(plt.hist, 'review_count', bins=10)
    #plt.show()

    #Top neighbourhood
    f, ax1 = plt.subplots(1, figsize=(14, 8))
    #print('Number of neighbourhood listed', restaurants['neighborhood'].nunique())
    cnt = df_rev_rest['neighborhood'].value_counts()[:16].to_frame()
    print(cnt)
    sns.barplot(cnt['neighborhood'], cnt.index, palette='RdBu', ax=ax1)
    ax1.set_xlabel('')
    ax1.set_title('Top neighborhood restaurants listed in Yelp')
    # storing image into the database for later use
    storeImagePlotInDB(plt, db_diagrams, "first")
    plt.show()



    df_rest['newneighborhood'] = df_rest['neighborhood']
    df_rest.loc[df_rest['neighborhood'] =="The Strip", 'newneighborhood'] = 1
    df_rest.loc[df_rest['neighborhood'] =="Westside", 'newneighborhood'] = 2
    df_rest.loc[df_rest['neighborhood'] =="Southeast", 'newneighborhood'] = 3
    df_rest.loc[df_rest['neighborhood'] == "Spring Valley", 'newneighborhood'] = 4
    df_rest.loc[df_rest['neighborhood'] == "Downtown", 'newneighborhood'] = 5
    df_rest.loc[df_rest['neighborhood'] == "Chinatown", 'newneighborhood'] = 6
    df_rest.loc[df_rest['neighborhood'] == "Southwest", 'newneighborhood'] = 7
    df_rest.loc[df_rest['neighborhood'] == "Northwest", 'newneighborhood'] = 8
    df_rest.loc[df_rest['neighborhood'] == "South Summerlin", 'newneighborhood'] = 9
    df_rest.loc[df_rest['neighborhood'] == "Summerlin", 'newneighborhood'] =10
    df_rest.loc[df_rest['neighborhood'] == "University", 'newneighborhood'] = 11
    df_rest.loc[df_rest['neighborhood'] == "Sunrise", 'newneighborhood'] = 12
    df_rest.loc[df_rest['neighborhood'] == "The Lakes", 'newneighborhood'] = 13
    df_rest.loc[df_rest['neighborhood'] == "Centennial", 'newneighborhood'] = 14
    df_rest.loc[df_rest['neighborhood'] == "Eastside" ,'newneighborhood'] = 15
    df_rest.loc[df_rest['neighborhood'] == "Anthem", 'newneighborhood'] = 16
    df_rest.loc[df_rest['neighborhood'] == "", 'newneighborhood'] = 0

    print(df_rest.neighborhood.unique())
    #df_rev_rest = pa.merge(df_rev, df_rest, on='business_id', how='inner')
    #ax2=sns.heatmap(data=df_rev_rest.corr(), annot=True)
    #plt.show()

    #df_rev["date"] = df_rev["date"].str.replace('\d+]', '')
    test_cases=df_rev["date"].values
    months=[]
    years=[]
    for date_string in test_cases:
        tempM=parse(date_string).strftime("%m")
        months.append(tempM)
        years=parse(date_string).strftime("%Y")
        #print(years+"-"+months)
    print(months)

    df_rev["months"]=list(months)
    print(df_rev.months.unique())
    df_rev['newmonths'] = df_rev['months']
    df_rev.loc[df_rev['months'] == "01", 'newmonth'] = 'January'
    df_rev.loc[df_rev['months'] == "02", 'newmonth'] = 'February'
    df_rev.loc[df_rev['months'] == "03", 'newmonth'] = 'March'
    df_rev.loc[df_rev['months'] == "04", 'newmonth'] = 'April'
    df_rev.loc[df_rev['months'] == "05", 'newmonth'] = 'May'
    df_rev.loc[df_rev['months'] == "06", 'newmonth'] = 'June'
    df_rev.loc[df_rev['months'] == "07", 'newmonth'] = 'July'
    df_rev.loc[df_rev['months'] == "08", 'newmonth'] = 'August'
    df_rev.loc[df_rev['months'] == "09", 'newmonth'] = 'September'
    df_rev.loc[df_rev['months'] == "10", 'newmonth'] = 'Octomber'
    df_rev.loc[df_rev['months'] == "11", 'newmonth'] = 'November'
    df_rev.loc[df_rev['months'] == "12", 'newmonth'] = 'December'

    f, ax2 = plt.subplots(1, figsize=(14, 8))
    #print('Number of neighbourhood listed', restaurants['neighborhood'].nunique())
    cnt = df_rev['newmonth'].value_counts()[:16].to_frame()
    print(cnt)
    sns.barplot(cnt['newmonth'], cnt.index, palette='RdBu', ax=ax2)
    ax2.set_xlabel('')
    ax2.set_title('Top neighborhood restaurants listed in Yelp')
    #plt.show()

    df_rev.loc[df_rev['months'] == "01", 'newmonth'] = 1
    df_rev.loc[df_rev['months'] == "02", 'newmonth'] = 2
    df_rev.loc[df_rev['months'] == "03", 'newmonth'] = 3
    df_rev.loc[df_rev['months'] == "04", 'newmonth'] = 4
    df_rev.loc[df_rev['months'] == "05", 'newmonth'] = 5
    df_rev.loc[df_rev['months'] == "06", 'newmonth'] = 6
    df_rev.loc[df_rev['months'] == "07", 'newmonth'] = 7
    df_rev.loc[df_rev['months'] == "08", 'newmonth'] = 8
    df_rev.loc[df_rev['months'] == "09", 'newmonth'] = 9
    df_rev.loc[df_rev['months'] == "10", 'newmonth'] = 10
    df_rev.loc[df_rev['months'] == "11", 'newmonth'] = 11
    df_rev.loc[df_rev['months'] == "12", 'newmonth'] = 12

    print(df_rev['newmonth'])
    g = sns.FacetGrid(data=df_rev, col='stars')
    g.map(plt.hist,'newmonth', bins=12)

    print ("Listing " + str(plt.get_figlabels()))
    # storing image into the database for later use
    storeImagePlotInDB(plt, db_diagrams, "second")
    plt.show()