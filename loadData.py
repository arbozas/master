import pymongo
import pandas as pa
import matplotlib.pyplot as plt
import seaborn as sns

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
def starCount():
    #for i in range(6):
       # starcount=db['reviews'].find({"stars": i}, {'text':1,"_id":0,"stars":1 }).count()
        #print("Number of reviews with "+ str(i)+" stars")
       # print(starcount)

    df_rev = pa.DataFrame(list(db['reviews'].find({}, {"_id":0,"stars":1,"business_id":1,"user_id":1})))
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
    #f, ax = plt.subplots(1, 2, figsize=(14, 8))
    cnt = df_rev_rest['neighborhood'].value_counts()[:16].to_frame()
    print(cnt)
    sns.barplot(cnt['neighborhood'], cnt.index, palette='RdBu', ax=ax1)
    ax1.set_xlabel('')
    ax1.set_title('Top neighborhood restaurants listed in Yelp')
    #plt.show()

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
    df_rev_rest = pa.merge(df_rev, df_rest, on='business_id', how='inner')
    ax2=sns.heatmap(data=df_rev_rest.corr(), annot=True)
    plt.show()
