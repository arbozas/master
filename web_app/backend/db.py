import pymongo

client = pymongo.MongoClient('localhost', 27017)
db = client['db']

#total number of instances in a collection
def find_count(collection_name):
    return db[collection_name].find({}).count()