import pymongo
import pandas as pa
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import gridfs
client = pymongo.MongoClient('localhost', 27017)
db = client['db']

def find_all_restaurants():
    return db['restaurants'].find({})

def insert_to_db(json_obj, collection_name):
    db[collection_name].insert(json_obj)

#Load reviews
def loadReviews(datalimit):
    if datalimit is 0:
        df = pa.DataFrame(list(db['reviews'].find({}, {'text': 1, "_id": 0,"stars": 1})))
    else:
        df = pa.DataFrame(list(db['reviews'].find({}, {'text': 1, "_id": 0, "stars": 1}).limit(datalimit)))
    return df


def storeImagePlotInDB(plot, db_to_store, name):
    # tmp local image, to be delete
    path_to_store = "./" + name + ".png"
    # print(path_to_store)
    plot.savefig(path_to_store)

    # open the file
    datafile = open(path_to_store, "rb")
    thedata = datafile.read()

    # create a new gridfs object.
    fs = gridfs.GridFS(db_to_store)

    # store the data in the database
    fs.put(thedata, filename=name)