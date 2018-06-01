import matplotlib.pyplot as plt
import pandas as pa
import pymongo
import seaborn as sns
from dateutil.parser import parse
import gridfs
import numpy as np
from collections import Counter

client = pymongo.MongoClient('localhost', 27017)
db = client['db']
db_diagrams = client['diagrams']

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


# ------------- Load data in Dataframes
#for figures 1, 2, 6
df_rest_star = pa.DataFrame(list(db['restaurants'].find({}, {"_id": 0, "stars": 1, "business_id":1, "neighborhood":1, 'review_count':1})))
df_rest= pa.DataFrame(list(db['restaurants'].find({}, { "_id": 0,"business_id":1, 'review_count':1})))
#for figure 4, 5  --note: Add limit in order to run the code faster
df_rev= pa.DataFrame(list(db['reviews'].find({}, {"_id":0,"stars":1,"business_id":1, "date":1, "useful":1, "funny":1, "cool":1}).limit(0)))  #limit
#figure 3
df_rev_rest=pa.merge(df_rev, df_rest_star, on='business_id', how='inner')
#figure 8
df_rev_rest_nostar=pa.merge(df_rev, df_rest, on='business_id', how='inner')
#figure 7
dfcat = pa.DataFrame(list(db['restaurants'].find({}, {"_id": 0, "categories": 1})))
#figure 9
df_users = pa.DataFrame(list(db['users'].find({}, {"_id": 0, "review_count": 1, 'average_stars':1})))

# ------------- Figure 1: Distribution of star rating -----------#
print(df_rest_star.stars.describe())
ax = sns.countplot(df_rest_star['stars'])
plt.title('Figure 1a: Distribution of star rating wrt restaurants')
plt.xlabel("Stars")
plt.ylabel("Number of restaurants")
storeImagePlotInDB(plt, db_diagrams, 'Figure 1a_Distribution of star rating')
plt.show()

print(df_rev.stars.describe())
ax = sns.countplot(df_rev['stars'])
plt.title('Figure 1b: Distribution of star rating wrt reviews')
plt.xlabel("Stars")
plt.ylabel("Number of reviews")
storeImagePlotInDB(plt, db_diagrams, 'Figure 1b_Distribution of star rating')
plt.show()

# ------------- Figure 2: Distribution of restaurants per neighborhood -----------#
print(df_rest_star.neighborhood.describe())
print('Number of neighbourhood listed', df_rest_star['neighborhood'].nunique())
cnt = df_rest_star['neighborhood'].value_counts()[:17].to_frame()
sns.barplot(cnt['neighborhood'], cnt.index, palette = 'summer')
plt.xlabel('Number of restaurants')
plt.ylabel("Neighborhoods")
plt.title('Figure 2: Distribution of restaurants per neighborhood')
storeImagePlotInDB(plt, db_diagrams, 'Figure 2_Distribution of restaurants per neighborhood')
plt.show()

# ------------- Figure 3: Distribution of reviews per neighborhood -----------#
print(df_rev_rest.neighborhood.describe())
cnt = df_rev_rest['neighborhood'].value_counts()[:17].to_frame()
print(cnt)
sns.barplot(cnt['neighborhood'], cnt.index, palette='RdBu')
plt.xlabel('Number of reviews')
plt.ylabel("Neighborhoods")
plt.title('Figure 3: Distribution of reviews per neighborhood')
storeImagePlotInDB(plt, db_diagrams, 'Figure 3_Distribution of reviews per neighborhood')
plt.show()


# ------------- Figure 8: Review_count distribution for businesses grouped by star -----------#
axa=sns.heatmap(data=df_rev_rest_nostar.corr(), annot=True, linewidths=.5)
plt.title('Figure 8: Correlation between attributes')
storeImagePlotInDB(plt, db_diagrams, 'Heatmap')
plt.show()


# ------------- Figure 4: Distribution of reviews per month -----------#
test_cases = df_rev["date"].values
months = []
years = []
for date_string in test_cases:
    tempM = parse(date_string).strftime("%m")
    months.append(tempM)
    years = parse(date_string).strftime("%Y")
    # print(years+"-"+months)
print(months)

df_rev["months"] = list(months)
print(df_rev.months.unique())
df_rev['month_name'] = df_rev['months']
df_rev.loc[df_rev['months'] == "01", 'month_name'] = 'January'
df_rev.loc[df_rev['months'] == "02", 'month_name'] = 'February'
df_rev.loc[df_rev['months'] == "03", 'month_name'] = 'March'
df_rev.loc[df_rev['months'] == "04", 'month_name'] = 'April'
df_rev.loc[df_rev['months'] == "05", 'month_name'] = 'May'
df_rev.loc[df_rev['months'] == "06", 'month_name'] = 'June'
df_rev.loc[df_rev['months'] == "07", 'month_name'] = 'July'
df_rev.loc[df_rev['months'] == "08", 'month_name'] = 'August'
df_rev.loc[df_rev['months'] == "09", 'month_name'] = 'September'
df_rev.loc[df_rev['months'] == "10", 'month_name'] = 'October'
df_rev.loc[df_rev['months'] == "11", 'month_name'] = 'November'
df_rev.loc[df_rev['months'] == "12", 'month_name'] = 'December'

print(df_rev.month_name.describe())
cnt = df_rev['month_name'].value_counts()[:12].to_frame()
print(cnt)
sns.countplot(df_rev['month_name'],  palette = 'Blues_d')
plt.title('Figure 4: Distribution of reviews per month')
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.ylabel("Number of reviews")
storeImagePlotInDB(plt, db_diagrams, 'Figure 4_Distribution of reviews per month')
plt.show()


# ------------- Figure 5: Month of review grouped by star -----------#
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


sns.set_style("darkgrid")
g = sns.FacetGrid(data=df_rev, col='stars', margin_titles=True)
g.axes[0,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
g.map(plt.hist, 'newmonth', bins=12)
g.fig.suptitle('Figure 5: Month of review grouped by star')
storeImagePlotInDB(plt, db_diagrams, 'Figure 5_Month of review grouped by star')
plt.show()


# ------------- Figure 6: Review_count distribution for restaurants grouped by star -----------#

# Transform values for average_stars into Integers
df_rest_star['star'] = df_rest_star['stars']
df_rest_star.loc[df_rest_star['stars'] == 1.5, 'star'] = 1
df_rest_star.loc[df_rest_star['stars'] == 2.5, 'star'] = 2
df_rest_star.loc[df_rest_star['stars'] == 3.5, 'star'] = 3
df_rest_star.loc[df_rest_star['stars'] == 4.5, 'star'] = 4

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
g = sns.FacetGrid(data=df_rest_star, col='star')
print(df_rest_star.corr())
g.map(plt.hist, 'review_count', bins=10)
g.fig.suptitle('Figure 6: Review count distribution for restaurants grouped by star')
storeImagePlotInDB(plt, db_diagrams, 'Figure 6_Review count distribution for restaurants grouped by star')
plt.show()

# ------------- Figure 7: Top k categories of restaurants -----------#

dfcat = pa.DataFrame(list(db['restaurants'].find({}, {"_id": 0, "categories": 1})))

#Create a new row for each category
newdf1 = dfcat.apply(lambda x: pa.Series(x['categories']),axis=1).stack().reset_index(level=1, drop=True)
newdf1.name = 'newCategories'
dfcat.drop('categories', axis=1).join(newdf1)
#print(newdf1[0]) # Select first category of each restaurant
newdf1 = newdf1.reset_index()
#print(newdf1)

# Exclude tags of restaurants and food as all data refers to restaurants
newdf1 = newdf1[newdf1.newCategories != 'Restaurants' ]
newdf1 = newdf1[newdf1.newCategories != 'Food' ]

top_cat = 20 # Select top-k categories
cnt = newdf1['newCategories'].value_counts()[:top_cat].to_frame()
print(cnt)
sns.barplot(cnt['newCategories'], cnt.index, palette='tab20')
plt.xlabel('Number of restaurants')
plt.ylabel("Categories", wrap=True)
plt.title('Figure 7 Top 20 categories of restaurants')
storeImagePlotInDB(plt, db_diagrams, 'Figure 7 Top-k categories of restaurants')
plt.show()

# ------------- Figure 9: Total reviews and average stars per user -----------#
print(df_users.review_count.values)
g = plt.scatter(df_users.review_count, df_users.average_stars)
plt.xlabel('Total reviews per user')
plt.ylabel("Average stars per user", wrap=True)
plt.title('Figure 9a: Total reviews and average stars per user')
storeImagePlotInDB(plt, db_diagrams, 'Figure 9a Total reviews and average stars per user')
plt.show()

x = np.log(df_users.review_count)
print(df_users.review_count.values)
g = plt.scatter(x, df_users.average_stars, alpha=0.05, edgecolors='none')
plt.xlabel('Total reviews per user')
plt.ylabel("Average stars per user", wrap=True)
plt.title('Figure 9b: Total reviews and average stars per user (x log)')
storeImagePlotInDB(plt, db_diagrams, 'Figure 9b LOG Total reviews and average stars per user')
plt.show()

#Show the the imballance class
def plot_pie(y):
    target_stats = Counter(y)
    labels = target_stats.keys()
    sizes = target_stats.values()
    explode = tuple([0.1] * len(target_stats))
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True,
           autopct='%1.1f%%')
    ax.axis('equal')
    plt.show()