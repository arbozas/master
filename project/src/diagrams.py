import matplotlib.pyplot as plt
import pandas as pa
import pymongo
import seaborn as sns
from dateutil.parser import parse
import gridfs

client = pymongo.MongoClient('localhost', 27017)
db = client['db']
db_diagrams = client['diagrams']

def storeImagePlotInDB(plot, db_to_store, name):
    # tmp local image, to be delete
    path_to_store = "./" + name + ".png"
    # print(path_to_store)
    plot.savefig(path_to_store)

    # open the file
    datafile = open(path_to_store, "rb");
    thedata = datafile.read()

    # create a new gridfs object.
    fs = gridfs.GridFS(db_to_store)

    # store the data in the database
    fs.put(thedata, filename=name)

# ------------- Figure 1: Distribution of star rating -----------#
plt.figure(figsize=(12, 6))
df_rest_star = pa.DataFrame(list(db['restaurants'].find({}, {"_id": 0, "stars": 1})))
ax = sns.countplot(df_rest_star['stars'])
plt.title('Figure 1: Distribution of star rating')
plt.xlabel("Stars")
plt.ylabel("Number of restaurants")
storeImagePlotInDB(plt, db_diagrams, 'Figure 1_Distribution of star rating')
plt.show()

# ------------- Figure 2: Distribution of restaurants per neighborhood -----------#
plt.figure(figsize=(14, 8))
df_rest= pa.DataFrame(list(db['restaurants'].find({}, { "_id": 0,"business_id":1, "neighborhood":1, "stars":1, 'review_count':1})))
print('Number of neighbourhood listed', df_rest['neighborhood'].nunique())
cnt = df_rest['neighborhood'].value_counts()[:17].to_frame()
sns.barplot(cnt['neighborhood'], cnt.index, palette = 'summer')
plt.xlabel('Number of restaurants')
plt.ylabel("Neighborhoods")
plt.title('Figure 2: Distribution of restaurants per neighborhood')
storeImagePlotInDB(plt, db_diagrams, 'Figure 2_Distribution of restaurants per neighborhood')
plt.show()

# ------------- Figure 3: Distribution of reviews per neighborhood -----------#
plt.figure(figsize=(14, 8))
df_rev = pa.DataFrame(list(db['reviews'].find({}, {"_id":0,"stars":1,"business_id":1, "date":1})))
df_rev_rest=pa.merge(df_rev, df_rest, on='business_id', how='inner')
cnt = df_rev_rest['neighborhood'].value_counts()[:17].to_frame()
print(cnt)
sns.barplot(cnt['neighborhood'], cnt.index, palette='RdBu')
plt.xlabel('Number of reviews')
plt.ylabel("Neighborhoods")
plt.title('Figure 3: Distribution of reviews per neighborhood')
storeImagePlotInDB(plt, db_diagrams, 'Figure 3_Distribution of reviews per neighborhood')
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

plt.figure(figsize=(12, 6))
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

print(df_rev['months'])
sns.set_style("darkgrid")
g = sns.FacetGrid(data=df_rev, col='stars', margin_titles=True)
g.axes[0,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
g.map(plt.hist, 'newmonth', bins=12)
g.fig.suptitle('Figure 5: Month of review grouped by star')
storeImagePlotInDB(plt, db_diagrams, 'Figure 5_Month of review grouped by star')
plt.show()


# ------------- Figure 6: Review_count distribution for businesses grouped by star -----------#
df_rest['star'] = df_rest['stars']
df_rest.loc[df_rest['stars'] == 1.5, 'star'] = 1
df_rest.loc[df_rest['stars'] == 2.5, 'star'] = 2
df_rest.loc[df_rest['stars'] == 3.5, 'star'] = 3
df_rest.loc[df_rest['stars'] == 4.5, 'star'] = 4

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
g = sns.FacetGrid(data=df_rest, col='star')
g.map(plt.hist, 'review_count', bins=10)
g.fig.suptitle('Figure 6: Review count distribution for businesses grouped by star')
storeImagePlotInDB(plt, db_diagrams, 'Figure 6_Review count distribution for businesses grouped by star')
plt.show()


# -------------
#print ("Listing " + str(plt.get_figlabels()))
# storing image into the database for later use
#storeImagePlotInDB(plt, db_diagrams, "second")
#plt.show()