import matplotlib.pyplot as plt
import pandas as pa
import pymongo
import seaborn as sns
from dateutil.parser import parse

client = pymongo.MongoClient('localhost', 27017)
db = client['db']

# ------------- Figure 1: Distribution of star rating -----------#
plt.figure(figsize=(12, 6))
df_rest_star = pa.DataFrame(list(db['restaurants'].find({}, {"_id": 0, "stars": 1})))
ax = sns.countplot(df_rest_star['stars'])
plt.title('Distribution of star rating')
plt.xlabel("Stars")
plt.ylabel("Number of restaurants")
plt.show()


# ------------- Figure 2: Distribution of restaurants per neighborhood -----------#
plt.figure(figsize=(14, 8))
df_rest= pa.DataFrame(list(db['restaurants'].find({}, { "_id": 0,"business_id":1, "neighborhood":1})))
print('Number of neighbourhood listed', df_rest['neighborhood'].nunique())
cnt = df_rest['neighborhood'].value_counts()[:17].to_frame()
sns.barplot(cnt['neighborhood'], cnt.index, palette = 'summer')
plt.xlabel('Number of restaurants')
plt.ylabel("Neighborhoods")
plt.title('Distribution of restaurants per neighborhood')
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
plt.title('Distribution of reviews per neighborhood')
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
sns.countplot(df_rev['month_name'],  palette = 'ocean')
plt.title('Distribution of reviews per month')
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.ylabel("Number of reviews")
plt.show()

