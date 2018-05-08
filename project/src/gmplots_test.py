import folium
import pandas as pa
import pymongo

GetAllCountersUrl = "http://webservices.commuterpage.com/counters.cfc?wsdl&method=GetAllCounters"

client = pymongo.MongoClient('localhost', 27017)
db = client['db']
df = pa.DataFrame(list(db['restaurants_coordinates'].find().limit(1000)))
df.drop(df.columns[[0, 1]], axis=1, inplace=True)
print(df)
#Put names to a list
labels = df["name"].values.tolist()
#Put coordinates to a list
locations = df[['latitude', 'longitude']]
locationlist = locations.values.tolist()
#generate map
map = folium.Map(location=[36.0, -115.17], zoom_start=12)
#put markers on top of the map
for point in range(6, len(locationlist)):
    popup = folium.Popup(labels[point], parse_html=True)
    folium.Marker(locationlist[point], popup=popup).add_to(map)

#save map on html
map.save('./map.html')

