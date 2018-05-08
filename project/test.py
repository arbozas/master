import pandas as pd
import requests
from xml.etree import ElementTree
import numpy as np
import folium

GetAllCountersUrl = "http://webservices.commuterpage.com/counters.cfc?wsdl&method=GetAllCounters"
xmlfile = open('xml_getallcounters.xml', 'w')
xmldata = requests.get(GetAllCountersUrl)
xmlfile.write(xmldata.text)
xmlfile.close()

xml_data = 'xml_getallcounters.xml'
tree = ElementTree.parse(xml_data)
counter = tree.find('counter')
name = counter.find('name')

id = []
name = []
latitude = []
longitude = []
region = []

for c in tree.findall('counter'):
    id.append(c.attrib['id'])
    name.append(c.find('name').text)
    latitude.append(float(c.find('latitude').text))
    longitude.append(float(c.find('longitude').text))
    region.append(c.find('region/name').text)

df_counters = pd.DataFrame(
    {'ID' : id,
     'Name' : name,
     'latitude' : latitude,
     'longitude' : longitude,
     'region' : region
    })
print(df_counters)
locations = df_counters[['latitude', 'longitude']]
locationlist = locations.values.tolist()
map = folium.Map(location=[38.9, -77.05], zoom_start=12)
for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point], popup=df_counters['Name'][point]).add_to(map)
map.save('./map.html')