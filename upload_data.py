from pymongo. mongo_client import MongoClient
import pandas as pd
import json

uri = "mongodb+srv://pankaj:MFDi1HC6d90NtZEd@cluster0.okfhp97.mongodb.net/?appName=Cluster0"

client = MongoClient(uri)

#create a database name and collcetion name

DATABASE_NAME = 'pwskills'
COLLECTION_NAME = 'waferfault'

df = pd.read_csv("C:\Users\LENOVO\Downloads\sensor project\notebooks\wafer_23012020_041211.csv")
df.head()

df = df.drop("Unnamed: 0",axis = 1)

json_record = list(json.loads(df.T.to_json()).values())
type(json_record)

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)