import bson
import json
import os

home = os.getenv("HOME")
# top_level = '/dynprice'
top_level = '/capstone/PB-38'

with open(home + '/dump/porton/receipts.bson','rb') as receiptsfile:
    receipts = bson.decode_all(receiptsfile.read())

with open(home + top_level + '/receipts_schema.json','r') as schemafile:
    schema = json.loads(schemafile.read())

dicts = {}

for datatype in schema:
    if(schema[datatype] == 'class'):
        dicts[datatype] = {}

for row in receipts:
    for datatype in schema:
        if(schema[datatype] == 'class'):
            if(row[datatype] not in dicts[datatype] and row[datatype] is not None):
                dicts[datatype][row[datatype]] = len(dicts[datatype])

jsontxt = json.dumps(dicts)

with open(home + top_level + '/bin/receipts_dict.json', 'w') as file:
    file.write(jsontxt)
