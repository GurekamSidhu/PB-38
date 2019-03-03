import bson

with open('../data/receipts.bson','rb') as f:
    data = bson.decode_all(f.read())

print(len(data))

