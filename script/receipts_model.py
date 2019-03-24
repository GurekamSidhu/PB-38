import pickle as pkl
import random
import os
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from dateutil import parser
import bson
import json
import runpy
import os
#import receipts_dict_gen

random.seed()

home = os.getenv("HOME")

with open(home + '/dump/porton/receipts.bson','rb') as datafile:
    data = bson.decode_all(datafile.read())

runpy.run_path('script/receipts_dict_gen.py')

with open('receipts_schema.json','r') as schemafile:
    schema = json.loads(schemafile.read())

with open('bin/receipts_dict.json', 'r') as dictfile:
    dicts = json.loads(dictfile.read())

features = []
labels = []

random.shuffle(data)

for row in data:
    price = (int)((float)(row['price']))        
    labels.append(price)
    row['price'] = price

labels.sort()
q1 = labels[(int)(len(labels)/4)]
q2 = labels[(int)(len(labels)/2)]
q3 = labels[(int)(3*len(labels)/4)]
tlo = q1 - 1.5*(q3-q1)
thi = q3 + 1.5*(q3-q1)


labels = []

for row in data:
    try:
        np.array([row['price']]).astype(float)
    except:
        print(row['price'])
        continue
    try:
        duration = (parser.parse(row['end']) - parser.parse(row['start'])).seconds
    except:
        duration = None
        continue
    if(row['price'] < tlo or row['price'] > thi):
        continue
    
    vector = np.asarray([duration])
    vector = [duration]
    skip = False
    for datatype in schema:
        if(schema[datatype] == 'class'):
            if(row[datatype] == None):
                skip = True
                break
            if(row[datatype] not in dicts[datatype]):
                print(datatype + ' "' + row['specialty'] + '" not mapped')
                exit(1)
            classvector = np.zeros(len(dicts[datatype]), dtype=int)
            classvector[dicts[datatype][row[datatype]]] = 1
            vector = np.concatenate((vector, classvector))
        elif(schema[datatype] == 'number'):
            num = [(float)(row[datatype])]
            vector = np.concatenate((vector, num))
        else:
            print("Unknown feature type " + datatype + ", " + schema[datatype])
            print("Should be 'class' or 'number'")
            exit(1)
    if(skip):
        continue
    
    features.append(vector)
    labels.append(row['price'])

print((str)(len(features)) + " usable training entries")

training_epochs = 50000

labels = np.array(labels).astype(float)

fakedata_file = 'bin/receipts_fake.json'
if(os.path.exists(fakedata_file)):
    with open(fakedata_file, 'r') as fakefile:
        fileobj = json.loads(fakefile.read())
        fakedata = fileobj['features']
        fakeprices = fileobj['labels']
        labels = np.concatenate((labels, fakeprices))
        features = np.concatenate((features, fakedata))
        
    print("Plus " + (str)(len(fakedata)) + " fake entries")

print(len(labels), len(features))

reg = LinearRegression().fit(features,labels)
print(reg.score(features, labels))
predictions = reg.predict(features)
print(mean_squared_error(labels, predictions))

model_file = 'bin/receipts_model.pkl'

with open(model_file, 'wb') as dumpfile:
    pkl.dump(reg, dumpfile)
    print("Created new model at " + model_file)



##fig, ax = plt.subplots()
##ax.scatter(labels, predictions)
##ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
##ax.set_xlabel('Measured')
##ax.set_ylabel('Predicted')
##plt.show()
