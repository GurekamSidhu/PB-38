import pickle as pkl
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from dateutil import parser
import bson
import json
import runpy
import os

random.seed()

home = os.getenv("HOME")

with open(home + '/dump/porton/receipts.bson','rb') as datafile:
    data = bson.decode_all(datafile.read())

runpy.run_path('./receipts_dict_gen.py')

with open('../receipts_schema.json','r') as schemafile:
    schema = json.loads(schemafile.read())

with open('../bin/receipts_dict.json', 'r') as dictfile:
    dicts = json.loads(dictfile.read())

features = []
labels = []

random.shuffle(data)

for row in data:
    price = (int)((float)(row['price']))        
    labels.append(price)
    row['price'] = price

#calculate standard distributions to discard outliers
labels.sort()
q1 = labels[(int)(len(labels)/4)]
q2 = labels[(int)(len(labels)/2)]
q3 = labels[(int)(3*len(labels)/4)]
tlo = q1 - 1.5*(q3-q1) #lower bound on price
thi = q3 + 1.5*(q3-q1) #upper bound on price


labels = []
feature_means = {}
feature_count = {}
for datatype in schema:
    feature_means[datatype] = [0]*len(dicts[datatype])
    feature_count[datatype] = [0]*len(dicts[datatype])

durations = []
prices = []
    
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

    durations.append(duration/60)
    prices.append(row['price'])
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
            feature_means[datatype][dicts[datatype][row[datatype]]] += row['price']
            feature_count[datatype][dicts[datatype][row[datatype]]] += 1
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

reg = LinearRegression().fit(features,labels)
print("Score: ", reg.score(features, labels))
predictions = reg.predict(features)
print("Means squared error: ", mean_squared_error(labels, predictions))

plt.title("Duration")
plt.xlabel("Duration (minutes)")
plt.ylabel("Price")
plt.scatter(durations,prices,marker='o')
plt.show()

for datatype in feature_means:
    feature_means[datatype] = np.divide(feature_means[datatype],feature_count[datatype])

    fig, ax = plt.subplots()

    index = np.arange(len(feature_means[datatype]))
    bar_width = 0.35
    opacity = 0.4

    rects1 = ax.bar(index, feature_count[datatype], bar_width,
                alpha=opacity, color='b',
                label='# of entries')
    
    rects2 = ax.bar(index + bar_width, feature_means[datatype], bar_width,
                alpha=opacity, color='r',
                label='Average price')
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_title(datatype)

    keys = sorted(dicts[datatype].items(), key = 
             lambda kv:(kv[1], kv[0]))
    keys = list(map(lambda x: x[0],keys))
    ax.set_xticklabels(keys)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    fig.tight_layout()
    plt.show()

fig, ax = plt.subplots()
ax.scatter(labels, predictions)
ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.set_title('Learning Model Accuracy')
plt.show()
