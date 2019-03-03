import bson
import numpy as np
import json
from dateutil import parser

with open('./receipts.bson','rb') as datafile:
    data = bson.decode_all(datafile.read())

features = []
labels = []

spa = {}

with open('./receipts-dict.json', 'r') as dictfile:
    obj = json.loads(dictfile.read())
    specialties = obj['specialties']
    eventTypes = obj['eventTypes']
    types = obj['types']
    
for row in data:
    price = (int)((float)(row['price']))        
    labels.append(price)
    row['price'] = price
    if(row['specialty'] in spa):
        spa[row['specialty']] += 1
    else:
        spa[row['specialty']] = 1

print(spa)

    
q1 = labels[(int)(len(labels)/4)]
q2 = labels[(int)(len(labels)/2)]
q3 = labels[(int)(3*len(labels)/4)]
tlo = q1 - 1.5*(q3-q1)
thi = q3 + 1.5*(q3-q1)

drv = []
spv = []
evv = []
tpv = []
prv = []

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
    if(row['specialty'] == None):
        continue
    
    vector = np.asarray([duration])
    speciality = np.zeros(len(specialties), dtype=int)
    if(row['specialty'] not in specialties):
        print('Specialty "' + row['specialty'] + '" not mapped')
        print('Run receipts-dict-gen.py or add ' + row['type'] + ' to receipts-dict.json manually')
        exit(1)
    if(row['type'] not in types):
        print('Type "' + row['type'] + '" not mapped')
        print('Run receipts-dict-gen.py or add ' + row['type'] + ' to receipts-dict.json manually')
        exit(1)
    if(row['eventType'] not in eventTypes):
        print('Event type "' + row['type'] + '" not mapped')
        print('Run receipts-dict-gen.py or add ' + row['eventType'] + ' to receipts-dict.json manually')
        exit(1)
    prv.append(row['price'])
    drv.append(duration/60)
    speciality[specialties[row['specialty']]] = 1
    typ = np.zeros(len(types), dtype=int)
    typ[types[row['type']]] = 1
    eventType = np.zeros(len(eventTypes), dtype=int)
    eventType[eventTypes[row['eventType']]] = 1
    
    vector = np.concatenate((vector, speciality,eventType,typ))
    
    features.append(vector)
    labels.append(row['price'])

print(prv)

prv = np.asarray(prv)
drv = np.asarray(drv)

print(len(prv),len(drv))
print(np.mean(prv), np.mean(drv))
print(np.std(prv), np.std(drv))
print(np.min(prv), np.min(drv))
print(np.quantile(prv,0.25), np.quantile(drv,0.25))
print(np.quantile(prv,0.50), np.quantile(drv,0.50))
print(np.quantile(prv,0.75), np.quantile(drv,0.75))
print(np.quantile(prv,1.0), np.quantile(drv,1.0))

##fig, ax = plt.subplots()
##ax.scatter(labels, predictions)
##ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
##ax.set_xlabel('Measured')
##ax.set_ylabel('Predicted')
##plt.show()
