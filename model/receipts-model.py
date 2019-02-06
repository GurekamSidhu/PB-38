import pickle as pkl
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from dateutil import parser
import bson

random.seed()

with open('../data/receipts.bson','rb') as file:
    data = bson.decode_all(file.read())

features = []
labels = []
start = parser.parse(data[0]['start'])
end = parser.parse(data[0]['end'])
print(data[0])
random.shuffle(data)

specialties = {}
eventTypes = {}
types = {}
for row in data:
    if(row['specialty'] not in specialties):
        specialties[row['specialty']] = len(specialties)
    if(row['eventType'] not in eventTypes):
        eventTypes[row['eventType']] = len(eventTypes)
    if(row['type'] not in types):
        types[row['type']] = len(types)

    price = (int)((float)(row['price']))        
    labels.append(price)
    row['price'] = price

labels.sort()
q1 = labels[(int)(len(labels)/4)]
q2 = labels[(int)(len(labels)/2)]
q3 = labels[(int)(3*len(labels)/4)]
tlo = q1 - 1.5*(q3-q1)
thi = q3 + 1.5*(q3-q1)

print(tlo, thi)

labels = []

print(specialties)

for row in data:
    try:
        np.array([row['price']]).astype(float)
    except:
        print(row['price'])
        continue
    if(row['price'] < tlo or row['price'] > thi):
        continue
    try:
        duration = (parser.parse(row['end']) - parser.parse(row['start'])).seconds
    except:
        duration = None
        continue
    vector = np.asarray([duration])
    speciality = np.zeros(len(specialties), dtype=int)
    speciality[specialties[row['specialty']]] = 1
    typ = np.zeros(len(types), dtype=int)
    typ[types[row['type']]] = 1
    eventType = np.zeros(len(eventTypes), dtype=int)
    eventType[eventTypes[row['eventType']]] = 1
    
    vector = np.concatenate((vector, speciality,eventType,typ))
    
    features.append(vector)
    labels.append(row['price'])

print(len(features))
print(len(labels))

training_epochs = 50000

labels = np.array(labels).astype(float)


##perceptron = MLPRegressor(solver='lbfgs', alpha=1e-5,
##                               hidden_layer_sizes=(5,6),random_state=1)
##
##perceptron.fit(features,labels)
##
##print(perceptron.score(features,labels))
##
##predictions = perceptron.predict(features)
##


reg = LinearRegression().fit(features,labels)
print(reg.score(features, labels))
predictions = reg.predict(features)

fig, ax = plt.subplots()
ax.scatter(labels, predictions)
ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

