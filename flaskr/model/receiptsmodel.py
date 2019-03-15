import pickle as pkl
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from dateutil import parser
import bson

class ReceiptsModel:
    def __init__(self):
        random.seed()

        home = os.getenv("HOME")

        with open(home + '/dump/porton/receipts.bson','rb') as file:
            data = bson.decode_all(file.read())

        features = []
        labels = []
        start = parser.parse(data[0]['start'])
        end = parser.parse(data[0]['end'])
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

        self.speciality_length = len(specialties)
        self.event_type_length = len(eventTypes)
        self.type_length = len(types)

        labels = []

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

        training_epochs = 50000

        labels = np.array(labels).astype(float)

        self.reg = LinearRegression().fit(features,labels)
        print(self.reg.score(features, labels))
        predictions = self.reg.predict(features)

    def predict_price(self, duration, speciality, eventType, typ):
        features = np.zeros((1,23))
        duration_vector = np.asarray([duration])
        speciality_vector = np.zeros(self.speciality_length, dtype=int)
        speciality_vector[speciality] = 1
        event_type_vector = np.zeros(self.event_type_length, dtype=int)
        event_type_vector[eventType] = 1
        type_vector = np.zeros(self.type_length, dtype=int)
        type_vector[typ] = 1
        vector = np.concatenate((duration_vector, speciality_vector, event_type_vector, type_vector))
        features[0] = vector
        return self.reg.predict(features)