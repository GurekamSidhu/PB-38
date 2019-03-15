import pickle as pkl
import random
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from dateutil import parser
import bson
import json

class ReceiptsModel:
    def __init__(self):
        random.seed()

        home = os.getenv("HOME")

        with open(home + '/dump/porton/receipts.bson','rb') as datafile:
            data = bson.decode_all(datafile.read())

        with open(home + '/dynprice/bin/receipts-dict.json', 'r') as dictfile:
            obj = json.loads(dictfile.read())
            specialties = obj['specialties']
            eventTypes = obj['eventTypes']
            types = obj['types']

        features = []
        labels = []

        self.speciality_length = len(specialties)
        self.event_type_length = len(eventTypes)
        self.type_length = len(types)

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
            if(row['specialty'] == None):
                continue
            
            vector = np.asarray([duration])
            speciality = np.zeros(self.speciality_length, dtype=int)
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
            speciality[specialties[row['specialty']]] = 1
            typ = np.zeros(self.type_length, dtype=int)
            typ[types[row['type']]] = 1
            eventType = np.zeros(self.event_type_length, dtype=int)
            eventType[eventTypes[row['eventType']]] = 1
            
            vector = np.concatenate((vector, speciality,eventType,typ))
            
            features.append(vector)
            labels.append(row['price'])

        print((str)(len(features)) + " usable training entries")

        training_epochs = 50000

        labels = np.array(labels).astype(float)

        fakedata_file = home + '/capstone/PB-38/bin/receipts-fake.json'
        if(os.path.exists(fakedata_file)):
            with open(fakedata_file, 'r') as fakefile:
                fileobj = json.loads(fakefile.read())
                fakedata = fileobj['features']
                fakeprices = fileobj['labels']
                labels = np.concatenate((labels, fakeprices))
                features = np.concatenate((features, fakedata))
                
            print("Plus " + (str)(len(fakedata)) + " fake entries")

        print(len(labels), len(features))

        self.reg = LinearRegression().fit(features,labels)
        print(self.reg.score(features, labels))
        predictions = self.reg.predict(features)
        print(mean_squared_error(labels, predictions))

        model_file = home + '/capstone/PB-38/bin/receipts-model.pkl'

        with open(model_file, 'wb') as dumpfile:
            pkl.dump(self.reg, dumpfile)
            print("Created new model at " + model_file)

    def predict_price(self, duration, speciality, eventType, typ):
        features = np.zeros((1,22))
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
