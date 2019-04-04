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
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import tkinter as tk 
from tkinter import ttk


with open('../receipts_schema.json','r') as schemafile:
    schema = json.loads(schemafile.read())

def row_complete(row):
    for datatype in schema:
        if(row[datatype] == None):
            return False
    return True

class Grapher(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.init_window()
        self.init_graphs()
        
    def init_window(self): 
        self.master.title("Receipts")
        self.pack(fill=tk.BOTH, expand=1)
        
        model_button = tk.Button(self, width=7, text="Model",command=self.show_model)
        model_button.place(x=0, y=50)

        duration_button = tk.Button(self, width=7, text="Duration",command=self.show_duration)
        duration_button.place(x=0, y=75)

        feature_buttons = [[]]*len(schema)
        i = 0
        for datatype in sorted(schema.keys()):
            feature_buttons[i] = tk.Button(self, width=7, text=datatype)
            feature_buttons[i].bind("<Button-1>",self.show_feature)
            feature_buttons[i].place(x=0, y=100+i*25)
            i+=1
            
    def show_model(self):
        self.top_text = tk.Text(self, width=100)
        self.top_text.destroy()
        self.top_text = tk.Text(self, width=100)
        self.top_text.insert(tk.END, "Accuracy: " + '%.2f'%(100*self.score) + '%')
        self.top_text.insert(tk.END, "\nRMS Error: " + '%.4f'%self.error)
        self.top_text.place(x=60, y=0)
        
        self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.model_fig, self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, pady=(30,0),
                                    fill=tk.Y, padx=(60,0), expand=True)
    
    def show_duration(self):

        z = self.durations_z
        self.top_text.destroy()
        self.top_text = tk.Text(self, width=100)
        self.top_text.insert(tk.END, "Equation: y = " +'%.4f'%z[0]
                             + "x + " + '%.3f'%z[1])
        self.top_text.insert(tk.END, "\nWeight: " + '%.4f'%self.feature_coef['duration'])
        self.top_text.place(x=60, y=0)
        
        self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.duration_fig, self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, pady=(35,0),
                                    fill=tk.Y, padx=(60,0), expand=True)

    def show_feature(self, event):
        datatype = event.widget['text']
        self.top_text.destroy()
        if(schema[datatype] == 'class'):
            self.top_text = tk.Text(self, width=100)
            self.top_text.insert(tk.END, ' ')
        elif(schema[datatype] == 'number'):
            z = self.features_z[datatype]
            self.top_text = tk.Text(self, width=100)
            self.top_text.insert(tk.END, "Equation: y = " +'%.4f'%z[0]
                                 + "x + " + '%.3f'%z[1])
            self.top_text.insert(tk.END, "\nWeight: " + '%.4f'%self.feature_coef[datatype])

        self.top_text.place(x=60, y=0)
        
        self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.feature_figs[datatype], self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, pady=(30,0),
                                    fill=tk.Y, padx=(60,0), expand=True)

    def init_graphs(self):

        random.seed()
        home = os.getenv("HOME")

        with open(home + '/dump/porton/receipts.bson','rb') as datafile:
            data = bson.decode_all(datafile.read())

        #runpy.run_path('../script/receipts_dict_gen.py')

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
        feature_vectors = {}
        self.feature_coef = {}
        feature_coef_pos = {}
        for datatype in schema:
            self.feature_coef[datatype] = 0
            if(schema[datatype] == 'class'):
                feature_means[datatype] = [0]*len(dicts[datatype])
                feature_count[datatype] = [0]*len(dicts[datatype])
                feature_coef_pos[datatype] = (0,0)
            elif(schema[datatype] == 'number'):
                feature_vectors[datatype] = []
                feature_coef_pos[datatype] = 0

        feature_coef_pos['duration'] = 0
        
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

            vector = []
            if(not row_complete(row)):
                continue
            
            for datatype in schema:
                if(schema[datatype] == 'class'):
                    if(row[datatype] not in dicts[datatype]):
                        print(datatype + ' "' + row['specialty'] + '" not mapped')
                        exit(1)
                    feature_means[datatype][dicts[datatype][row[datatype]]] += row['price']
                    feature_count[datatype][dicts[datatype][row[datatype]]] += 1
                    classvector = np.zeros(len(dicts[datatype]), dtype=int)
                    classvector[dicts[datatype][row[datatype]]] = 1
                    
                    feature_coef_pos[datatype] = (1+len(vector), 1+len(vector)+
                                                  classvector.size)
                    
                    
                    vector = np.concatenate((vector, classvector))

                elif(schema[datatype] == 'number'):
                    num = [(float)(row[datatype])]
                    vector = np.concatenate((vector, num))
                    feature_coef_pos[datatype] = vector.size
                    feature_vectors[datatype].append(num[0])
                else:
                    print("Unknown feature type " + datatype + ", " + schema[datatype])
                    print("Should be 'class' or 'number'")
                    exit(1)

            durations.append(duration/60)
            prices.append(row['price'])
            
            vector = np.concatenate(([duration],vector))
            
            features.append(vector)
            labels.append(row['price'])

        print((str)(len(features)) + " usable training entries")

        training_epochs = 50000

        labels = np.array(labels).astype(float)

        reg = LinearRegression().fit(features,labels)
        predictions = reg.predict(features)
        
        self.error = np.sqrt(mean_squared_error(labels, predictions))
        self.score = reg.score(features,labels)

        self.duration_fig = plt.scatter(durations,prices,marker='o')
        self.duration_fig, ax = plt.subplots()
        ax.scatter(durations, prices)
        ax.set_xlabel('Duration (minutes)')
        ax.set_ylabel('Price')
        ax.set_title('Duration')
        self.feature_coef['duration'] = reg.coef_[feature_coef_pos['duration']]

        print('Duration coefficient: ', self.feature_coef['duration'])

        z = np.polyfit(durations, prices, 1)
        self.durations_z = z
        p = np.poly1d(z)
        ax.plot(durations,p(durations),"k--")
        
        self.feature_figs = {}
        self.features_z = {}
        pos = 0
        for datatype in schema:

            if(schema[datatype] == 'class'):
                #get coefficients
                self.feature_coef[datatype] = reg.coef_[feature_coef_pos[datatype][0]:feature_coef_pos[datatype][1]]
                
                feature_means[datatype] = np.divide(feature_means[datatype],feature_count[datatype])

                fig, ax = plt.subplots()

                index = np.arange(len(feature_means[datatype]))*1.25
                bar_width = 0.35
                opacity = 0.4

                rects1 = ax.bar(index, feature_count[datatype], bar_width,
                            alpha=opacity, color='k',
                            label='# of entries')
                
                rects2 = ax.bar(index + bar_width, feature_means[datatype], bar_width,
                            alpha=opacity, color='g',
                            label='Average price')
                
                rects3 = ax.bar(index + bar_width*2, self.feature_coef[datatype], bar_width,
                            alpha=opacity, color='r',
                            label='Coefficient (effect on price)')

                print(datatype + ' coefficient: ', self.feature_coef[datatype])
                
                ax.set_xticks(index + bar_width / 2)
                ax.set_title(datatype)

                keys = sorted(dicts[datatype].items(), key = 
                         lambda kv:(kv[1], kv[0]))
                keys = list(map(lambda x: x[0],keys))
                ax.set_xticklabels(keys)
                ax.legend()
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
                fig.tight_layout()
                self.feature_figs[datatype] = fig
            elif(schema[datatype] == 'number'):
                self.feature_coef[datatype] = reg.coef_[feature_coef_pos[datatype]]
                fig, ax = plt.subplots()

                print(len(feature_vectors[datatype]), len(prices))
                ax.scatter(feature_vectors[datatype], prices)
                z = np.polyfit(feature_vectors[datatype], prices, 1)
                self.features_z[datatype] = z
                p = np.poly1d(z)
                ax.plot(feature_vectors[datatype],p(feature_vectors[datatype]),"k--")
                self.feature_figs[datatype] = fig


        self.model_fig, ax = plt.subplots()
        ax.scatter(labels, predictions)
        ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        ax.set_title('Learning Model Accuracy')

        
        self.canvas = FigureCanvasTkAgg(self.model_fig, self)
        self.show_model()

root = tk.Tk()
root.geometry("600x650")
app = Grapher(root)
root.mainloop()


