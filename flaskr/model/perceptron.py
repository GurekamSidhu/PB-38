import pickle as pkl
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import cm

class Perceptron:
    def __init__(self):
        random.seed()

        with open('bin/insurance.p','rb') as file:
            samples = pkl.load(file)

        training_epochs = 5000
        vlength = len(samples[0])-1

        random.shuffle(samples)
        self.features = [row[0:vlength] for row in samples]
        labels = [row[vlength] for row in samples]

        self.labels = np.array(labels).astype(int)

        avg_label = np.mean(labels)

    def run_regress(self, hidden_layers=(8),graph=False):
        self.perceptron = MLPRegressor(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=hidden_layers,
                        random_state=1)

    
        self.perceptron.fit(self.features,self.labels)
        score = self.perceptron.score(self.features,self.labels)

        predictions = self.perceptron.predict(self.features)
        rmse = np.sqrt(mean_squared_error(predictions,self.labels))
        print("Score: ",score)
        print("RMSE: ", rmse)
        if(graph):
            fig, ax = plt.subplots()
            ax.scatter(self.labels, predictions)
            ax.plot([self.labels.min(), self.labels.max()], [self.labels.min(), self.labels.max()], 'k--', lw=3)
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')
            plt.show()
        return [score,rmse]


    def test_multilayer_sizes(self):    
        scores = []
        rmses = []
        x1 = np.arange(2,10)
        x2 = np.arange(2,10)
        rmses = np.arange(8*8).reshape(8,8)

        for i in x1:
            for j in x2:
                score,rmse = self.run_regress((i,j),False)
                rmses[i-2][j-2] = rmse
            
    
        rmses = np.asarray(rmses)   
        plt.contour(x1,x2,rmses)
        plt.title("RMS error")
        plt.show()

    def test_layer_sizes(self):
        x = np.arange(2,15)
        rmses = np.arange(13)
        scores = np.arange(13)

        for i in x:
            score,rmse = self.run_regress((i), False)
            rmses[i-2] = rmse
            scores[i-2] = score
    
        fig, ax1 = plt.subplots()
        plt.title("RMS error")
        plt.plot(x,rmses)
        plt.show()

    def predict_price(self, age, sex, bmi, children, smoker, ne, nw, se, sw):
        input_features = [[age, sex, bmi, children, smoker, ne, nw, se, sw]]
        return self.perceptron.predict(input_features)

    # for testing in console
    def predict_price_console(self):
        age = float(input("Patient age?: "))
        sex = float(input("Patient sex? (1 for male 2 for female): "))
        bmi = float(input("Patient bmi?: "))
        children = float(input("Patient # of children: "))
        # TODO is this correct?
        smoker = float(input("Does the patient smoke? (1 for yes -1 for no): "))
        ne = float(input("Patient in NE? (1 for yes 0 for no): "))
        nw = float(input("Patient in NW? (1 for yes 0 for no): "))
        se = float(input("Patient in SE? (1 for yes 0 for no): "))
        sw = float(input("Patient in SW? (1 for yes 0 for no): "))
        input_features = [[age, sex, bmi, children, smoker, ne, nw, se, sw]]
        output_prediction = self.perceptron.predict(input_features)
        print("Predicted charge: $", output_prediction)


