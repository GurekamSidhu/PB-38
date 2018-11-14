import pickle as pkl
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import cm

random.seed()

with open('../data/insurance.p','rb') as file:
    samples = pkl.load(file)

training_epochs = 5000
vlength = len(samples[0])-1

random.shuffle(samples)
features = [row[0:vlength] for row in samples]
labels = [row[vlength] for row in samples]

labels = np.array(labels).astype(int)

avg_label = np.mean(labels)

def run_regress(hidden_layers=(8),graph=False):
    perceptron = MLPRegressor(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=hidden_layers,
                    random_state=1)

    
    perceptron.fit(features,labels)
    score = perceptron.score(features,labels)

    predictions = perceptron.predict(features)
    rmse = np.sqrt(mean_squared_error(predictions,labels))
    print("Score: ",score)
    print("RMSE: ", rmse)
    if(graph):
        fig, ax = plt.subplots()
        ax.scatter(labels, predictions)
        ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()
    return [score,rmse]


def test_multilayer_sizes():    
    scores = []
    rmses = []
    x1 = np.arange(2,10)
    x2 = np.arange(2,10)
    rmses = np.arange(8*8).reshape(8,8)

    for i in x1:
        for j in x2:
            score,rmse = run_regress((i,j),False)
            rmses[i-2][j-2] = rmse
            
    
    rmses = np.asarray(rmses)   
    plt.contour(x1,x2,rmses)
    plt.title("RMS error")
    plt.show()

def test_layer_sizes():
    x = np.arange(2,15)
    rmses = np.arange(13)
    scores = np.arange(13)

    for i in x:
        score,rmse = run_regress((i), False)
        rmses[i-2] = rmse
        scores[i-2] = score
    
    fig, ax1 = plt.subplots()
    plt.title("RMS error")
    plt.plot(x,rmses)
    plt.show()

#test_multilayer_sizes();
#test_layer_sizes();
run_regress((8,5))
