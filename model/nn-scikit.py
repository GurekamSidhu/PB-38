import pickle as pkl
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

random.seed()

with open('../data/insurance.p','rb') as file:
    samples = pkl.load(file)

training_epochs = 5000

random.shuffle(samples)
features = [row[0:6] for row in samples]
labels = [row[6] for row in samples]

labels = np.array(labels).astype(int)


perceptron = MLPRegressor(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(5,6),random_state=1)

perceptron.fit(features,labels)

print(perceptron.score(features,labels))

predictions = perceptron.predict(features)

fig, ax = plt.subplots()
ax.scatter(labels, predictions)
ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
