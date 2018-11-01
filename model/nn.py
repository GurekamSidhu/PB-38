import pickle as pkl
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed()

with open('../data/insurance.p','rb') as file:
    samples = pkl.load(file)

training_epochs = 5000


random.shuffle(samples)
features = [row[0:6] for row in samples]
labels = [row[6] for row in samples]

train_features = features[0:int(len(features)*2/3)]
train_labels = labels[0:int(len(labels)*2/3)]
test_features = features[int(len(features)*2/3):]
test_labels = labels[int(len(labels)*2/3):]



def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def append_bias_reshape(features,labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    #f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    l = np.reshape(labels,[n_training_samples,1])
    return features, l

feature_columns = [
      tf.feature_column.numeric_column(key="age"),
      tf.feature_column.numeric_column(key="sex"),
      tf.feature_column.numeric_column(key="bmi"),
      tf.feature_column.numeric_column(key="children"),
      tf.feature_column.numeric_column(key="smoker"),
      tf.feature_column.numeric_column(key="region"),
  ]

normalized_features = feature_normalize(train_features)
train_features, train_labels = append_bias_reshape(normalized_features,train_labels)

normalized_features = feature_normalize(test_features)
test_features, test_labels = append_bias_reshape(normalized_features,test_labels)

hidden_units = 6
learning_rate = 0.001

model = tf.estimator.LinearRegressor(feature_columns=feature_columns)
n_dim = train_features.shape[1]
X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])
b1 = tf.Variable(tf.zeros([hidden_units]))
b2 = tf.Variable(tf.zeros([1]))
W1 = tf.Variable(tf.ones([n_dim,hidden_units]))
W = tf.Variable(tf.ones([n_dim,1]))
W2 = tf.Variable(tf.ones([hidden_units,1]))

X2 = tf.add(tf.matmul(X,W1), b1)
X2 = tf.nn.sigmoid(X2)
#y_ = tf.add(tf.matmul(X2,W2), b2)
y_ = tf.add(tf.matmul(X,W), b2)
cost = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
cost_history = np.empty(shape=[1],dtype=float)  

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={X:train_features,Y:train_labels})
    cost_history = np.append(cost_history, \
        sess.run(cost,feed_dict={X:train_features,Y: train_labels})) \

print("Weights:",sess.run(W1))
print(sess.run(W2))
print("Biases:",sess.run(b1))
print(sess.run(b2))

plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

pred_y = sess.run(y_, feed_dict={X: test_features})
mse = tf.reduce_mean(tf.square(pred_y - test_labels))
print("MSE: %.4f" % sess.run(mse))

fig, ax = plt.subplots()
ax.scatter(test_labels, pred_y)
ax.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
