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

num_features = len(samples[0])-1
features = [row[0:num_features] for row in samples]
labels = [row[num_features] for row in samples]

train_X = np.asarray(features[0:int(len(features)*2/3)])
train_y = np.asarray(labels[0:int(len(labels)*2/3)])
test_X = np.asarray(features[int(len(features)*2/3):])
test_y = np.asarray(labels[int(len(labels)*2/3):])



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

normalized_features = feature_normalize(train_X)
train_X, train_y = append_bias_reshape(normalized_features,train_y)

normalized_features = feature_normalize(test_X)
test_X, test_y = append_bias_reshape(normalized_features,test_y)

hidden_units = 6
learning_rate = 0.001

#model = tf.estimator.LinearRegressor(feature_columns=feature_columns)
n_dim = train_X.shape[1]
X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])
b1 = tf.Variable(tf.zeros([hidden_units]))
b2 = tf.Variable(tf.zeros([1]))
W1 = tf.Variable(tf.ones([n_dim,hidden_units]))
W = tf.Variable(tf.ones([n_dim,1]))
W2 = tf.Variable(tf.ones([hidden_units,1]))

X2 = tf.add(tf.matmul(X,W1), b1)
X2 = tf.nn.sigmoid(X2)
y_ = tf.add(tf.matmul(X2,W2), b2)
y_ = tf.add(tf.matmul(X,W), b2)
cost = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
cost_history = np.empty(shape=[1],dtype=float)  

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={X:train_X,Y:train_y})
    cost_history = np.append(cost_history, \
        sess.run(cost,feed_dict={X:train_X,Y: train_y})) \

plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

pred_y = sess.run(y_, feed_dict={X: test_X})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("RMSE: %.4f" % np.sqrt(sess.run(mse)))

fig, ax = plt.subplots()
ax.scatter(test_y, pred_y)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
