#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

medical_dataframe = pd.read_csv("../data/insurance.csv", sep=",")

medical_dataframe = medical_dataframe.reindex(
    np.random.permutation(medical_dataframe.index))

replace_regions = {"region": {"southwest" : 1000, "southeast" : 2000, "northeast" : 3000, "northwest" : 4000},
                   "sex": {"male" : 0, "female" : 1},
                   "smoker" : {"yes" : 0, "no" : 1}}
medical_dataframe.replace(replace_regions, inplace=True)

def preprocess_features(medical_dataframe):
    # sex, smoker, region are excluded
  selected_features = medical_dataframe[
    ["age",
     "bmi",
     "children",
     "sex",
     "smoker",
     "region"
    ]]
  processed_features = selected_features.copy()
  return processed_features

 
def preprocess_targets(medical_dataframe):
  """Prepares target features (i.e., labels) from data set."""

  output_targets = pd.DataFrame()
  output_targets["charges"] = (
    medical_dataframe["charges"]).astype(float)
  return output_targets

test_examples = preprocess_features(medical_dataframe.head(1338))
test_targets = preprocess_targets(medical_dataframe.head(1338))
# Choose the first 939 (out of 1339) examples for training.
training_examples = preprocess_features(medical_dataframe.head(939))
training_targets = preprocess_targets(medical_dataframe.head(939))

# Choose the last 400 (out of 1339) examples for validation.
validation_examples = preprocess_features(medical_dataframe.tail(399))
validation_targets = preprocess_targets(medical_dataframe.tail(399))

# Double-check that we've done the right thing.
# print("Training examples summary:")
# display.display(training_examples.describe())
# print("Validation examples summary:")
# display.display(validation_examples.describe())

# print("Training targets summary:")
# display.display(training_targets.describe())
# print("Validation targets summary:")
# display.display(validation_targets.describe())

# display.display(validation_examples)

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_linear_regressor_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model.
    A `LinearRegressor` object trained on the training data.
  """

  periods = 100
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
    
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["charges"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["charges"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["charges"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")
  
  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  return linear_regressor

linear_regressor = train_linear_regressor_model(
    learning_rate=0.05,
    steps=100000,
    batch_size=50,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

predict_test_input_fn = lambda: my_input_fn(
      test_examples, 
      test_targets, 
      num_epochs=1, 
      shuffle=False)

test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)

test_predictions = np.array([item['predictions'][0] for item in test_predictions])

mean_squared_error = metrics.mean_squared_error(test_predictions, test_targets)
root_mean_squared_error = math.sqrt(
    mean_squared_error)
print("Mean square error: %0.3f" % mean_squared_error)

min_charge_value = medical_dataframe["charges"].min()
max_charge_value = medical_dataframe["charges"].max()
min_max_difference = max_charge_value - min_charge_value

print("Min. charges: %0.3f" % min_charge_value)
print("Max. charges: %0.3f" % max_charge_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

# Run the model in prediction mode.
input_dict = {
  "age": np.array([19]),
  "bmi": np.array([27.9]),
  "children": np.array([0]),
  "smoker" : np.array([0]),
  "sex" : np.array([1]),
    "region" :np.array([1000])
}
predict_input_fns = tf.estimator.inputs.numpy_input_fn(
    input_dict, shuffle=False)
predict_results = linear_regressor.predict(input_fn=predict_input_fns)

# Print the prediction results.
print("\nPrediction results:")
for i, prediction in enumerate(predict_results):
 msg = ("age: {: 4d} , "
        "bmi: {: 9.2f}, "
        "children: {: 2d}, "
        "smoker: {: 2d}, "
        "sex: {: 2d}, "
        "region: {: 2d}, "
        "Charge: ${: 9.2f}, ")
 msg = msg.format(input_dict["age"][i], input_dict["bmi"][i], input_dict["children"][i], input_dict["smoker"][i], input_dict["sex"][i], input_dict["region"][i],
                prediction["predictions"][0])

 print("    " + msg)
print()


# In[ ]:




