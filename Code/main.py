import tensorflow as tf
import glob
import pandas as pd
import matplotlib.pyplot as plt
from models import models
from load_data import get_vectorized_data

#Semantic parameters
dependent_var = 'flow'
pretime = 12 #How many timesteps in the past to predict from
posttime = 3 #How many timesteps in the future to predict

#Load, transform and split data
data_filenames = ['../Data/CaliPeMS/I5-N-3/2015.csv',
                  '../Data/CaliPeMS/I5-N-3/2016.csv',
                  '../Data/CaliPeMS/I5-S-3/2015.csv',
                  '../Data/CaliPeMS/I5-S-3/2016.csv',
                  '../Data/CaliPeMS/I5-S-4/2015.csv',
                  '../Data/CaliPeMS/I5-S-4/2016.csv'] #Only using one dataset for now

X_train, y_train, X_val, y_val, X_test, y_test = get_vectorized_data(data_filenames[0], 12, 3, 'flow')
df_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
df_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
df_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

#Create model
#model = models.MLP((27,12,3))
model = models.CNN((27,12,3))
model.summary()

#Compile model
model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['mae'])

#Train model
history = model.fit(df_train.batch(64), epochs = 10, validation_data = df_val.batch(64))

#Evaluate model
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Accuracy vs Epoch')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
print("Model performance:", model.evaluate(df_test.batch(64), verbose=2))