import tensorflow as tf
import glob
import pandas as pd
import matplotlib.pyplot as plt
from models import models
import numpy as np
from load_data import get_vectorized_data, get_vectorized_data_one
from tensorflow.keras.callbacks import LearningRateScheduler

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

#X_train, y_train, X_val, y_val, X_test, y_test = get_vectorized_data(data_filenames[0], pretime, posttime, dependent_var)
X_train, y_train, X_val, y_val, X_test, y_test = get_vectorized_data_one(data_filenames[0], pretime, posttime, dependent_var)
df_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
df_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
df_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

#Create model
#model = models.MLP((27,12,3))
#model = models.CNN((27,12,3))
model = models.LSTM((12,3))
model.summary()

#Compile model
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

def step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=10):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return LearningRateScheduler(schedule)
lr_sched = step_decay_schedule(initial_lr=1e-2, decay_factor=0.9, step_size=3)

#Train model
history = model.fit(df_train.batch(128), epochs = 50, callbacks=[lr_sched], validation_data = df_val.batch(128))

#Evaluate model
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Accuracy vs Epoch')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
print("Model performance:", model.evaluate(df_test.batch(64), verbose=2))