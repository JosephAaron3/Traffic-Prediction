import tensorflow as tf
import glob
import pandas as pd
import matplotlib.pyplot as plt
from models import models
from utils import step_decay_schedule
from load_data import get_vectorized_data

#Parameters
dependent_var = 'flow'
pretime = 12 #How many timesteps in the past to predict from
posttime = 3 #How many timesteps in the future to predict
model_type = 'CNN_LSTM'

#Load, transform and split data
X_train, y_train, X_val, y_val, X_test, y_test = get_vectorized_data(model_type, pretime, posttime, dependent_var, lim = 0.5)
df_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
df_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
df_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

#Create model
#model = models.model_selector(model_type)
model = models.CNN_LSTM((27,3))
model.summary()

#Compile and train model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2), loss = 'mse', metrics = ['mae'])
#lr_sched = step_decay_schedule(initial_lr=1e-2, decay_factor=0.9, step_size=2)
history = model.fit(df_train.batch(64), epochs = 50, validation_data = df_val.batch(64))

#Evaluate model
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Accuracy vs Epoch')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
print("Model performance:", model.evaluate(df_test.batch(32), verbose=2))