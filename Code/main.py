import tensorflow as tf
import glob
import pandas as pd
import matplotlib.pyplot as plt

#Load data files
df = pd.read_csv('../Data/CaliPeMS/I5-N-3/2015.csv')

#Split
split_index = int(0.8*len(df))
val_split_index = int(0.8*split_index)
features = ['timestep', 'road_section', 'density', 'speed']
target = 'flow'

df_train = tf.data.Dataset.from_tensor_slices((tf.cast(df.loc[:val_split_index, features].values, tf.int32),
                                              tf.cast(df.loc[:val_split_index, target].values, tf.int32)))

df_validate = tf.data.Dataset.from_tensor_slices((tf.cast(df.loc[val_split_index:split_index, features].values, tf.int32),
                                              tf.cast(df.loc[val_split_index:split_index, target].values, tf.int32)))

df_test = tf.data.Dataset.from_tensor_slices((tf.cast(df.loc[split_index:, features].values, tf.int32),
                                              tf.cast(df.loc[split_index:, target].values, tf.int32)))

df_train = df_train.shuffle(split_index)
df_validate = df_validate.shuffle(split_index-val_split_index)
df_test = df_test.shuffle(len(df)-split_index)


#Create model
input_layer = tf.keras.layers.Input(shape=(4))
l1 = tf.keras.layers.Dense(512)(input_layer)
l1a = tf.keras.layers.LeakyReLU(alpha=0.001)(l1)
l2 = tf.keras.layers.Dense(512)(l1a)
l2a = tf.keras.layers.LeakyReLU(alpha=0.001)(l2)
output_layer = tf.keras.layers.Dense(1)(l2a)
model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
model.summary()

#Compile model
model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['mae'])

#Train model
history = model.fit(df_train.batch(128), epochs = 2, validation_data = df_validate.batch(128))

#Evaluate model
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Accuracy vs Epoch')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
model.evaluate(df_test.batch(16), verbose=2)

# from tqdm import tqdm
# #Calculate prediction score
# scores = []
# data, label = next(iter(df_test.batch(567000)))
# pred = model.predict(data).squeeze()
# scores = ((pred-label)**2).numpy()
# print("Mean squared error:", sum(scores)/len(scores))