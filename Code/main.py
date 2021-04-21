import tensorflow as tf
import glob
import pandas as pd
import matplotlib.pyplot as plt
import models

#Load data files and transform to multidimensional array
data_filenames = ['../Data/CaliPeMS/I5-N-3/2015.csv',
                  '../Data/CaliPeMS/I5-N-3/2016.csv',
                  '../Data/CaliPeMS/I5-S-3/2015.csv',
                  '../Data/CaliPeMS/I5-S-3/2016.csv',
                  '../Data/CaliPeMS/I5-S-4/2015.csv',
                  '../Data/CaliPeMS/I5-S-4/2016.csv']
df_train = pd.read_csv(data_filenames[0], nrows = 54000, index_col = (0, 1))
df_val = pd.read_csv(data_filenames[1], nrows = 54000, index_col = (0, 1))
df_test = pd.read_csv(data_filenames[2], nrows = 50000, index_col = (0, 1))
vector_df_train = df_train.values.reshape(len(df_train.index.levels[0]), len(df_train.index.levels[1]), -1).swapaxes(0, 1)
vector_df_val = df_val.values.reshape(len(df_val.index.levels[0]), len(df_val.index.levels[1]), -1).swapaxes(0, 1)
vector_df_test = df_test.values.reshape(len(df_test.index.levels[0]), len(df_test.index.levels[1]), -1).swapaxes(0, 1)

#Split (80/10/10 train/validation/test)
#val_split_index = int(0.8*len(df))
#test_split_index = int(0.9*len(df))
#features = ['timestep', 'road_section', 'density', 'speed']
#target = 'flow'
df_train = tf.data.Dataset.from_tensor_slices((vector_df_train[:,:,1:], vector_df_train[:,:,0]))

df_val = tf.data.Dataset.from_tensor_slices((vector_df_val[:,:,1:], vector_df_val[:,:,0]))

df_test = tf.data.Dataset.from_tensor_slices((vector_df_test[:,:,1:], vector_df_test[:,:,0]))

#Pre-processing
# def map_fn(image, label):
#     img.set_shape([2,100,1])
#     lbl.set_shape([256,256,1])
#     return img, lbl

# #Map datasets to pre-processing function
# train_ds = train_ds.map(map_fn)
# validate_ds = validate_ds.map(map_fn)
# test_ds = test_ds.map(map_fn)


#Create model
model = models.basic((2000,2))
model.summary()

#Compile model
model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['mae'])

#Train model
history = model.fit(df_train.batch(1), epochs = 50, validation_data = df_val.batch(1))

#Evaluate model
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Accuracy vs Epoch')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
print("Model performance:", model.evaluate(df_test.batch(1), verbose=2))