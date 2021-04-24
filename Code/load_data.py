import pandas as pd
import numpy as np

data_filenames = ['../Data/CaliPeMS/I5-N-3/2015.csv',
                  '../Data/CaliPeMS/I5-N-3/2016.csv',
                  '../Data/CaliPeMS/I5-S-3/2015.csv',
                  '../Data/CaliPeMS/I5-S-3/2016.csv',
                  '../Data/CaliPeMS/I5-S-4/2015.csv',
                  '../Data/CaliPeMS/I5-S-4/2016.csv'] 

measurement_index = {'flow': 0, 'density': 1, 'speed': 2}
nospacetime = ['LSTM']

def get_sample(model_type, vector, currenttime, pretime, posttime, measurement):
    X = vector[:, currenttime-pretime+1:currenttime+1, :]
    y = vector[:, currenttime+posttime, measurement]
    if model_type == 'CNN_LSTM':
        X = vector[:, currenttime, :]
    return X, y

def get_vectorized_data(model_type, pre = 12, post = 3, measurement = 'flow', split = (0.8, 0.9), lim = None):
    X_train, y_train, X_val, y_val, X_test, y_test = ([] for i in range(6))
    measurement = measurement_index[measurement]
    
    for f in data_filenames[:2]:
        df = pd.read_csv(f, index_col = (0, 1))
        vector_df = df.values.reshape(len(df.index.levels[0]), len(df.index.levels[1]), -1).swapaxes(0, 1)
        #n = vector_df.shape[1]-pre-post #Number of samples in dataset
        
        for t in range(pre-1, int(lim*vector_df.shape[1]-post)):
            x,y = get_sample(model_type, vector_df, t, pre, post, measurement)
            U = np.random.rand()
            if U < split[0]:
                if model_type in nospacetime:
                    X_train.extend(x)
                    y_train.append(y)
                else:
                    X_train.append(x)
                    y_train.append(y)
            elif U < split[1]:
                if model_type in nospacetime:
                    X_val.extend(x)
                    y_val.append(y)
                else:
                    X_val.append(x)
                    y_val.append(y)
            else:
                if model_type in nospacetime:
                    X_test.extend(x)
                    y_test.append(y)
                else:
                    X_test.append(x)
                    y_test.append(y)
            
    return X_train, y_train, X_val, y_val, X_test, y_test