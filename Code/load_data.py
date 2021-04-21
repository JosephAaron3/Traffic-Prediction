import pandas as pd

measurement_index = {'flow': 0, 'density': 1, 'speed': 2}

def get_sample(vector, currenttime, pretime, posttime, measurement):
    X = vector[:, currenttime-pretime+1:currenttime+1, :]
    y = vector[:, currenttime+posttime, measurement]
    return X, y

def get_vectorized_data(filename, pre, post, measurement, split = (0.8, 0.9), lim = None):
    X_train, y_train, X_val, y_val, X_test, y_test = ([] for i in range(6))
    measurement = measurement_index[measurement]
    
    df = pd.read_csv(filename, index_col = (0, 1))
    vector_df = df.values.reshape(len(df.index.levels[0]), len(df.index.levels[1]), -1).swapaxes(0, 1)
    #n = vector_df.shape[1]-pre-post #Number of samples in dataset
    
    for t in range(pre-1, vector_df.shape[1]-post):
        x,y = get_sample(vector_df, t, pre, post, measurement)
        if t < split[0]*vector_df.shape[1]:
            X_train.append(x)
            y_train.append(y)
        elif t < split[1]*vector_df.shape[1]:
            X_val.append(x)
            y_val.append(y)
        else:
            X_test.append(x)
            y_test.append(y)
            
    return X_train, y_train, X_val, y_val, X_test, y_test