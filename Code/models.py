import tensorflow as tf

def basic(shape):
    input_layer = tf.keras.layers.Input(shape=shape)
    l1 = tf.keras.layers.Dense(2048)(input_layer)
    l1a = tf.keras.layers.LeakyReLU(alpha=0.001)(l1)
    d1 = tf.keras.layers.Dropout(0.6)(l1a)
    l2 = tf.keras.layers.Dense(1024)(d1)
    l2a = tf.keras.layers.LeakyReLU(alpha=0.001)(l2)
    output_layer = tf.keras.layers.Dense(1)(l2a)
    return tf.keras.Model(inputs = input_layer, outputs = output_layer)
    