import tensorflow as tf

lrelu_alp = 0.001

def MLP(shape):
    input_layer = tf.keras.layers.Input(shape=shape)
    l1 = tf.keras.layers.Dense(2048)(input_layer)
    l1a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(l1)
    d1 = tf.keras.layers.Dropout(0.6)(l1a)
    l2 = tf.keras.layers.Dense(1024)(d1)
    l2a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(l2)
    f1 = tf.keras.layers.Flatten()(l2a)
    output_layer = tf.keras.layers.Dense(27)(f1)
    return tf.keras.Model(inputs = input_layer, outputs = output_layer)


def res_block(prev, n, absfirst = False, first = False):
    if first:
        res = tf.keras.layers.Conv2D(n, (1,1), padding = 'same')(prev)
    else:
        res = prev
    c1 = tf.keras.layers.Conv2D(n, (3,3), padding = 'same')(prev)
    c1n = tf.keras.layers.BatchNormalization()(c1)
    c1a = tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(c1n)
    c2 = tf.keras.layers.Conv2D(n, (3,3), padding = 'same')(c1a)
    c2n = tf.keras.layers.BatchNormalization()(c2)
    if absfirst:
        out = c2n
    else:
        out = c2n + res
    return tf.keras.layers.LeakyReLU(alpha=lrelu_alp)(out)

def CNN(shape):
    input_layer = tf.keras.layers.Input(shape=shape)
    r1 = res_block(input_layer, 32, absfirst = True)
    r2 = res_block(r1, 32)
    r3 = res_block(r2, 32)
    r4 = res_block(r3, 64, first = True)
    r5 = res_block(r4, 64)
    r6 = res_block(r5, 64)
    r7 = res_block(r6, 96, first = True)
    r8 = res_block(r7, 96)
    r9 = res_block(r8, 96)
    l1 = tf.keras.layers.Dense(2048)(r9)
    l1a = tf.keras.layers.LeakyReLU(alpha=0.001)(l1)
    d1 = tf.keras.layers.Dropout(0.6)(l1a)
    l2 = tf.keras.layers.Dense(1024)(d1)
    l2a = tf.keras.layers.LeakyReLU(alpha=0.001)(l2)
    f1 = tf.keras.layers.Flatten()(l2a)
    output_layer = tf.keras.layers.Dense(27)(f1)
    return tf.keras.Model(inputs = input_layer, outputs = output_layer)