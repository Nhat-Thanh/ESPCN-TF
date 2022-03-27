from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
import tensorflow as tf

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def ESPCNx2():
    # X_in = Input(shape=(None, None, 1))
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=5, 
               kernel_initializer=RandomNormal(mean=0, stddev=0.001), 
               padding='same', activation='tanh')(X_in)
    X = Conv2D(filters=32, kernel_size=3, 
               kernel_initializer=RandomNormal(mean=0, stddev=0.001), 
               padding='same', activation='tanh')(X)
    # X = Conv2D(filters=4,  kernel_size=3, 
    X = Conv2D(filters=12,  kernel_size=3, 
               kernel_initializer=RandomNormal(mean=0, stddev=0.001), 
               padding='same')(X)
    # X = Lambda(pixel_shuffle(scale=2))(X)
    X = Lambda(pixel_shuffle(scale=2))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    return Model(X_in, X_out, name="ESPCNx2")


def ESPCNx3():
    # X_in = Input(shape=(None, None, 1))
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=5, 
               kernel_initializer=RandomNormal(mean=0, stddev=0.001), 
               padding='same', activation='tanh')(X_in)
    X = Conv2D(filters=32, kernel_size=3, 
               kernel_initializer=RandomNormal(mean=0, stddev=0.001), 
               padding='same', activation='tanh')(X)
    # X = Conv2D(filters=9,  kernel_size=3, 
    X = Conv2D(filters=27,  kernel_size=3, 
               kernel_initializer=RandomNormal(mean=0, stddev=0.001), 
               padding='same')(X)
    X = Lambda(pixel_shuffle(scale=3))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    return Model(X_in, X_out, name="ESPCNx3")


def ESPCNx4():
    # X_in = Input(shape=(None, None, 1))
    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=64, kernel_size=5, 
               kernel_initializer=RandomNormal(mean=0, stddev=0.001), 
               padding='same', activation='tanh')(X_in)
    X = Conv2D(filters=32, kernel_size=3, 
               kernel_initializer=RandomNormal(mean=0, stddev=0.001), 
               padding='same', activation='tanh')(X)
    # X = Conv2D(filters=16, kernel_size=3, 
    X = Conv2D(filters=48, kernel_size=3, 
               kernel_initializer=RandomNormal(mean=0, stddev=0.001), 
               padding='same')(X)
    X = Lambda(pixel_shuffle(scale=4))(X)
    X_out = tf.clip_by_value(X, 0.0, 1.0)
    return Model(X_in, X_out, name="ESPCNx4")
