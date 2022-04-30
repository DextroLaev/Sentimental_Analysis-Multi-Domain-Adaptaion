import numpy as np
import tensorflow as tf
from tqdm import tqdm

def lenet5(input_shape=(64,64,1)):
	inputs = tf.keras.Input(shape=input_shape)
	conv1 = tf.keras.layers.Conv2D(filters=16,kernel_size=5,activation='tanh',padding='same')(inputs)
	maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=None,padding='valid')(conv1)
	conv3 = tf.keras.layers.Conv2D(filters=32,kernel_size=5,activation='tanh',padding='same')(maxpool1)
	maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=None,padding='valid')(conv3)

	flatten = tf.keras.layers.Flatten()(maxpool2)

	hd1 = tf.keras.layers.Dense(240,activation='tanh')(flatten)
	hd2 = tf.keras.layers.Dense(120,activation='tanh')(hd1)
	final = tf.keras.layers.Dense(1,activation='sigmoid')(hd2)

	model = tf.keras.Model(inputs=inputs,outputs=final)
	return model