import tensorflow as tf
import numpy as np



data = tf.ones([5,5,5])
data_r = tf.reshape(data,[5,25,1])
print(data_r)