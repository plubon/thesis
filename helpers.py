import tensorflow as tf
import uuid

cnt = 0

def convolve(x, w, strides):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')


def weight_variable_conv(shape):
    return tf.get_variable("W"+str(uuid.uuid4()), shape=shape,
           initializer=tf.contrib.layers.xavier_initializer_conv2d())

def weight_variable_xavier(shape):
    return tf.get_variable("W"+str(uuid.uuid4()), shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.005, shape=shape)
    return tf.Variable(initial)


def max_pool(x, size):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='VALID')


def relu(feats, name=None):
    return tf.nn.relu(feats, name=name)


def dropout(x, probs, name=None):
    return tf.nn.dropout(x, probs, name)
