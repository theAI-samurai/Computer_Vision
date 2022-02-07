import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

#mnist = keras.datasets.mnist()  # Loading MNIST dataset from Keras
mnist = input_data.read_data_sets('MNIST_data')

tf.reset_default_graph() # resets default global graph
tf.get_default_graph()
print('------global graph resetted--------')

real_images = tf.placeholder(tf.float32, shape=[None, 784])  # size of image 28*28=784
z = tf.placeholder(tf.float32, shape=[None, 100])            # planning to keep batch size = 100

def generator(z, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):  # creates namespace for variable and operator in DEFAULT GRAPH
        hidden1 = tf.layers.dense(inputs=z, units=128, activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=128, activation=tf.nn.leaky_relu)
        output = tf.layers.dense(inputs=hidden2, units=784, activation=tf.nn.tanh)
        return output

def discriminator(X, reuse=None):
    with tf.variable_scope('dis', reuse=reuse):
        hidden1 = tf.layers.dense(inputs=X, units=128, activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=128, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(hidden2, units=1)
        output = tf.sigmoid(logits)
        return output, logits

# defining the loss function
def loss_func(logits_in, labels_in):
    temp = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in)
    return tf.reduce_mean(temp)

G = generator(z)
D_output_real, D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G, reuse=True)

# Discriminator is a Binary Classifier,
# Loss is calculated for (Real X) and (Fake X) and then combined
D_real_loss = loss_func(D_logits_real, tf.ones_like(D_logits_real) * 0.9)
D_fake_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_real))
D_loss = D_real_loss + D_fake_loss

G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))

lr = 0.001  # learning rate
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

D_trainer = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=g_vars)

batch_size = 100  # batch size
epochs = 50  # number of epochs. The higher the better the result
init = tf.global_variables_initializer()

# creating a session to train the networks
samples = []  # generator examples

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        # diviiding MNIST dataset of 55000 images into 550 batches of 100 images each
        batch_num = len(mnist.train.images) // batch_size
        for i in range(batch_num):
            # batch consist of array of 100 images and its label
            # batch[0] :  images
            # batch[1] :  labels
            # 'next_batch' is a function that stores batch info
            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0]
            # batch_labels = batch[1]
            batch_images = batch_images * 2 - 1
            batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
            _ = sess.run(D_trainer, feed_dict={real_images: batch_images, z: batch_z})
            _ = sess.run(G_trainer, feed_dict={z: batch_z})
        print("on epoch{}".format(epoch))
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = sess.run(generator(z, reuse=True), feed_dict={z: sample_z})
        samples.append(gen_sample)

plt.imshow(samples[0].reshape(28, 28))

# result after 499th epoch
plt.imshow(samples[49].reshape(28, 28))
