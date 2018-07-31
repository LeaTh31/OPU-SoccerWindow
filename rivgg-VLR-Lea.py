from scipy import misc
import glob
import numpy as np
import time
import tensorflow as tf
import cv2
import csv
from array import array
import shutil
import os
import socket

# To start : 
# python2.7 rivgg-VLR-Lea.py 


# Situation score : A positive score when left team will score the next goal and vice versa

MODE = 1
DROP_RATE = 0.15
UNITS = 4096
POO = 0
INIT_FILTERS = 32
BATCH_SIZE = 16

##################### 1 - We restore the model and the graph

# Directory where we have :
## checkpoint file (keeps a record of latest checkpoint files saved)
## model.ckpt.data-00000-of-00001 file (binary file which contains all the values of the weights, biases, gradients and all the other variables saved. function with model.ckpt.index)
## model.ckpt.meta file (protocol buffer which saves the complete Tensorflow graph; i.e. all variables, operations, collections etc.)
## model.ckpt.index file (binary file which contains all the values of the weights, biases, gradients and all the other variables saved. function with model.ckpt.data-00000-of-00001) 

# When we launch the program, it outputs .IZO files
model_directory = "/home/lea/Documents/Python/PythonSituationScore/TanguyModel/"

# We restore the graph

graph = tf.Graph()
tf.reset_default_graph()  

imported_meta = tf.train.import_meta_graph(model_directory + "model.ckpt.meta")  

graph = tf.get_default_graph()

with graph.as_default():
    imgs = tf.placeholder(tf.float32, shape=[None,160,256,3])
    Training = tf.placeholder(tf.bool)
    x_images = tf.reshape(imgs, [-1,160,256,3])

norm_images = tf.map_fn(lambda x_image: tf.image.per_image_standardization(x_image), x_images)

conv1 = tf.layers.conv2d(
      inputs=norm_images,
      filters=INIT_FILTERS,
      kernel_size=3,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      padding="same",
      activation=tf.nn.relu)

conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=INIT_FILTERS,
      kernel_size=3,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      padding="same",
      activation=tf.nn.relu)

# 80*128*INIT_FILTERS
pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
drop1 = tf.layers.dropout(inputs=pool1, rate=DROP_RATE, training=Training)

# 80*128*(2*INIT_FILTERS)
conv3 = tf.layers.conv2d(
      inputs=drop1,
      filters=2*INIT_FILTERS,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=2*INIT_FILTERS,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

# 40*64*(2*INIT_FILTERS)
pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
drop2 = tf.layers.dropout(inputs=pool2, rate=DROP_RATE, training=Training)

# 40*64*(2*INIT_FILTERS)
conv5 = tf.layers.conv2d(
      inputs=drop2,
      filters=4*INIT_FILTERS,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

conv6 = tf.layers.conv2d(
      inputs=conv5,
      filters=4*INIT_FILTERS,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

conv7 = tf.layers.conv2d(
      inputs=conv6,
      filters=4*INIT_FILTERS,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

# 20*32*(2*INIT_FILTERS)
pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)
drop3 = tf.layers.dropout(inputs=pool3, rate=DROP_RATE, training=Training)

# 20*32*(2*INIT_FILTERS)
conv8 = tf.layers.conv2d(
      inputs=drop3,
      filters=8*INIT_FILTERS,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

conv9 = tf.layers.conv2d(
      inputs=conv8,
      filters=8*INIT_FILTERS,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

conv10 = tf.layers.conv2d(
      inputs=conv9,
      filters=8*INIT_FILTERS,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

# 10*16*(2*INIT_FILTERS)
pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)
drop4 = tf.layers.dropout(inputs=pool4, rate=DROP_RATE, training=Training)

# 5*8*(2*INIT_FILTERS)
conv11 = tf.layers.conv2d(
      inputs=drop4,
      filters=8*INIT_FILTERS,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

conv12 = tf.layers.conv2d(
      inputs=conv11,
      filters=8*INIT_FILTERS,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

conv13 = tf.layers.conv2d(
      inputs=conv12,
      filters=8*INIT_FILTERS,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)


# 5*8*(2*INIT_FILTERS)
pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)
drop5 = tf.layers.dropout(inputs=pool5, rate=DROP_RATE, training=Training)

pool5_flat = tf.reshape(drop5,[-1,5*8*(8*INIT_FILTERS)])

dense = tf.layers.dense(inputs=pool5_flat, units=UNITS, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=DROP_RATE, training=Training)

dense2 = tf.layers.dense(inputs=dropout, units=UNITS, activation=tf.nn.relu)
dropout2 = tf.layers.dropout(inputs=dense2, rate=DROP_RATE, training=Training)

dense3 = tf.layers.dense(inputs=dropout2, units=UNITS, activation=tf.nn.relu)
dropout3 = tf.layers.dropout(inputs=dense3, rate=DROP_RATE, training=Training)


logits = tf.layers.dense(inputs=dropout3, units=MODE)
# logits : Prediction code = output

# Lea Eisti 2018
##################### 2 - We start the server

"""We create the socket : 
socket = socket.socket(family, type);
    socket : socket descriptor, an integer
    family : integer, communication domain (AF_INET : IPv4 addresses)
    type : communication type (SOCK_STREAM : connection-based service, TCP socket)
return -1 if fail. """
socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.bind(('', 15555))

# We load the values of variables from the graph
with tf.Session(graph=graph) as sess:  

    imported_meta.restore(sess, tf.train.latest_checkpoint(model_directory))

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    while True:

        # socket.listen(queueLimit);
        # Enable a server to accept connection. 
        #   queuelen : integer, number of active participants that can wait for a connection
        #   return 0 if listeningm -1 if error
        socket.listen(5)

        # client, addr = socket.accept()
        #   Accept a connection. Return a tuple. 
        #   client : the new socket used for data-transfer
        #   addr : address bound to the socket on the other end of the connection
        client, address = socket.accept()

        # print " {} connected".format( address )

        # client.recv(bufsize)
        #   Receive data from the socket. The maximum amount of data to be received at once is bufsize
        #   Return the data receive. 
        dirname = client.recv(4096)

        # print dirname

        validation_images = []

        #Image directory
        # dirname = "/home/lea/rcss/soccerwindow2-5.1.1/MatchesFrames/20180720165257-Alice-vs-HELIOS2017"

        for image_name in glob.glob(dirname + "/*"):
            validation_images.append(image_name)

        writer = tf.summary.FileWriter(model_directory, sess.graph)

        L = len(validation_images)

        # print("Nb d'image : " + str(L)) 

        total_logits = [0.0 for i in range(L)]

        # L : images number in the directory
        # In this case, if the program is running normaly, we only have one image in the directory.
        # But if we have delay, we can evaluate the previous images too. 
        for i in range(L):
            batch = []
            # im is the image number one
            im = validation_images[i]
            # we add the reading of the image to batch
            # cv2.imread : read an image
            batch.append(cv2.imread(im))
                

            # logits : output
            # sess.run return the first parameter, here logits
            # imags:batch = data
            batch_logits = sess.run(logits, feed_dict={imgs:batch, Training:False})

            total_logits[i] = batch_logits


        f = open(dirname + "-predictions.csv",'a')

        writer = csv.writer(f, lineterminator='\n')
        for nb_logits in total_logits:
            print(nb_logits)
            writer.writerow([nb_logits])
        f.close()

        # We create a directory to move the image already evaluated
        if not os.path.exists(dirname + "-Done"):
            os.mkdir(dirname + "-Done")

        dest = dirname + "-Done"
        for image_name in glob.glob(dirname + "/*"):
            try:
                shutil.move(image_name,dest)
                # print "Copy"
            except:
                print "Copy error"

# print "Close"
# We close the client and socket
client.close()
stock.close()