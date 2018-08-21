from scipy import misc
import Tkinter as tk
import glob
import numpy as np
import time
import sys
import tensorflow as tf
import cv2
import csv
from array import array
import shutil
import os
import socket
from tensorflow.python.framework import graph_util
from PyQt4.QtGui import QLabel, QApplication
from PyQt4 import QtTest

MODE = 1
DROP_RATE = 0.15
UNITS = 4096
POO = 0
INIT_FILTERS = 32
BATCH_SIZE = 16


# 1 - We restore the model and the graph

model_directory = "./"

graph = tf.Graph()
with graph.as_default():

    with open(model_directory + 'tanguy_model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


    # Lea Eisti 2018

    # 2 - We start the server

    """We create the socket : 
    socket = socket.socket(family, type);
        socket : socket descriptor, an integer
        family : integer, communication domain (AF_INET : IPv4 addresses)
        type : communication type (SOCK_STREAM : connection-based service, TCP socket)
    return -1 if fail. """
    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket.bind(('', 15555))


    # We load the values of variables from the graph
    with tf.Session() as sess:  

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

            # We create a directory to move the image already evaluated
            if not os.path.exists(dirname + "-Done"):
                os.mkdir(dirname + "-Done")

            #Image directory
            validation_images = []
            for image_name in glob.glob(dirname + "/*"):
                validation_images.append(image_name)

            L = len(validation_images)

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
                                

                # logits/BiasAdd:0 : output
                # sess.run return the first parameter, here logits
                # 'Placeholder:0': batch = data
                # 'Placeholder_3:0':False = not training
                batch_logits = sess.run('logits/BiasAdd:0', feed_dict={'Placeholder:0': batch, 'Placeholder_3:0':False})

                if batch_logits < 100:
                    total_logits[i] = int(round(batch_logits)) - 100
                else:
                    total_logits[i] = int(round(batch_logits)) - 99
                    if total_logits[i] > 100:
                        total_logits[i] = 100

                dest = dirname + "-Done"

                # We move the image already evaluated
                try:
                    shutil.move(im,dest)
                    # print "Move"
                except:
                    print "Move error"


            filenameCsv = dirname + "-predictions.csv"
            f = open(filenameCsv,'a')

            writer = csv.writer(f, lineterminator='\n')
            for nb_logits in total_logits:
                print(nb_logits)
                writer.writerow([nb_logits])
            f.close()
            
        # We close the client and socket
        client.close()
        stock.close()

