from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from base_model import CifarModel

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class MLP(CifarModel):
    def __init__(self, classes):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(MLP, self).__init__()

        # Initialize all hyperparameters
        self.loss_list = []
        self.batch_size = 64
        self.input_width = 32
        self.input_height = 32
        self.image_channels = 3
        self.num_classes = len(classes)
        self.first_hidden_layer_size = 3072
        self.second_hidden_layer_size = 3072
        self.third_hidden_layer_size = 2048
        self.dropout_p =0.1

        # TODO mlp.MLP.__init__(): Initialize your Layers here.
        self.first_layer = tf.keras.layers.Dense(self.first_hidden_layer_size, activation="gelu", use_bias=True)
        self.first_dropout = tf.keras.layers.Dropout(self.dropout_p)

        self.second_layer = tf.keras.layers.Dense(self.second_hidden_layer_size, activation="gelu", use_bias=True)
        self.second_dropout = tf.keras.layers.Dropout(self.dropout_p)

        self.third_layer = tf.keras.layers.Dense(self.third_hidden_layer_size, activation="gelu", use_bias=True)
        self.third_dropout = tf.keras.layers.Dropout(self.dropout_p)

        self.last_layer = tf.keras.layers.Dense(self.num_classes, activation=None, use_bias=True)


    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # TODO mlp.MLP.call(): Implement your forward pass here.
        
        # assert(self.batch_size ==tf.shape(inputs)[0]) can fail on the last smaller batch
        
        num_inputs = tf.shape(inputs)[0]
        flattened = tf.reshape(inputs, [num_inputs, -1])
        h1 = self.first_layer(flattened)
        d1 = self.first_dropout(h1,training= (not is_testing))
        
        h2 = self.second_layer(d1)
        d2 = self.first_dropout(h2,training= (not is_testing))
        
        h3 = self.third_layer(d2)
        d3 = self.first_dropout(h3,training= (not is_testing))

        logits = self.last_layer(d3)
        
        return logits
