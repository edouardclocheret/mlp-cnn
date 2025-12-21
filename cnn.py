from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from manual_convolution import ManualConv2d
from base_model import CifarModel

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class CNN(CifarModel):
    def __init__(self, classes):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(CNN, self).__init__()

        # Initialize all hyperparameters
        self.loss_list = []
        self.batch_size = 64
        self.input_width = 32
        self.input_height = 32
        self.image_channels = 3
        self.num_classes = len(classes)

        self.first_kernel_size = (3,3) #(filter_height, filter_width, in_channels, out_channels)
        self.first_filter = 32
        self.second_kernel_size = (3,3)
        self.second_filter = 64
        self.third_kernel_size = (3,3)
        self.third_filter = 128

        

        self.dropout_p =0.1
        self.epsilon = 1e-3  # this is used for batch normalization only!
        self.first_hidden_layer_size = 512
        self.second_hidden_layer_size = 256
        self.last_hidden_layer_size = self.num_classes

        # Fill the rest of this out!
        # strides=[1,1,1,1], #batch, w, h, channel is strides=(1,1)
        self.conv_1 = tf.keras.layers.Conv2D(filters = self.first_filter, 
                                                 kernel_size = self.first_kernel_size,
                                                 strides=(1,1),
                                                 padding='valid',
                                                 use_bias =True)
        
        manual_filter_shape = [self.first_kernel_size[0], self.first_kernel_size[1],3,self.first_filter]#[filter_height, filter_width, in_channels, out_channels]
        manual_strides = [1,1,1,1]
        self.conv_1_manual = ManualConv2d(filter_shape=manual_filter_shape, strides=manual_strides, padding = "VALID", use_bias = True, trainable=True)
        
        
        self.bn_1 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.relu_1 = tf.keras.layers.ReLU()
        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv_2 = tf.keras.layers.Conv2D(filters = self.second_filter, 
                                                 kernel_size = self.second_kernel_size,
                                                 strides=(1,1),
                                                 padding='valid',
                                                 use_bias =True)
        self.bn_2 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.relu_2 = tf.keras.layers.ReLU()
        self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        manual_filter_shape_3 = [self.third_kernel_size[0], self.third_kernel_size[1],
                         self.second_filter, self.third_filter]  # shape: [H, W, in_ch, out_ch]
        manual_strides = [1,1,1,1]
        self.conv_3_manual = ManualConv2d(filter_shape=manual_filter_shape_3,
                                        strides=manual_strides,
                                        padding="VALID",
                                        use_bias=True,
                                        trainable=True)


        self.conv_3 = tf.keras.layers.Conv2D(filters = self.third_filter, 
                                                 kernel_size = self.third_kernel_size,
                                                 strides=(1,1),
                                                 padding='valid',
                                                 use_bias =True)
        self.bn_3 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.relu_3 = tf.keras.layers.ReLU()
        
        self.dense_1 = tf.keras.layers.Dense(self.first_hidden_layer_size, activation="gelu", use_bias=True)
        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_p)
        self.dense_2 = tf.keras.layers.Dense(self.second_hidden_layer_size, activation="gelu", use_bias=True)
        self.dropout_2 = tf.keras.layers.Dropout(self.dropout_p)
        self.last_layer = tf.keras.layers.Dense(self.last_hidden_layer_size, activation=None, use_bias=True)



    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        num_inputs = inputs.shape [0]
        
        c1 = self.conv_1(inputs)
        bn1 = self.bn_1(c1, training=not is_testing)
        rl1 = self.relu_1(bn1)
        p1 = self.pool_1(rl1)

        c2 = self.conv_2(p1)
        bn2 = self.bn_2(c2, training=not is_testing)
        rl2 = self.relu_2(bn2)
        p2 = self.pool_2(rl2)

        if is_testing:
            tf_weights, tf_bias = self.conv_3.get_weights()
            self.conv_3_manual.set_weights(tf_weights, tf_bias)
            c3 = self.conv_3_manual(p2)
        else:
            c3 = self.conv_3(p2)

        bn3 = self.bn_3(c3, training=not is_testing)
        rl3 = self.relu_3(bn3)

        r = tf.reshape(rl3, [num_inputs, -1])
        d1 = self.dense_1(r)
        dr1 = self.dropout_1(d1, training=not is_testing)
        d2 = self.dense_2(dr1)
        dr2 = self.dropout_2(d2, training=not is_testing)
        out = self.last_layer(dr2)
        
        return out
