from __future__ import absolute_import
from preprocess import unpickle, get_next_batch, get_data

import os
import tensorflow as tf
import numpy as np
import random
import math


class ManualConv2d(tf.keras.layers.Layer):
    def __init__(self, filter_shape: list[int], strides: list[int]=[1,1,1,1], padding = "VALID", use_bias = True, trainable=True, *args, **kwargs):
        """
        :param filter_shape: list of [filter_height, filter_width, in_channels, out_channels]
        :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
        :param padding: either "SAME" or "VALID", capitalization matters
        """
        super().__init__()

        self.strides = strides
        self.padding = padding
        self.filter_shape = filter_shape

        def get_var(name, shape, trainable):
            return tf.Variable(tf.random.truncated_normal(shape, dtype=tf.float32, stddev=1e-1), name=name, trainable = trainable)

        self.filters = get_var("conv_filters", filter_shape, trainable)
        self.use_bias = use_bias
        if use_bias: self.bias = get_var("conv_bias", [filter_shape[-1]], trainable)
        else: self.bias = None

    def get_weights(self):
        if self.bias is not None: return self.filters, self.bias
        return self.filters

    def set_weights(self, filters, bias=None): 
        self.filters = filters
        if bias is not None: self.bias = bias

    def call(self, inputs):
        """
        :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
        """
         
        num_examples, in_height, in_width, input_in_channels = inputs.shape
        filter_height, filter_width, filter_in_channels, filter_out_channels = self.filters.shape

        strideX = self.strides[0]
        strideY = self.strides[1]
        assert strideX == 1 and strideY == 1
        assert filter_in_channels == input_in_channels

        if self.padding == 'VALID':
            up_padY = down_padY = left_padX = right_padX = 0
            padded_inputs = inputs
        else: #SAME
            if filter_height % 2 == 1:
                up_padY = down_padY = (filter_height - 1) // 2
            else:
                up_padY = filter_height // 2 - 1
                down_padY = filter_height // 2
            if filter_width % 2 == 1:
                left_padX = right_padX = (filter_width - 1) // 2
            else:
                left_padX = filter_width // 2 - 1
                right_padX = filter_width // 2

            padded_inputs = tf.pad(inputs,
                                paddings=[[0, 0], #num_examples
                                            [up_padY, down_padY],
                                            [left_padX, right_padX],
                                            [0, 0]], # in channels
                                mode='CONSTANT')

        out_h = (in_height + up_padY + down_padY - filter_height) // strideY + 1
        out_w = (in_width + left_padX + right_padX - filter_width) // strideX + 1

        output = tf.zeros((num_examples, out_h, out_w, filter_out_channels), dtype=inputs.dtype)

        for f in range(filter_out_channels):
            filt = self.filters[:, :, :, f]
            for i in range(out_h):
                for j in range(out_w):
                    patch = padded_inputs[:, i:i+filter_height, j:j+filter_width, :]
                    conv_sum = tf.reduce_sum(patch * filt, axis=[1, 2, 3])
                    
                    # To make the gradients traceable, update output with a matrix sum
                    mask_h = tf.one_hot(i, out_h, dtype=inputs.dtype)[None, :, None, None]
                    mask_w = tf.one_hot(j, out_w, dtype=inputs.dtype)[None, None, :, None]
                    mask_f = tf.one_hot(f, filter_out_channels, dtype=inputs.dtype)[None, None, None, :]

                    # Broadcasted outer product to target position
                    mask = mask_h * mask_w * mask_f  # [1, out_h, out_w, out_channels]
                    value = tf.reshape(conv_sum, [num_examples, 1, 1, 1])
                    output += value * mask

        return output
