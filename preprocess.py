import pickle
import numpy as np
import tensorflow as tf
import os


def unpickle(file) -> dict[str, np.ndarray]:
    """
    CIFAR data contains the files data_batch_1, data_batch_2, ..., 
    as well as test_batch. We have combined all train batches into one
    batch for you. Each of these files is a Python "pickled" 
    object produced with cPickle. The code below will open up a 
    "pickled" object (each file) and return a dictionary.
    NOTE: DO NOT EDIT
    :param file: the file to unpickle
    :return: dictionary of unpickled data
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_next_batch(idx, inputs, labels, batch_size=100) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an index, returns the next batch of data and labels. Ex. if batch_size is 5, 
    the data will be a numpy matrix of size 5 * 32 * 32 * 3, and the labels returned will be a numpy matrix of size 5 * 10.
    """
    return (inputs[idx*batch_size:(idx+1)*batch_size], np.array(labels[idx*batch_size:(idx+1)*batch_size]))


def get_data(file_path, classes) -> tuple[np.ndarray, tf.Tensor]:
    """
    Given a file path and a list of class indices, returns an array of 
    normalized inputs (images) and an array of labels. 
    
    - **Note** that because you are using tf.one_hot() for your labels, your
    labels will be a Tensor, hence the mixed output typing for this function. This 
    is fine because TensorFlow also works with NumPy arrays, which you will
    see more of in the next assignment. 

    :param file_path: file path for inputs and labels, something 
                        like 'CIFAR_data_compressed/train'
    :param classes: list of class labels (0-9) to include in the dataset

    :return: normalized NumPy array of inputs and tensor of labels, where 
                inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and 
                Tensor of labels with size (num_examples, num_classes)
    """
    unpickled_file: dict[str, np.ndarray] = unpickle(file_path)
    inputs: np.ndarray = np.array(unpickled_file[b'data'])
    labels: np.ndarray = np.array(unpickled_file[b'labels'])

    # TODO: Extract only the data that matches the corresponding classes we want
    
    #1.1.1
    mask = np.isin(labels, classes)
    filtered_inputs = inputs[mask]
    filtered_labels = labels[mask]

    #1.1.2
    num_examples = len(filtered_labels)
    reshaped_inputs = tf.reshape(filtered_inputs, (num_examples,3,32,32)) 
    reshaped_inputs = tf.transpose(reshaped_inputs,[0, 2, 3, 1]) #(num_examples,32,32,3)
    
    normalized_inputs = tf.cast(reshaped_inputs, tf.float32) / 255.0

    #1.1.3
    left_classes = np.unique(filtered_labels)
    new_labels = np.zeros_like(filtered_labels)
    for new_id, old_id in enumerate(left_classes):
        new_labels = np.where(filtered_labels == old_id, new_id, new_labels)
    one_hot_new_labels = tf.one_hot(new_labels, len(left_classes))

    return normalized_inputs.numpy(), one_hot_new_labels
    