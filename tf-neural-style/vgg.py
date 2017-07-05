# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'    
)
LAPLACIAN_LAYERS = ( 'pool_lap1', 'lap1', 'pool_lap2', 'lap2', 'pool_lap3', 'lap3' )

def load_net(data_path):
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    return weights, mean_pixel

def net_preloaded(weights, input_image, pooling):
    net = {}
    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, pooling)
        net[name] = current

    laplacian = np.array( [ [0,-1,0], [-1,4,-1], [0,-1,0] ], dtype=np.float32 )
    lapW = np.zeros( (3, 3, 3, 1), dtype=np.float32 )
    for t in range(3):
        lapW[:,:,t,0] = laplacian
    
    for i in range(1,4):
        net['pool_lap%d'%i] = _pool_layer( input_image, 'avg', 2**i )
        #net['lap%d'%i] = _conv_layer(net['pool_lap%d'%i], lapW, [0.0])
        net['lap%d'%i] = _lap_layer(net['pool_lap%d'%i])
    assert len(net) == len(VGG19_LAYERS) + len(LAPLACIAN_LAYERS)
    return net

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input, pooling, poolsize=2):
    ksize = (1, poolsize, poolsize, 1)
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize, strides=(1, poolsize, poolsize, 1),
                padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize, strides=(1, poolsize, poolsize, 1),
                padding='SAME')

def _lap_layer(input):
    laplacian = np.array( [ [0,-1,0], [-1,4,-1], [0,-1,0] ], dtype=np.float32 )
    lapW = np.zeros( (3, 3, 1, 1), dtype=np.float32 )
    lapW[:,:,0,0] = laplacian
    color_outs = []
    for i in range(3):
        color = input[:,:,:,i]
        color4d = tf.expand_dims(color, -1)
        color_out = _conv_layer(color4d, lapW, [0.0])
        color_outs.append(color_out)
    output = tf.concat(color_outs, axis=-1)
    output = tf.abs(output)
    sum_output = tf.reduce_sum(output, reduction_indices=[3], keep_dims=False)
    #cut_cond = tf.less( max_output, tf.ones(tf.shape(max_output)) * thres )
    #cut_output = tf.where( cut_cond, tf.zeros(tf.shape(max_output)), max_output )
    return sum_output
    
    
def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel
