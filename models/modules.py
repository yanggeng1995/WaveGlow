# Copyright 2018 ASLP@NPU.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: yanggeng1995@gmail.com (YangGeng)

import tensorflow as tf
import numpy as np

def create_variable(name, shape):
    with tf.device("/cpu:0"):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        variable = tf.get_variable(initializer=initializer(shape=shape), name=name)
        return variable
    
def create_bias_variable(name, shape):
    with tf.device("/cpu:0"):
        initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
        return tf.get_variable(initializer=initializer(shape=shape), name=name)
    
def Invertible1x1Conv(inputs, channels, reverse=False, name="invertible1x1conv"):
    with tf.variable_scope(name):
        shape = tf.shape(inputs)
        batch, length = shape[0], shape[1]
        
        w_init = np.linalg.qr(np.random.randn(channels, channels))[0].astype('float32')
        w = tf.get_variable(name='w', dtype=tf.float32, initializer=w_init)
       
        logdet = tf.log(tf.abs(tf.cast(
             tf.matrix_determinant(tf.cast(w, tf.float64)), tf.float32))) 
        dlogdet = logdet * tf.cast(batch * length, tf.float32)
        if not reverse:
            _w = tf.reshape(w, [1, channels, channels])
            outputs = tf.nn.conv1d(inputs, _w, stride=1, padding='SAME')
            return outputs, dlogdet
        else:
            _w = tf.matrix_inverse(w)
            _w = tf.reshape(_w, [1, channels, channels])
            outputs = tf.nn.conv1d(inputs, _w, stride=1, padding='SAME')
            return outputs
        
def WaveNet(samples, local_conditions, n_in_channels, lc_channels, n_layers=8,
            residual_channels=256, skip_channels=256, kernel_size=3,
            use_weight_normalization=True, name="wavenet"):
    with tf.variable_scope(name):
        #pre processing
        w_s = create_variable('w_start', [1, n_in_channels, residual_channels])
        b_s = create_bias_variable('b_start', [residual_channels])
        if use_weight_normalization:
           w_s = weight_normalization(w_s, 'w_start_g')
        outputs = tf.nn.bias_add(tf.nn.conv1d(samples, w_s, 1, 'SAME'), b_s)

        skip_outputs_list = []
        
        for i in range(n_layers):
            dilation = 2**i
            outputs, skip_outputs = causal_dilated_conv1d(outputs, local_conditions,
                                             dilation, lc_channels, kernel_size,
                                             residual_channels, skip_channels,
                                             use_weight_normalization)
            # outputs = outputs + residual_outpus
            skip_outputs_list.append(skip_outputs)
        skip_outputs = sum(skip_outputs_list)
        
        # post processing
        w_e = create_variable('w_end', [1, skip_channels, n_in_channels * 2])
        b_e = create_bias_variable('b_end', [n_in_channels * 2])
        if use_weight_normalization:
            w_e = weight_normalization(w_e, 'w_end_g')
        outputs = tf.nn.bias_add(tf.nn.conv1d(skip_outputs, w_e, 1, 'SAME'), b_e)
       
        '''
        #log_s = alpha * tanh(logs) + beta, helpful for training stability

        log_s = outputs[:, :, 0::2]
        shift = outputs[:, :, 1::2] 
        rescale = tf.get_variable("rescale", [],
                      initializer=tf.constant_initializer(1.))
        scale_shift = tf.get_variable("scale_shift", [],
                      initializer=tf.constant_initializer(-3.))
        log_s = tf.tanh(log_s) * rescale + scale_shift

        return log_s, shift
        ''' 
        return outputs[:, :, :n_in_channels], outputs[:, :, n_in_channels:]
            
def causal_dilated_conv1d(samples, local_conditions, dilation, lc_channels,
                          kernel_size=3, residual_channels=256, skip_channels=256,
                          use_weight_normalization=False):
    assert (kernel_size % 2 == 1)
    input = samples
    with tf.variable_scope("dilated_conv1d_{}".format((str(dilation)))):
        w_s = create_variable("w_s", 
                              [kernel_size, residual_channels, 2 * residual_channels])
        b_s = create_bias_variable('b_s', [2 * residual_channels])
        if use_weight_normalization:
            w_s = weight_normalization(w_s, "w_s_g")
        samples = tf.nn.bias_add(causal_conv(samples, w_s, dilation, kernel_size), b_s)
        
        #process local condition
        w_c = create_variable('w_c', [1, lc_channels, 2 * residual_channels])
        b_c = create_bias_variable('b_c', [2 * residual_channels])
        if use_weight_normalization:
            w_c = weight_normalization(w_c, "w_c_g")
        local_conditions = tf.nn.bias_add(
            tf.nn.conv1d(local_conditions, w_c, 1, 'SAME'), b_c)
        
        out = samples + local_conditions
        filter = tf.nn.tanh(out[:, :, :residual_channels])
        gate = tf.nn.sigmoid(out[:, :, residual_channels:])
        out = gate * filter
        
        #skip
        w_skip = create_variable('w_skip', [1, residual_channels, skip_channels])
        b_skip = create_bias_variable('b_skip', [skip_channels])
        if use_weight_normalization:
            w_skip = weight_normalization(w_skip, "w_skip_g")
        skip_output = tf.nn.bias_add(
            tf.nn.conv1d(out, w_skip, 1, 'SAME'), b_skip)
        
        # residual conv1d
        w_residual = create_variable('w_residual', 
                                     [1, residual_channels, residual_channels])
        b_residual = create_bias_variable('b_residual', [residual_channels])
        if use_weight_normalization:
            w_residual = weight_normalization(w_residual, 'w_residual_g')
        residual_output = tf.nn.bias_add(
            tf.nn.conv1d(out, w_residual, 1, 'SAME'), b_residual)
        
        return residual_output + input, skip_output 
        
def weight_normalization(w, name):
    units = w.get_shape()[-1].value
    g = create_variable(name, [units])
    w = tf.nn.l2_normalize(w, [0, 1]) * g
    return w 
   
def causal_conv(value, filter_, dilation, filter_width, name='causal_conv'):
    with tf.name_scope(name):
        # Pad beforehand to preserve causality.
        pad = int((filter_width - 1) * dilation / 2)
        padding = [[0, 0], [pad, pad], [0, 0]]
        padded = tf.pad(value, padding)
        if dilation > 1:
            transformed = time_to_batch(padded, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1,
                                padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(padded, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, tf.shape(value)[1], -1])
        return result
    
def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])
    
'''    
import numpy as np
inputs = np.random.randn(2, 1000, 8)
inputs_mel = np.random.randn(2, 1000, 80 * 8)
x = tf.placeholder(shape=[None, None, 8], dtype=tf.float32)
mel = tf.placeholder(shape=[None, None, 80 * 8], dtype=tf.float32)

a, b = WaveNet(x, mel, 8, 640)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x1, x2 = sess.run([a, b], feed_dict={x :inputs, mel:inputs_mel})
    print(x1.shape)
    print(x2.shape)
'''
