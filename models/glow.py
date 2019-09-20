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
from models.modules import Invertible1x1Conv, WaveNet

class WaveGlow(object):
    def __init__(self,
                 lc_channels=80,
                 n_flows=12,
                 n_group=8,
                 n_early_every=4,
                 n_early_size=2):
        self.lc_channels = lc_channels
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.upsample_factores = (5, 5, 8)
        self.upsample_widths = (3, 3, 3)
        
        self.upsample_filters = []
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        for i, stride in enumerate(self.upsample_factores):
            filter_width = self.upsample_widths[i]
            filter = tf.get_variable('upsample_filter_{}'.format(i),
                           [stride * filter_width, lc_channels, lc_channels],
                           initializer=initializer)
            self.upsample_filters.append(filter)

    def create_network(self, samples, local_conditions, name="waveglow"):
        with tf.variable_scope(name):
            shape = tf.shape(samples)
            batch, length = shape[0], shape[1]
            local_conditions = self.upsample_network(local_conditions)
            
            samples = tf.reshape(samples, [batch, -1, self.n_group * 1])
            local_conditions = tf.reshape(local_conditions,
                    [batch, -1, self.n_group * self.lc_channels])
            
            output_samples = []
            log_s_list = []
            log_det_w_list = []
            
            n_remaining_channels = self.n_group
            
            for k in range(self.n_flows):
                if k % self.n_early_every == 0 and k > 0:
                    output_samples.append(samples[:, :, :self.n_early_size])
                    samples = samples[:, :, self.n_early_size:]
                    n_remaining_channels -= self.n_early_size
                with tf.variable_scope('glow_{}'.format(str(k))):
                    samples, dlogdet = Invertible1x1Conv(samples, n_remaining_channels, False)
                    log_det_w_list.append(dlogdet)
                    
                    #affine coupling layer
                    n_half = int(n_remaining_channels // 2)
                    samples_a, samples_b = samples[:, :, :n_half], samples[:, :, n_half:]
                    log_s, shift = WaveNet(samples_a, local_conditions, n_half,
                                            self.lc_channels * self.n_group)
                    samples_b = samples_b * tf.exp(log_s) + shift
                    samples = tf.concat([samples_a, samples_b], axis=-1)
                    
                    log_s_list.append(log_s)
            output_samples.append(samples)
            
            self.z = tf.concat(output_samples, axis=-1)
            self.log_s_list = log_s_list
            self.log_det_w_list = log_det_w_list
    
    def add_loss(self, sigma=1.0):
        '''negative log-likelihood of the data x'''
        for i, log_s in enumerate(self.log_s_list):
            if i == 0:
                log_s_total = tf.reduce_sum(log_s)
                log_det_w_total = self.log_det_w_list[i]
            else:
                log_s_total += tf.reduce_sum(log_s)
                log_det_w_total += self.log_det_w_list[i]
        loss = tf.reduce_sum(self.z * self.z) / (2 * sigma * sigma)\
                    - log_det_w_total - log_s_total
        shape = tf.shape(self.z)
        size = tf.cast(shape[0] * shape[1] * shape[2], "float32")
        loss = loss / size
        
        self.loss = loss
        self.mean_log_det = -log_det_w_total / size
        self.mean_log_scale = -log_s_total / size
        self.prior_loss = tf.reduce_sum(self.z * self.z / (2 * sigma * sigma)) / size
        
    def upsample_network(self, local_conditions):
      
        for i, stride in enumerate(self.upsample_factores):
            filter = self.upsample_filters[i]
            lc_shape = tf.shape(local_conditions)
            batch_size, lc_length = lc_shape[0], lc_shape[1]
            output_shape = [batch_size, lc_length * stride, self.lc_channels]
            local_conditions = tf.contrib.nn.conv1d_transpose(value=local_conditions,
                            filter=filter, output_shape=output_shape, stride=stride)
        
        return tf.cast(local_conditions, tf.float32)
    
    def inference(self, local_conditions, sigma=0.7, name="waveglow"):
        with tf.variable_scope(name):
            batch = tf.shape(local_conditions)[0]
            remaining_channels = self.n_group
            for k in range(self.n_flows):
                if k % self.n_early_every == 0 and k > 0:
                    remaining_channels = remaining_channels - self.n_early_size
            
            local_conditions = self.upsample_network(local_conditions)
            shape = tf.shape(local_conditions)
            
            # need to make sure that length of lc_batch be multiple times of n_group
            pad = self.n_group - 1 - (shape[1] + self.n_group - 1) % self.n_group
            local_conditions = tf.pad(local_conditions, [[0, 0], [0, pad], [0, 0]])
            local_conditions = tf.reshape(local_conditions,
                                          [shape[0], -1, self.lc_channels * self.n_group])
            
            shape = tf.shape(local_conditions)
            samples = tf.random_normal([shape[0], shape[1], remaining_channels])
            samples = samples * sigma
            
            for k in reversed(range(self.n_flows)):
                with tf.variable_scope('glow_{}'.format(str(k))):
                    #affine coupling layer
                    n_half = int(remaining_channels / 2)
                    samples_a, samples_b = samples[:, :, :n_half], samples[:, :, n_half:]
                    log_s, shift = WaveNet(samples_a, local_conditions, n_half,
                                           self.lc_channels * self.n_group)
                    samples_b = (samples_b - shift) / tf.exp(log_s)
                    samples = tf.concat([samples_a, samples_b], axis=-1)
                    
                    #inverse 1*1 conv1d
                    samples = Invertible1x1Conv(samples, remaining_channels, True)
                if k % self.n_early_every == 0 and k > 0:
                    z = tf.random_normal([shape[0], shape[1], self.n_early_size])
                    z = z * sigma
                    samples = tf.concat([z, samples], axis=-1)
                    
                    remaining_channels += self.n_early_size
            
            outputs = tf.reshape(samples, [shape[0], -1, 1])
            return outputs

'''
import numpy as np
model = WaveGlow()
inputs = np.random.randn(2, 10, 80)

x = tf.placeholder(shape=[None, None, 80], dtype=tf.float32)
out = model.inference(x, 1.0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(out, feed_dict={x :inputs})
    print(output.shape)
'''
