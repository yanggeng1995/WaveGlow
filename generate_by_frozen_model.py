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
from scipy.io import wavfile
import argparse
import numpy as np
from hparam import hparams
import os
import time

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveGlow Network')
    parser.add_argument('--lc_dir', type=str, default=None, required=True,
                        help='local condition file')
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--frozen_model', type=str, default=None,
                        help='frozen model path')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    if args.frozen_model is None:
        raise Exception('frozen model could not be None')

    os.makedirs(args.out_dir, exist_ok=True)

    sess = tf.Session()
    with tf.gfile.FastGFile(args.frozen_model, 'rb') as f:
       graph_def = tf.GraphDef()
       graph_def.ParseFromString(f.read())
       sess.graph.as_default()
       tf.import_graph_def(graph_def, name='')
 
    sess.run(tf.global_variables_initializer())

    condition_placeholder = sess.graph.get_tensor_by_name("model/condition:0")
    outputs = sess.graph.get_tensor_by_name("model/outputs:0")

    print('generating samples')
    for filename in os.listdir(args.lc_dir):
        # load local condition
        start = time.time()
        lc = np.load(os.path.join(args.lc_dir, filename))
        lc = np.reshape(lc, [1, -1, hparams.num_mels])

        wave_path = os.path.join(args.out_dir, filename.replace("npy", "wav"))
        audios = sess.run(outputs, feed_dict={condition_placeholder: lc})
        audios = np.squeeze(audios)
        wavfile.write(wave_path, hparams.sample_rate, np.asarray(audios))
        print(filename, "{:.3f} seconds".format(time.time() - start))
    print("Generate Done!")
