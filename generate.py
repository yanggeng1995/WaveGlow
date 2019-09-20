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

import sys
import tensorflow as tf
from scipy.io import wavfile
from models.glow import WaveGlow
import argparse
import numpy as np
from tqdm import tqdm
from hparam import hparams
import librosa
import os
import time

def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("Global step was: {}".format(global_step))
        print("Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Done.")
        return global_step
    else:
        print("No checkpoint found.")
        return None

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
    parser.add_argument('--out_dir', type=str, default='samples')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='restore model from checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    if args.restore_from is None:
        raise Exception('restore_from could not be None')

    batch_size = hparams.batch_size
    os.makedirs(args.out_dir, exist_ok=True)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

        model = WaveGlow()

        condition_placeholder = tf.placeholder(dtype=tf.float32,
                shape=[1, None, hparams.num_mels])

        outputs = model.inference(condition_placeholder)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        load(saver, sess, args.restore_from)

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
