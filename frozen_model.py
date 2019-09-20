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
import argparse
from hparam import hparams
from models.glow import WaveGlow
from tensorflow.python.framework import graph_util
import tensorflow as tf

def main(unused_argv):

    with tf.Graph().as_default() as graph:
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            model = WaveGlow()

            condition_placeholder = tf.placeholder(dtype=tf.float32,
                                       shape=[1, None, hparams.num_mels], name='condition')

            outputs = model.inference(condition_placeholder)
            outputs = tf.identity(outputs, name='outputs')

            print(condition_placeholder)
            print(outputs)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAGS.restore_from)
            saver.restore(sess, ckpt.model_checkpoint_path)

            output_graph_def = graph_util.convert_variables_to_constants(
                    sess,
                    graph.as_graph_def(),
                    "model/outputs".split(","))
            with tf.gfile.GFile(FLAGS.frozen_model, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--restore_from',
            default="logdir/waveglow",
            help='Path to model checkpoint')
    parser.add_argument(
            '--frozen_model',
            default="WaveGlow.pb",
            help='Path to trained frozen model.')
    parser.add_argument(
            '--hparams',
            default='',
            help='Hyperparameter overrides as a comma-separated list of name=value pairs'
    )
    parser.add_argument(
            '--model',
            default='wavernn')
    parser.add_argument(
            '--output_node_name',
            type=str,
            default='output',
            help='Name of output node')

    args = parser.parse_args()
    hparams.parse(args.hparams)
    if not args.restore_from and not args.frozen_model:
        raise Exception("Give at least one of checkpoint and frozen model path.")
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    #main()
