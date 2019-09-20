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

from __future__ import print_function
from data_reader import DataReader
from hparam import hparams
import tensorflow as tf
import time
import argparse
import os
import sys
from datetime import datetime
from models.glow import WaveGlow

def get_arguments():

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveRNN Network')
    parser.add_argument('--wave_dir', type=str, default="data/train/audio",
                        help='wave data directory for training data.')
    parser.add_argument('--lc_dir', type=str, default="data/train/mel",
                        help='local condition directory for training data.')
    parser.add_argument('--run_name', type=str, default='waveglow',
                        help='run name for log saving')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='restore model from checkpoint')
    
    return parser.parse_args()

def consine_learning_rate_decay(global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 2000.0
    init_lr = hparams.initial_learning_rate
    step = tf.cast(global_step + 1, dtype=tf.float32)
    learning_rate = init_lr * warmup_steps**0.5 * \
           tf.minimum(step * warmup_steps**-1.5, step**-0.5)

    return tf.maximum(hparams.final_learning_rate, learning_rate)

def linear_learning_rate_decay(global_step):
    learning_rate = tf.cond(
        global_step < hparams.warmup_steps,
        lambda: tf.convert_to_tensor(hparams.initial_learning_rate),
        lambda: tf.train.exponential_decay(hparams.initial_learning_rate,
                            global_step - hparams.warmup_steps + 1,
                            hparams.warmup_steps, hparams.decay_rate)
    )
    return tf.maximum(hparams.final_learning_rate, learning_rate)

def save(saver, sess, logdir, step, write_meta_graph=True):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step,
            write_meta_graph=write_meta_graph)

    print('Done.')

def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        none_grad = False
        for g, _ in grad_and_vars:
            if g is None:
                none_grad = True
                break

            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        if not none_grad:
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
        else:
            grad = None

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def assign_to_device(device, ps_device=None):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on.  Example values are GPU:0 and
        CPU:0.

    If ps_device is not set then the variables will be placed on the device.
    The best device for shared varibles depends on the platform as well as the
    model.  Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.

    """
    ps_ops = ['Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
              'MutableHashTableOfTensors', 'MutableDenseHashTable']

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            return "/" + ps_device
        else:
            return device

    return _assign

def add_stats(model):
    tf.summary.scalar('mean_log_det', model.mean_log_det)
    tf.summary.scalar('mean_log_scale', model.mean_log_scale)
    tf.summary.scalar('prior_loss', model.prior_loss)
    tf.summary.scalar('total_loss', model.loss)

def main():
    args = get_arguments()
    args.logdir = os.path.join(hparams.logdir_root, args.run_name)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # Create coordinator.
    coord = tf.train.Coordinator()

    with tf.device('/cpu:0'):
        with tf.name_scope('inputs'):
            reader = DataReader(coord, args.wave_dir,
                        args.lc_dir, hparams.sample_size)

    global_step = tf.get_variable("global_step", [], 
            initializer=tf.constant_initializer(0), trainable=False)

    if hparams.learning_rate_decay_way == "cosine":
        learning_rate = consine_learning_rate_decay(global_step)
    else:
        learning_rate = linear_learning_rate_decay(global_step)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    tf.summary.scalar('learning_rate', learning_rate)

    gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    tower_grads, tower_losses = [], []

    x_placeholder = []
    lc_placeholder = []
    for _ in range(len(gpu_ids)):
        x_placeholder.append(
                tf.placeholder(dtype=tf.float32, shape=[None, None, 1]))
        lc_placeholder.append(
                tf.placeholder(dtype=tf.float32, shape=[None, None, hparams.num_mels]))

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        for i in range(len(gpu_ids)):
            with tf.device(assign_to_device('/gpu:%d' % int(gpu_ids[i]),
                ps_device='cpu:0')), tf.name_scope('tower_%d' % int(i)):
                model = WaveGlow(lc_channels=hparams.num_mels)
                
                model.create_network(x_placeholder[i], lc_placeholder[i])

                #with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                #    tf.contrib.quantize.create_training_graph(quant_delay=0)
                model.add_loss()
                tower_losses.append(model.loss)
                grads = optimizer.compute_gradients(model.loss)
                tower_grads.append(grads)
                add_stats(model)

    with tf.name_scope('average_grad'):
        averaged_loss = tf.add_n(tower_losses) / len(tower_losses)
        tf.summary.scalar('average_loss', averaged_loss)
        averaged_gradients = average_gradients(tower_grads)

        # gradient clipping
        gradients = [grad for grad, var in averaged_gradients]
        params = [var for grad, var in averaged_gradients]
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, hparams.clip_norm)

        # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
        # https://github.com/tensorflow/tensorflow/issues/1122
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.apply_gradients(zip(clipped_gradients, params),
                                                     global_step=global_step)
            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)     # batch norm update

            train_ops = tf.group([train_op, update_op])

    stats =  tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=10)
    config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    #tf.contrib.quantize.create_training_graph(
    #        input_graph=tf.get_default_graph(),
    #        quant_delay=0)
    with tf.Session(config=config) as sess:
        try:
            reader.start_threads()
            summary_writer = tf.summary.FileWriter(args.logdir, sess.graph)
            sess.run(tf.global_variables_initializer())
            saved_global_step = 0
            last_saved_step = 0
            step = 0

            if args.restore_from is not None:
                try:
                    saved_global_step = load(saver, sess, args.restore_from)
                except Exception:
                    print("Something went wrong while restoring checkpoint. "
                          "We will terminate training to avoid accidentally overwriting "
                          "the previous model.")
                    raise
                print("Restore model successfully!")
            else:
                print("Start new training.")
            last_saved_step = saved_global_step

            for step in range(saved_global_step + 1, hparams.train_steps):
                start_time = time.time()
                x, lc = reader.dequeue(num_elements=hparams.batch_size * len(gpu_ids))
                dicts = dict()
                for i in range(len(gpu_ids)):
                    dicts[x_placeholder[i]] = x[i * hparams.batch_size : (i + 1) * hparams.batch_size]
                    dicts[lc_placeholder[i]] = lc[i * hparams.batch_size : (i + 1) * hparams.batch_size]

                _, loss, lr, _, summary = sess.run(
                        [global_step, averaged_loss,
                         learning_rate, train_ops, stats],
                         feed_dict=dicts)
                duration = time.time() - start_time
                step_log = 'step {:d} loss={:.3f} lr={:.8f} time={:4f}'\
                        .format(step, loss, lr, duration)
                print(step_log)

                if step % hparams.save_model_every == 0:
                    save(saver, sess, args.logdir, step)
                    last_saved_step = step

                if step % hparams.summary_interval == 0:
                    summary_writer.add_summary(summary, step)

        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()
        finally:
            if step > last_saved_step:
                save(saver, sess, args.logdir, step)
                coord.request_stop()
                coord.join()

if __name__ == '__main__':
    main()
