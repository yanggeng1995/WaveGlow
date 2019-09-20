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

import os
import random
import threading
import codecs
import queue
import numpy as np

def read_wave_and_lc_features(wave_dir, lc_dir):
    filelist = []
    for filename in os.listdir(wave_dir):
        if os.path.exists(os.path.join(lc_dir, filename)):
            filelist.append(filename)

    random.shuffle(filelist)
    for file_id in filelist:
        wave_path = os.path.join(wave_dir, file_id)
        lc_path = os.path.join(lc_dir, file_id)

        # read wave
        audio = np.load(wave_path)
        audio = audio.reshape(-1, 1)

        # read local condition
        lc_features = np.load(lc_path)

        yield audio, lc_features, file_id

class DataReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 coord,
                 wave_dir,
                 lc_dir,
                 sample_size=18000,
                 upsample_rate=200,
                 queue_size=128):
        self.coord = coord
        self.wave_dir = wave_dir
        self.lc_dir = lc_dir
        self.sample_size = sample_size
        self.upsample_rate = upsample_rate
        self.threads = []
        self.queue = queue.Queue(maxsize=queue_size)

        self.lc_frames = sample_size // upsample_rate

    def dequeue(self, num_elements):
        sample_list = []
        lc_list = []
        for i in range(num_elements):
            sample, lc = self.queue.get(block=True)

            sample_list.append(sample)
            lc_list.append(lc)
        max_len = max([len(x) for x in sample_list])
        sample_batch = [
            np.pad(x, [[0, max_len - len(x)], [0, 0]], 'constant')
            for x in sample_list
        ]
        max_len = max([len(x) for x in lc_list])
        lc_batch = [
            np.pad(x, [[0, max_len - len(x)], [0, 0]],'edge')
            for x in lc_list
        ]
        
        sample_batch = np.stack(sample_batch)
        lc_batch = np.stack(lc_batch)
        
        return sample_batch, lc_batch

    def thread_main(self):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = read_wave_and_lc_features(self.wave_dir,
                                                 self.lc_dir)
            for sample, condition, file_id in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                # force align wave & local condition
                if len(sample) > len(condition) * self.upsample_rate:
                    # clip audio
                    sample = sample[:len(condition) * self.upsample_rate, :]
                elif len(sample) < len(condition) * self.upsample_rate:
                    # clip local condition and audio
                    length = min(len(sample), len(condition) * self.upsample_rate)
                    length = length // self.upsample_rate * self.upsample_rate
                    sample = sample[:length]
                    condition = condition[:length // self.upsample_rate]
                else:
                    pass

                lc_frame = self.sample_size // self.upsample_rate
                if len(condition) > lc_frame:
                    c_offset = np.random.randint(0, len(condition) - lc_frame)
                    s_offset = c_offset * self.upsample_rate
                    sample = sample[s_offset : s_offset + self.sample_size]
                    condition = condition[c_offset : c_offset + lc_frame]
                else:
                    sample = sample[:len(condition) * self.upsample_rate]
                    condition = condition[:len(condition)]
                    
                self.queue.put([sample, condition])

    def start_threads(self, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=())
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

        return self.threads

'''
import tensorflow as tf
coord = tf.train.Coordinator()

reader = DataReader(coord, "data/train/audio", "data/train/mel")

reader.start_threads(10)

for _ in range(10):
    x, lc = reader.dequeue(10)
    print(x.shape)
    print(lc.shape)
'''
