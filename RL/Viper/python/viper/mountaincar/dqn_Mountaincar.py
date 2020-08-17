# Copyright 2017-2018 MIT
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

import tensorflow as tf
import tensorflow.contrib.layers as layers

#SOHEIL #######################################################
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# NEEDED FOR STORING OUTPUTS
import json
import h5py

# FOR CUSTOM TEST SAVING
import os

import numpy as np

# As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
class DQNPolicy:
    def __init__(self, env, model_path):
        # Setup
        self.env = env
        self.model_path = model_path
        self.num_actions = env.action_space.n
        self.input_shape = env.observation_space.shape
        self.env_name = 'MountainCar-v0'
        self.dqn = None

        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,) + self.input_shape,name = 'input'))
        self.model.add(Dense(48))
        self.model.add(Activation('relu'))
        self.model.add(Dense(48))
        self.model.add(Activation('relu'))
        self.model.add(Dense(48))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.num_actions, kernel_initializer='zeros', name='output_weights'))
        self.model.add(Activation('linear'))
        print(self.model.summary())

        memory = SequentialMemory(limit=50000, window_length=1) # limit=50000
        policy = BoltzmannQPolicy()
        self.dqn = DQNAgent(model=self.model, nb_actions=self.num_actions, memory=memory, nb_steps_warmup=10,target_model_update=1e-2, policy=policy)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        weights_filename = 'dqn_{}_weights.h5f'.format(self.env_name)
        # if args.weights:
        #     weights_filename = args.weights
        self.dqn.load_weights(self.model_path + weights_filename)


    def predict_q(self, imgs):
        # #SOHEIL################################################
        # if isinstance(imgs,np.ndarray):
        #     print("predict_Q_imgs_INPUT",type(imgs), imgs.shape)
        # elif type(imgs) == list:
        #     print("predict_Q_imgs_INPUT",type(imgs), len(imgs),type(imgs[0]),imgs[0].shape)
        # #######################################################
        qs1 = []
        for state in imgs:
            current_state = np.expand_dims(state, axis=0) 
            current_state = list(np.expand_dims(current_state, axis=0))
            qs = self.dqn.compute_batch_q_values(current_state)
            qs1.append(qs.tolist()[0])
        qs1 = np.asarray(qs1)

        # #SOHEIL################################################
        # print("predict_Q_imgs_OUTPUT",type(qs1), qs1.shape)
        # #######################################################
        return qs1
    
    def predict(self, imgs):
        # #SOHEIL################################################
        # if isinstance(imgs,np.ndarray):
        #     print("predict_ACTIONS_imgs_INPUT",type(imgs), imgs.shape)
        # elif type(imgs) == list:
        #     print("predict_ACTIONS_imgs_INPUT",type(imgs), len(imgs),type(imgs[0]),imgs[0].shape)
        # #######################################################
        acts = []
        if type(imgs) == list:
            for state in imgs:
                current_state = np.expand_dims(np.expand_dims(state, axis=0), axis = 0)
                current_state = list(np.expand_dims(current_state, axis=0))
                acts.append(np.asarray([np.argmax(self.dqn.model.predict(current_state))]))
        elif isinstance(imgs,np.ndarray):
            # print(imgs)
            current_state = np.expand_dims(imgs, axis=0) 
            current_state = list(np.expand_dims(current_state, axis=0))
            acts = np.asarray([np.argmax(self.dqn.model.predict(current_state))])

        # #SOHEIL################################################
        # print("predict_ACTIONS_imgs_OUTPUT",type(acts), acts.shape)
        # #######################################################
        return acts