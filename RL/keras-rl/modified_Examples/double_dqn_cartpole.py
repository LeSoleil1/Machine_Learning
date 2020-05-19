import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


# NEEDED FOR CALLBACK
from rl.callbacks import ModelIntervalCheckpoint, Callback

# NEEDED FOR STORING OUTPUTS
import json
import h5py

# I'M GOING TO PARSE INPUT IN ORDER TO PASS PARAMETERS SUCH AS TRAINING, TESTING , ...##########
import argparse

# NEEDED FOR WORKING WITH COMPUTATIONAL GRAPH AND WEIGHTS
from keras import backend as K

# SAVING DATA AS HDF5 #################################################################
def save_data_hdf5(data, ds_name, file_address, mode="w"):

    h5_file = h5py.File(file_address+".hdf5", mode)
    dset = h5_file.create_dataset(ds_name, shape=data.shape)
    dset[:] = data
    h5_file.close()

# CREATING MY OWN KERAS-RL CALLBACKS######################################################
class GradsCallback(Callback):
    """
    This class is a custom callback class which is used to represent gradients of inputs w.r.t  the taken action.
    """

    def __init__(self):
        self.h_actions = []
        self.h_gradients_0 = []
        self.h_gradients_1 = []
        self.h_states=[]
        self.epis = 0

    def _set_env(self, env):
        self.env = env
        self._register_gradients_func()

    def _register_gradients_func(self):
        """
        This function build a computational graph required to obtain gradients of outputs w.r.t to inputs
        :return: it does not return anything
        """
        self.input_tensors = []
        for i in self.model.model.inputs:
            self.input_tensors.append(i)

        self.get_gradients = {}
        self.outputs = 2
        for c in range(self.outputs):
            self.grads = K.gradients(self.model.model.get_output_at(0)[:, c], self.model.model.inputs)
            self.get_gradients[c] = K.function(inputs=self.input_tensors, outputs=self.grads)

    def _read_current_state(self):
        """
        This function return current states in a format that
         can be used by register functions to calculate gradients.
        :return: a 3-d tuple
        """
        #print(self.env.state)
        # current_state = self.env.get_reduced_features_tuple()
        current_state = self.env.state
        current_state = np.expand_dims(current_state, axis=0) # SOHEIL: WHY YOU DO THIS TWICE
        current_state = list(np.expand_dims(current_state, axis=0))
        return current_state

    def on_episode_begin(self, episode, logs={}):
        """Called at beginning of each episode"""
        pass

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""

        save_data_hdf5(np.asarray(self.h_gradients_0), "ep-grads-0_"+str(self.epis), "/home/lesoleil/Desktop/RL/keras-rl/soheil_logs/cartpole_double_dqn_log", 'a')
        save_data_hdf5(np.asarray(self.h_gradients_1), "ep-grads-1_" + str(self.epis), "/home/lesoleil/Desktop/RL/keras-rl/soheil_logs/cartpole_double_dqn_log", 'a')
        save_data_hdf5(np.asarray(self.h_actions), "ep-actions_" + str(self.epis), "/home/lesoleil/Desktop/RL/keras-rl/soheil_logs/cartpole_double_dqn_log", 'a')
        save_data_hdf5(np.asarray(self.h_states), "ep-states_" + str(self.epis), "/home/lesoleil/Desktop/RL/keras-rl/soheil_logs/cartpole_double_dqn_log", 'a')
        self.h_actions = []
        self.h_gradients_0 = []
        self.h_gradients_1 = []
        self.h_states = []

        self.epis+=1
        pass

    def on_step_begin(self, step, logs={}):
        """Called at beginning of each step"""

        pass

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        pass

    def on_action_begin(self, action, logs={}):
        """Called at beginning of each action"""

        c_s = self._read_current_state()

        out_grad_0 = self.get_gradients[0](c_s)[0]
        out_grad_1 = self.get_gradients[1](c_s)[0]

        self.h_gradients_0.append(out_grad_0[0])
        self.h_gradients_1.append(out_grad_1[0])

        self.h_actions.append(action)
        self.h_states.append(c_s[0][0])

        self.env.current_gradients = out_grad_0[0]

    def on_action_end(self, action, logs={}):
        """Called at end of each action"""
        pass

##############################################################################################################################

# HERE, I ADDED PARSER TO SEPARATE TRAIN AND TEST WHEN I DONT WANT TO TRAIN EACH TIME I'M TESTING##################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='CartPole-v0')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--nb_episodes_for_test',type=int,default=5,help="Number of episodes to test/evaluate. Default is 5")
parser.add_argument('--visualize', type=bool, default=False)
parser.add_argument('--callbacks', type=bool, default=False)
args = parser.parse_args()

ENV_NAME = args.env_name
######################################################################################################################

# ENV_NAME = 'CartPole-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model regardless of the double architecture
# if you enable double network in DQN , DQN will build a double network base on your model automatically
# Also, you can build a double network by yourself and turn off the double network in DQN.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

# enable the double network
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               enable_double_dqn=True, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


if args.mode == 'train':
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    
    # callbacks = []
    # if model_checkpoints:
    #     callbacks += [
    #         ModelIntervalCheckpoint(
    #             './checkpoints/checkpoint_weights.h5f',
    #             interval=checkpoint_interval)
    #     ]
    # if tensorboard:
    #     callbacks += [TensorBoard(log_dir='./logs')]
    # dqn.fit(env, nb_steps=50000, visualize=args.visualize, verbose=2,callbacks=callbacks)
    
    dqn.fit(env, nb_steps=50000, visualize=args.visualize, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('double_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=args.visualize)



# TESTING BASED ON SAVED WEIGHTS
if args.mode == 'test':
    weights_filename = 'double_dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    if args.callbacks == True:
        dqn.test(env, nb_episodes = args.nb_episodes_for_test , visualize = args.visualize , callbacks = [GradsCallback()])
    else:
        dqn.test(env, nb_episodes = args.nb_episodes_for_test , visualize = args.visualize)

