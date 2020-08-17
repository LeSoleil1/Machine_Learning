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

from ..core.rl import *
from .pong import *
from .dqn import *
from ..core.dt import *
from ..util.log import *

# SS
import argparse
import time
import h5py
import numpy as np
import random


def learn_dt():
    # Parameters
    log_fname = '../pong_dt.log'
    model_path = '../data/model-atari-pong-1/saved'
    max_depth = 12
    n_batch_rollouts = 10
    max_samples = 200000
    max_iters = 80
    train_frac = 0.8
    is_reweight = True
    n_test_rollouts = 100 # SS: default is 50
    save_dirname = '../tmp/pong'
    save_fname = 'dt_policy.pk'
    save_viz_fname = 'dt_policy.dot'
    is_train = False # True
    
    # SS ######################################################################3
    is_Naive = True     
    criterion = 'Entropy'
    trained_Episodes = 800
    model_Max_Depth = 18
    ###########################################################################3


    # Logging
    set_file(log_fname)
    
    # Data structures
    env = get_pong_env()
    teacher = DQNPolicy(env, model_path)
    student = DTPolicy(max_depth)
    state_transformer = get_pong_symbolic

    #SOHEIL ###################################################################3
    # print(type(state_transformer)) #
    ###########################################################################3

    # Train student
    if is_train:
        student = train_dagger(env, teacher, student, state_transformer, max_iters, n_batch_rollouts, max_samples, train_frac, is_reweight, n_test_rollouts)
        save_dt_policy(student, save_dirname, save_fname)
        save_dt_policy_viz(student, save_dirname, save_viz_fname)
        # Test student
        rew = test_policy(env, student, state_transformer, n_test_rollouts)
        log('Final reward: {}'.format(rew), INFO)
        log('Number of nodes: {}'.format(student.tree.tree_.node_count), INFO)

    elif is_Naive == True:
        # Logging
        log_fname = '../pong_DT_Naive.log'
        set_file(log_fname)
        for depth in range(1,model_Max_Depth + 1):
            saved_Name = criterion+'_model_'+ str(trained_Episodes) + '_' + str(depth) + '.sav'
            saved_Directory = save_dirname + '/Naive'
            #Load The Models
            student = load_dt_policy(saved_Directory, saved_Name)
            # Test Model
            rew = test_policy(env, student, state_transformer, n_test_rollouts)
            log('Naive approach: Final reward in {} rollouts for trained DT_{} episodes {} and depth:{} is {}'.format(n_test_rollouts,criterion,trained_Episodes, depth, rew), INFO)

    else:
        student = load_dt_policy(save_dirname, save_fname)
        # Test student
        rew = test_policy(env, student, state_transformer, n_test_rollouts)
        log('Final reward: {}'.format(rew), INFO)
        log('Number of nodes: {}'.format(student.tree.tree_.node_count), INFO)

def generate_Data(env_name, rand_seed, nb_episodes_for_test, visualize, direction,max_depth):
    step = 0
    seed = rand_seed  # the seed (1 for the training_set, 10 for the test set)
    # visualize = False  # True if you want the visualization
    nb_episodes = nb_episodes_for_test
    
    env = wrap_dqn(gym.make(env_name))
    agent = DQNPolicy(env, model_path='../data/model-atari-pong-1/saved')
    nb_actions = env.action_space.n
    print(f'The number of actions: {nb_actions}.')



    dataset_path = f'../dataset/pong_{direction}_dqn_log_{seed}_for_{nb_episodes}_episodes'

    def save_data_hdf5(data, ds_name, file_address, mode="w", dtype='uint8'):
        """ Save the data as .hdf5
        """
        h5_file = h5py.File(file_address + ".hdf5", mode)
        dset = h5_file.create_dataset(ds_name, shape=data.shape, dtype=dtype)
        dset[:] = data
        h5_file.close()

    unwrapped = env
    while unwrapped != env.unwrapped:
        unwrapped.seed(seed)
        unwrapped = unwrapped.env
    set_global_seeds(seed)

    start_time = time.time()

    if direction == 'backward':

        save_dirname = '../tmp/pong'
        save_fname = 'dt_policy.pk'
        student = DTPolicy(max_depth)
        state_transformer = get_pong_symbolic
        student = load_dt_policy(save_dirname, save_fname)
        wrapped_student = TransformerPolicy(student, state_transformer)

        for episode in range(nb_episodes):
            episode_reward = 0.
            episode_step = 0

            h_actions_DQN = []
            h_actions_DT = []
            h_states = []
            h_Q_values = []

            # Obtain the initial observation by resetting the environment.
            observation = env.reset()
            if episode == 0:
                print(f'The shape of the DQN inputs: {np.array(observation).shape}')
            done = False

            while not done:
                observation = np.array([np.array(observation)], dtype=np.uint8)
                q_values = agent.predict_q(observation)[0]
                action_DQN = agent.predict(observation)[0]
                action_DT = wrapped_student.predict(observation)[0]
                accumulated_info = {}

                h_actions_DQN.append(action_DQN)
                h_actions_DT.append(action_DT)
                h_states.append(get_pong_symbolic(observation[0]))
                h_Q_values.append(q_values)
                

                observation, reward, d, info = env.step(action_DT)
                episode_reward += reward

                if visualize:
                    env.render()

                if d:
                    done = True
                    break

                step_logs = {'action': action_DT, 'observation': observation, 'reward': reward,
                            'episode': episode, 'info': accumulated_info}
                episode_step += 1
                step += 1

            # Report end of episode.
            episode_logs = {'episode_reward': episode_reward, 'nb_steps': episode_step}
            # callback.on_episode_end(episode, episode_logs)
            print(f'Episode {episode + 1}: reward {episode_reward}, steps {episode_step}')

            save_data_hdf5(np.asarray(h_actions_DQN, dtype=np.uint8), f"ep-actions_DQN{episode}", dataset_path, 'a', 'uint8')
            save_data_hdf5(np.asarray(h_actions_DT, dtype=np.uint8), f"ep-actions_DT{episode}", dataset_path, 'a', 'uint8')
            save_data_hdf5(np.asarray(h_states, dtype=np.int8), f"ep-states_{episode}", dataset_path, 'a', 'int8')
            save_data_hdf5(np.asarray(h_Q_values, dtype=np.float64), f"ep-q_values_{episode}", dataset_path, 'a', 'float64')

        end_time = time.time()
        print(f'Total time elapsed: {end_time - start_time: .2f} sec.')
    
    elif direction == 'forward':
        for episode in range(nb_episodes):
            episode_reward = 0.
            episode_step = 0

            h_actions = []
            h_states = []
            h_Q_values = []

            # Obtain the initial observation by resetting the environment.
            observation = env.reset()
            if episode == 0:
                print(f'The shape of the DQN inputs: {np.array(observation).shape}')
            done = False

            while not done:
                observation = np.array([np.array(observation)], dtype=np.uint8)
                q_values = agent.predict_q(observation)[0]
                action = agent.predict(observation)[0]
                accumulated_info = {}

                h_actions.append(action)
                h_states.append(get_pong_symbolic(observation[0]))
                h_Q_values.append(q_values)

                observation, reward, d, info = env.step(action)
                episode_reward += reward

                if visualize:
                    env.render()

                if d:
                    done = True
                    break

                step_logs = {'action': action, 'observation': observation, 'reward': reward,
                            'episode': episode, 'info': accumulated_info}
                episode_step += 1
                step += 1

            # Report end of episode.
            episode_logs = {'episode_reward': episode_reward, 'nb_steps': episode_step}
            # callback.on_episode_end(episode, episode_logs)
            print(f'Episode {episode + 1}: reward {episode_reward}, steps {episode_step}')

            save_data_hdf5(np.asarray(h_actions, dtype=np.uint8), f"ep-actions_{episode}", dataset_path, 'a', 'uint8')
            save_data_hdf5(np.asarray(h_states, dtype=np.int8), f"ep-states_{episode}", dataset_path, 'a', 'int8')
            save_data_hdf5(np.asarray(h_Q_values, dtype=np.float64), f"ep-q_values_{episode}", dataset_path, 'a', 'float64')

        end_time = time.time()
        print(f'Total time elapsed: {end_time - start_time: .2f} sec.')

def bin_acts():
    # Parameters
    seq_len = 10
    n_rollouts = 10
    log_fname = 'pong_options.log'
    model_path = 'model-atari-pong-1/saved'
    
    # Logging
    set_file(log_fname)
    
    # Data structures
    env = get_pong_env()
    teacher = DQNPolicy(env, model_path)

    # Action sequences
    seqs = get_action_sequences(env, teacher, seq_len, n_rollouts)

    for seq, count in seqs:
        log('{}: {}'.format(seq, count), INFO)

def print_size():
    # Parameters
    dirname = 'results/run9'
    fname = 'dt_policy.pk'

    # Load decision tree
    dt = load_dt_policy(dirname, fname)

    # Size
    print(dt.tree.tree_.node_count)

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)

if __name__ == '__main__':
    # HERE, I ADDED PARSER TO SEPARATE TRAIN AND TEST WHEN I DONT WANT TO TRAIN EACH TIME I'M TESTING##################
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices = ['train', 'test','generate_data', 'DT_test','DT_SEAN','DT_Multi'], default='train')
    parser.add_argument('--direction', choices = ['forward','backward'], default = 'forward')
    parser.add_argument('--tree_type', choices=['entropy', 'gini'], default='entropy')
    parser.add_argument('--env_name', type = str, default = 'PongNoFrameskip-v4')
    parser.add_argument('--rand_seed', type = int, default = None)
    parser.add_argument('--weights', type = str, default = None)
    parser.add_argument('--nb_episodes_for_test', type = int, default = 5, help = "Number of episodes to test/evaluate. Default is 5")
    parser.add_argument('--nb_steps_for_train', type = int, default = 50000, help = "Number of steps to train/evaluate. Default is 50000")
    parser.add_argument('--visualize', type = bool, default = False)
    parser.add_argument('--plot', type = bool, default = False)
    # parser.add_argument('--callbacks', type = bool, default = False)
    parser.add_argument('--depth', type = int, default = None)
    parser.add_argument('--video',type = bool, default= False)
    parser.add_argument('--nb_trained_with',type=int,default=None,help="Number of episodes that the decision tree trained with")
    args = parser.parse_args()
    ######################################################################################################################
    if args.mode == 'train':
        learn_dt()
    elif args.mode == 'generate_data':
        generate_Data(env_name = args.env_name,rand_seed = args.rand_seed,nb_episodes_for_test = args.nb_episodes_for_test,visualize = args.visualize, direction = args.direction,max_depth = args.depth)