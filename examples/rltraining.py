from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.rl.agents.dqn import DQNAgent
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import numpy as np
import tensorflow as tf
import argparse

import gym_xplane
#import p3xpc
import gym
import os
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # client = p3xpc.XPlaneConnect()

    # parser.add_argument('--client', help='client address',default=client)
    parser.add_argument('--clientAddr', help='xplane host address', default='0.0.0.0')
    parser.add_argument('--xpHost', help='x plane port', default='127.0.0.1')
    parser.add_argument('--xpPort', help='client port', default=49009)
    parser.add_argument('--clientPort', help='client port', default=0)  # default=1)

    args = parser.parse_args()

    env = gym.make('gymXplane-v2')
    env.clientAddr = args.clientAddr
    env.xpHost = args.xpHost
    env.xpPort = args.xpPort
    env.clientPort = args.xpPort

    print(env.observation_space.shape)
    print(env.action_space)

    learning_rate = 0.005
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16, activation="relu"))
    model.add(Activation('relu'))
    model.add(Dense(4, activation="linear"))
    model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=learning_rate))
    # model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    # model.add(Dense(16))
    # model.add(Activation('relu'))
    # model.add(Dense(7))
    # model.add(Activation('linear'))
    # model.add(Flatten(input_shape=(1,) + env.action_space.shape))
    print(model.summary())

    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    # dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
    #                target_model_update=1e-2, policy=policy)]
    nb_actions = 4
    dqn = DQNAgent(model=model, policy=policy, nb_actions=nb_actions, memory=memory)

    # (self, model, policy=None, test_policy=None, enable_double_dqn=False, enable_dueling_network=False,
    # dueling_type='avg', * args, ** kwargs):

    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)



    env.seed(123)
