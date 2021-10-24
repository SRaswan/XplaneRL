import numpy as np
import gym
import gym_xplane

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent

from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger

import argparse

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

ENV_NAME = 'CartPole-v0'
# ENV_NAME = "MountainCar-v0"
#ENV_NAME = 'gymXplane-v2'


def build_model(state_size, num_actions):
    input = Input(shape=(1, state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
    return model


def build_model_xlpane(observation_space_size, num_actions):
    learning_rate = 0.005
    model = Sequential()
    # model.add(Flatten(input_shape=(1,) + observation_space_size))
    model.add(Flatten(input_shape=(1, observation_space_size)))
    model.add(Dense(16, activation="relu"))
    model.add(Activation('relu'))
    model.add(Dense(num_actions, activation="linear"))
    # model.compile(loss="mean_squared_error",
    #               optimizer=Adam(lr=learning_rate))
    print(model.summary())
    return model


def build_callbacks(env_name):
    checkpoint_weights_filename = 'dqn/dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = 'dqn/dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks


def setup_xplane_env(parse):
    parser.add_argument('--clientAddr', help='xplane host address', default='0.0.0.0')
    parser.add_argument('--xpHost', help='x plane port', default='127.0.0.1')
    parser.add_argument('--xpPort', help='client port', default=49009)
    parser.add_argument('--clientPort', help='client port', default=0)  # default=1)

    args = parser.parse_args()
    # env = gym.make('gymXplane-v2')
    print("Making gym env for::::::: {} ::::::::".format(ENV_NAME))
    env = gym.make(ENV_NAME)
    env.clientAddr = args.clientAddr
    env.xpHost = args.xpHost
    env.xpPort = args.xpPort
    env.clientPort = args.xpPort
    print(env.observation_space.shape)
    print(env.action_space)
    return env


def main(parser):
    # Get the environment and extract the number of actions.
    # env = gym.make(ENV_NAME)

    env = setup_xplane_env(parser)

    np.random.seed(42)
    env.seed(42)
    # num_actions for any other gym
    num_actions = 0
    # num_actions for gym_xplane (ENV_NAME = 'gymXplane-v2')
    if type(env.action_space) is gym.spaces.discrete.Discrete:
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    observation_space_size = env.observation_space.shape[0]

    print("num_actions:{}, observation_space_size:{}".format(num_actions, observation_space_size))

    model = None
    if(ENV_NAME == 'gymXplane-v2'):
        model = build_model_xlpane(observation_space_size, num_actions)
    else:
        model = build_model(observation_space_size, num_actions)

    memory = SequentialMemory(limit=50000, window_length=1)

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=10000)

    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    learning_rate = 1e-3 #0.005
    dqn.compile(optimizer=Adam(lr=learning_rate), metrics=['mae'])

    callbacks = build_callbacks(ENV_NAME)

    nb_steps = 50
    dqn.fit(env, nb_steps=nb_steps,
            visualize=False,
            nb_max_start_steps=2,
            verbose=2,
            callbacks=callbacks)
    print("fit finished after {} steps".format(nb_steps))
    # After training is done, we save the final weights.
    # dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    # dqn.test(env, nb_episodes=50, visualize=True)
    dqn.test(env, nb_episodes=50, visualize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser)
