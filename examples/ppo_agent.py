import argparse

import gym_xplane
# import p3xpc
import gym
import os
import time

# from stable_baselines3.common.policies import MlpPolicy, LstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

ENV_NAME = 'gymXplane-v2'

def setup_xplane_env(parse):
    log_dir = "./gym/{}".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)

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
    env = Monitor(env, log_dir, allow_early_resets=True)
    return env



def main(parser):
    env = setup_xplane_env(parser)

    # env.seed(123)
    n_cpu = 1
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    # env = make_vec_env("CartPole-v1", n_envs=4)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main(parser)
