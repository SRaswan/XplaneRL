import gym
import argparse

import gym_xplane
# from stable_baselines3.common.policies import FeedForwardPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Custom MLP policy of three layers of size 128 each
# class CustomPolicy(FeedForwardPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomPolicy, self).__init__(*args, **kwargs,
#                                            net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])],
#                                            feature_extraction="mlp")

#model = A2C(CustomPolicy, 'LunarLander-v2', verbose=1)
#model = A2C(CustomPolicy, 'CartPole-v0', verbose=1)

def main_a2c(env):
    num_inputs = env.observation_space.shape[0]
    if type(env.action_space) is gym.spaces.discrete.Discrete:
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    # model = build_model_xlpane(observation_space_size, num_actions)
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=50, save_path="./logs/",
                                             name_prefix="a2c_rl_model")
    # Evaluate the model periodically
    # and auto-save the best model and evaluations
    # Use a monitor wrapper to properly report episode stats
    eval_env = Monitor(env)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                 log_path="./logs/", eval_freq=20,
                                 deterministic=True, render=False)

    # env = SubprocVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    model = A2C('MlpPolicy', env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=100, callback=[checkpoint_callback, eval_callback])
    model.save("a2c_xplane")

def setup_xplane_env(parser):
    parser.add_argument('--clientAddr', help='xplane host address', default='0.0.0.0')
    parser.add_argument('--xpHost', help='x plane port', default='127.0.0.1')
    parser.add_argument('--xpPort', help='client port', default=49009)
    parser.add_argument('--clientPort', help='client port', default=0)  # default=1)

    args = parser.parse_args()
    ENV_NAME = 'gymXplane-v2'
    print("Making gym env for::::::: {} ::::::::".format(ENV_NAME))
    env = gym.make(ENV_NAME)
    env.clientAddr = args.clientAddr
    env.xpHost = args.xpHost
    env.xpPort = args.xpPort
    env.clientPort = args.xpPort
    print(env.observation_space.shape)
    print(env.action_space)
    return env

if __name__ == "__main__":

    ENV_NAME = 'gymXplane-v2'
    # ENV_NAME = 'CartPole-v0'

    parse = argparse.ArgumentParser()
    # env = gym.make("CartPole-v0")
    if(ENV_NAME == 'gymXplane-v2'):
        env = setup_xplane_env(parse)
    else:
        env = gym.make(ENV_NAME)
    main_a2c(env)