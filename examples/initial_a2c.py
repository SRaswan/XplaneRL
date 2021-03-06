import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import gym
import gym_xplane
import argparse

# hyperparameters
hidden_size = 256
learning_rate = 3e-4 #1e-8 #

# Constants
GAMMA = 0.99
num_steps = 300
max_episodes = 3000


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        # print(type(state))
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        # policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)
        policy_dist = F.sigmoid(self.actor_linear2(policy_dist))
        # policy_dist = F.relu(self.actor_linear2(policy_dist))
        print("policy_dist: {}".format(policy_dist))
        return value, policy_dist

    # def state_array(self, state):
    #     state2  = []
    #     for s in state:
    #         if type(s) is tuple:
    #             state2.append(s[0])
    #         else:
    #             state2.append(s)
    #     return np.array(state2)

    def our_func(self):
        act_1 = tf.clip_by_value(dist.sample(1), env.action_space.low[0], env.action_space.high[0])
        act_2 = tf.clip_by_value(dist.sample(1), env.action_space.low[1], env.action_space.high[1])
        act_3 = tf.clip_by_value(dist.sample(1), env.action_space.low[2], env.action_space.high[2])
        act_4 = tf.clip_by_value(dist.sample(1), env.action_space.low[3], env.action_space.high[3])
        self.action = tf.concat([act_1, act_2, act_3, act_4], 0)


def a2c(env):
    num_inputs = env.observation_space.shape[0]
    if type(env.action_space) is gym.spaces.discrete.Discrete:
        num_outputs = env.action_space.n
    else:
        num_outputs = env.action_space.shape[0]
    # num_outputs = env.action_space.n

    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for steps in range(num_steps):
            # state = actor_critic.state_array(state)
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()[0]


            action = dist
            # dist = dist[0]
            # action = np.random.choice(num_outputs, p=np.squeeze(dist))
            # action = policy_dist
            # log_prob = torch.log(policy_dist.squeeze(0)[action])
            log_prob = torch.log(policy_dist.squeeze(0))
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, info = env.step(action) # changed action to dist
            # new_state = actor_critic.state_array(new_state)
            print("action1, new_state, reward, done :::: {} :: {} :: {} :: {}".format(action, new_state, reward, done))

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state

            if done or steps == num_steps - 1:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:
                    sys.stdout.write(
                        "episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode,
                                                                                                  np.sum(rewards),
                                                                                                  steps,
                                                                                                  average_lengths[
                                                                                                      -1]))
                break

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()

def setup_xplane_env(parser):
    parser.add_argument('--clientAddr', help='xplane host address', default='0.0.0.0')
    parser.add_argument('--xpHost', help='x plane port', default='127.0.0.1')
    parser.add_argument('--xpPort', help='client port', default=49009)
    parser.add_argument('--clientPort', help='client port', default=0)  # default=1)

    args = parser.parse_args()
    ENV_NAME = 'gymXplane-v2'
    print("Making gym env for::::::: {} ::::::::".format(ENV_NAME))
    env = gym.make('gymXplane-v2')
    env.clientAddr = args.clientAddr
    env.xpHost = args.xpHost
    env.xpPort = args.xpPort
    env.clientPort = args.xpPort
    print(env.observation_space.shape)
    print(env.action_space)
    return env

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    # env = gym.make("CartPole-v0")
    env = setup_xplane_env(parse)
    a2c(env)