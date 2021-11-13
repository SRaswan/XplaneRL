
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import gym
import matplotlib.pyplot as plt
import numpy as np
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
from collections import deque
import random
import os
import gym_xplane
from tf_slim.layers import layers as _layers;

from lib import plotting
import matplotlib
matplotlib.style.use('ggplot')

NUM_EPISODES = 10
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.00046415888336127773
BATCH_SIZE = 32

GAMMA = 0.98

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


env = gym.make('gymXplane-v2')

run_id = np.random.randint(10000)
#run_id = "no_render"
env.reset()

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)


featurizer = sklearn.pipeline.FeatureUnion([
		("rbf1", RBFSampler(gamma=5.0, n_components=100)),
		("rbf2", RBFSampler(gamma=2.0, n_components=100)),
		("rbf3", RBFSampler(gamma=1.0, n_components=100)),
		("rbf4", RBFSampler(gamma=0.5, n_components=100))
])


featurizer.fit(scaler.transform(observation_examples))

#memory = deque(maxlen=2000)

def featurize_state(state):
	scaled = None
	featurized = None
	try:
		scaled = scaler.transform([state])
		#print("scaled", scaled)
		# featurized = featurizer.transform(scaled)
	except:
		print("stated",state)
	
	
	featurized = featurizer.transform(scaled)
	return featurized



class Policy():
	def __init__(self,lr,entropy_scalar):
		# Initialize policy graph
		# with tf.variable_scope("policy"):
		with tf.compat.v1.variable_scope("policy"):
			self.state_placeholder = tf.compat.v1.placeholder(tf.float32,shape=(None,400),name="state")
			self.target_placeholder = tf.compat.v1.placeholder(tf.float32,shape=(None),name = "target")

			# self.state_placeholder = tf.placeholder(tf.float32,shape=(None,400),name="state")
			# self.target_placeholder = tf.placeholder(tf.float32,shape=(None),name = "target")
			self.learning_rate = lr
			self.entropy_scalar = entropy_scalar

			mu = tf.squeeze(
				# tf.contrib.layers.fully_connected(
				_layers.fully_connected(
				# tf.keras.layers.Dense(
				# units=1,
				inputs=self.state_placeholder,
				num_outputs=1,
				activation_fn=None,
				# kernel_initializer='glorot_uniform'
				weights_initializer=tf.compat.v1.keras.initializers.glorot_normal()
				# weights_initializer=_layers.xavier_initializer()
			))

			self.sigma = tf.squeeze(
				_layers.fully_connected(
				# tf.keras.layers.Dense(
				# units=1,
				inputs=self.state_placeholder,
				num_outputs=1,
				activation_fn=None,
				# kernel_initializer='glorot_uniform'
				# weights_initializer=tf.contrib.layers.xavier_initializer()
				weights_initializer = tf.compat.v1.keras.initializers.glorot_normal()
			))
			
			self.sigma = tf.nn.sigmoid(self.sigma) 

			dist = tf.compat.v1.distributions.Normal(loc=mu,scale=self.sigma)
			# dist = tf.distributions.Normal(loc=mu,scale=self.sigma)
			#self.action = tf.clip_by_value(dist.sample(1), env.action_space.low[0], env.action_space.high[0])
			act_1 =  tf.clip_by_value(dist.sample(1), env.action_space.low[0], env.action_space.high[0])
			act_2 =  tf.clip_by_value(dist.sample(1), env.action_space.low[1], env.action_space.high[1])
			act_3 =  tf.clip_by_value(dist.sample(1), env.action_space.low[2], env.action_space.high[2])
			act_4 =  tf.clip_by_value(dist.sample(1), env.action_space.low[3], env.action_space.high[3])
			self.action = tf.concat([act_1,act_2,act_3,act_4],0)
			#print('action',self.action)
			#act_1 =  tf.clip_by_value(dist.sample(1), env.action_space.low[0], env.action_space.high[0])

			# Use adam
			update = - (dist.log_prob(self.action) * self.target_placeholder) - (self.entropy_scalar * dist.entropy())
			# self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
			self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
			self.updates = self.optimizer.minimize(update)


			"""
			policy_log = dist.log_prob(self.action)
			policy_log += self.entropy_scalar * dist.entropy()
			self.updates = []
			for v in tf.trainable_variables(scope="policy"):
				grad = tf.gradients(policy_log,v)[0]
				update = self.learning_rate * grad * self.target_placeholder
				self.updates.append(tf.assign_add(v, update, name='update'))
			"""
			
	def update(self,sess,state,target,act):
		state = featurize_state(state)
		sess.run(self.updates,feed_dict={self.state_placeholder:state,self.target_placeholder:target,self.action:act})
		return 1
	
	def get_action(self,sess,state):
		state = featurize_state(state)
		return sess.run(self.action,feed_dict={self.state_placeholder:state})
	


class Critic():
	# Maps state to action
	def __init__(self,lr):
		self.state_placeholder = tf.compat.v1.placeholder(tf.float32,shape=(None,400))
		self.target_placeholder = tf.compat.v1.placeholder(tf.float32,shape=(None))
		self.learning_rate = lr
		self.out = tf.squeeze(
			_layers.fully_connected(
			# tf.compat.v1.layers.dense(
			inputs=self.state_placeholder,
			num_outputs=1,
			activation_fn=None,
			# weights_initializer=tf.contrib.layers.xavier_initializer()
			weights_initializer = tf.compat.v1.keras.initializers.glorot_normal()
		))
		
		# We get the value of being in a state and taking an action

		self.loss = tf.math.squared_difference(self.out,self.target_placeholder)
		# self.loss = tf.squared_difference(self.out,self.target_placeholder)

		# optimizer = tf.train.AdamOptimizer(self.learning_rate)
		optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
		self.step = optimizer.minimize(self.loss)
		
		
	def predict_value(self,sess,state):
		state = featurize_state(state)
		return sess.run(self.out,feed_dict={self.state_placeholder:state})
		
	def update(self,sess,state,target):
		state = featurize_state(state)		
		_,loss = sess.run([self.step,self.loss], feed_dict={self.state_placeholder:state,self.target_placeholder:target})
		return loss




def actor_critic(num_episodes,learning_rate_critic,learning_rate_actor,entropy_scalar):
	# Keeps track of useful statistics
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))


	run_id = np.random.randint(10000)

	# tf.reset_default_graph()
	tf.compat.v1.reset_default_graph()

	# Create our actor and critic
	actor = Policy(lr=learning_rate_actor,entropy_scalar=entropy_scalar)
	critic = Critic(lr=learning_rate_critic)
	
	sess = tf.Session() # tf.compat.v1.Session #
	
	sess.run(tf.compat.v1.global_variables_initializer()) #tf.global_variables_initializer())
	filewriter = tf.summary.FileWriter(logdir="logs/" + str(run_id), graph=sess.graph)
	
	steps = 0

	scores = []
	
	for e in range(num_episodes):
		
		state = env.reset()
		avg_critic_loss = 0
		avg_actor_loss = 0
		i=0
		score = 0

		for g in range(num_episodes):
		# while True:
			
			# Take a step
			#env.render()
			action = actor.get_action(sess,state)
			#print('action', action)
			next_state,reward,done,_ = env.step(action)
		
			# Append transition
			#memory.append([state,action,reward,next_state,done])

			#sample = random.sample(memory,1)

			s_state, s_action, s_reward, s_next_state, s_done = [state,action,reward,next_state,done]

			critic_target = s_reward + GAMMA * critic.predict_value(sess,s_next_state)
			td_error = critic_target - critic.predict_value(sess,s_state)

			critic_loss = critic.update(sess,s_state,critic_target)
			actor_loss = actor.update(sess,s_state,td_error,s_action)
				
			#action_value_summary = tf.Summary(value=[tf.Summary.Value(tag='Action Value',simple_value=action)])
			#filewriter.add_summary(action_value_summary,steps)
			#print('reward', s_reward)

			i += 1
			steps += 1
			#avg_actor_loss += 1
			#avg_critic_loss += critic_loss
			# Update statistics
			stats.episode_rewards[e] += reward
			stats.episode_lengths[e] = g

			score += reward
			#print("Episode: " + str(e) + " Score: " + str(score))
			print("Episode: " + str(e) + " Score: " + str(score))

			if done:
				break
			
			state = next_state
		#print("Episode: " + str(e) + " Score: " + str(score))
		print("Episode: " + str(e) + " Score: " + str(score))
		scores.append(score)
		#reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Reward',simple_value=score)])
		#filewriter.add_summary(reward_summary,e)

	plotting.plot_episode_stats(stats, smoothing_window=10)
	return scores

tf.reset_default_graph()
# s = actor_critic(num_episodes=100000000000,learning_rate_critic=1e-8,learning_rate_actor=1e-8,entropy_scalar=2.6366508987303556e-02)
s = actor_critic(num_episodes=10,learning_rate_critic=1e-8,learning_rate_actor=1e-8,entropy_scalar=2.6366508987303556e-02)



print(s)
