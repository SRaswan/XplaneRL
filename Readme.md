----------------------------
### X-Plane Reinforcement Learning Agents
-----------------------------
Advantage Actor-Critic Method
-----------------------------
The A2C agent in this repository will be used to assess the capability of using deep reinforcement learning for developing an AI agent that autonomously flies a plane. The goal of this project is to evaluate the practicality of Deep Reinforcement Learning for pilotless planes. The reinforcement learning agent would need to achieve safe autonomous flight in a realistic flight simulator called X-Plane. Using an Advantage Actor-Critic model with a custom policy, the machine learning agent attempts to maintain a constant heading and altitude for a plane in mid-flight. 

The agent interacts with X-Plane by exploring and training from the environment, using X-Plane plugins and a comprehensive gym environment built on OpenAI’s gym standards. The X-Plane gym environment is used to observe and control the aircraft in real time. The agent and environment base communicates with the gym environment to determine the best action in the 4 action space parameters, using the 11 state space observable parameters to train. It then determines what action to take by manipulating the continuous action space of the latitudinal stick, longitudinal stick, rudder pedals, or throttle, based on the gym-based rewards. The FlyWithLua plugin for X-Plane is used for resetting the environment each episode when the plane crashes or its engine is turned off, while the X-Plane Connect plugin is used for programmatically retrieving the state space parameters from the plane as it moves in the simulator. The action space parameters are continuous and the agent uses activation functions from TensorFlow, a python reinforcement learning toolset for algorithms like Advantage Actor-Critic models. The critic in the Actor-Critic agent estimates the Q value function for the actions and the actor updates the policy gradient based on the critic. 

By analyzing the results, the pilotless plane is able to fly without crashing when using optimized A2C policies, allowing the plane to keep heading but with few instabilities. In the near future, the Deep Reinforcement Learning agent may have closer results to human pilots, however, more research for convolutional neural networks is needed to support fully autonomous planes, additionally accounting for taxiing, takeoff, landing, and air traffic. Using computer vision, we can improve the model by including additional state space observable parameters with an ensemble of machine learning models.

Look at examples and https://github.com/adderbyte/GYM_XPLANE_ML for directions on how to run the agent with X-Plane.

Future examples:
---------------------------
PPO Agent
___________________________

___________________________
DQN Agent
___________________________
