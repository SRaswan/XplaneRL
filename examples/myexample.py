import argparse

import gym_xplane
# import p3xpc
import gym
from gym_xplane.envs.xplane_envBase import XplaneEnv


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


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

    # env.seed(123)
    agent = RandomAgent(env.action_space)

    episodes = 0
    length = 0
    # flag = True
    final_stop = False
    while episodes < 500:
        obs = env.reset()
        done = False
        while not done:
            action = agent.act()
            obs, reward, done, info = env.step(action)

            # print(obs, reward, done, info)
            print("Episode", episodes, ": ", reward)
            on_ground = XplaneEnv.CLIENT.getDREF("sim/flightmodel2/gear/on_ground")
            #print("I am on ground: flag:{} onground:{}::: {}".format(done, on_ground, on_ground[0]))
            on_crash = XplaneEnv.CLIENT.getDREF("sim/flightmodel/engine/ENGN_running")
            #print(
                #"I am in crasher:flag:{} onENGGrunning:{}::: {}".format(done, on_crash, on_crash[0]))
            if on_crash[0] == 0.0:
                print("we crashed..................XXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx.....................")

                while on_crash[0] == 0.0:
                    on_crash = XplaneEnv.CLIENT.getDREF("sim/flightmodel/engine/ENGN_running")
                    print(on_crash[0])
                    continue
                break

            #if on_ground[0] == 1.0:
                #print("we are on ground.................._______________________.....................")
                #final_stop = True


            #print(done, info)
        # if final_stop:
        #     break
        episodes += 1
    env.close()
