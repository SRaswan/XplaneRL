import gym
from scipy.spatial.distance import pdist, squareform

import gym_xplane.xpc as xp
import gym_xplane.parameters as parameters
import gym_xplane.space_definition as envSpaces
import numpy as np
import itertools
from time import sleep, clock

class initial:

    def connect( clientAddr, xpHost, xpPort  , clientPort, timeout ,max_episode_steps):
            return xp.XPlaneConnect(clientAddr,xpHost,xpPort,clientPort,timeout ,max_episode_steps)

class XplaneEnv(gym.Env):

    def __init__(self,clientAddr, xpHost, xpPort  , clientPort, timeout=3000 ,max_episode_steps=303,test=False):
        #CLIENT = client
        XplaneEnv.CLIENT = None
        #print(parameters)
        envSpace = envSpaces.xplane_space()
        
        
        self.ControlParameters = parameters.getParameters()
        self.action_space = envSpace._action_space()
        self.observation_space = envSpace._observation_space()
        #self.episode_steps = 0
        self.ControlParameters.episodeStep =0
        self.max_episode_steps = max_episode_steps
        self.statelength = 10
        self.actions = [0,0,0,0]
        self.test=test
        try:
            XplaneEnv.CLIENT = initial.connect(clientAddr,xpHost,xpPort,clientPort,timeout ,max_episode_steps)
        except:
            print("connection error. Check your paramters")
        print('I am client', XplaneEnv.CLIENT )
        
         
    

    def close(self):
        XplaneEnv.CLIENT.close()
    
    def rewardCalcul(self,target_state,xplane_state,sigma=0.45):
        '''
        input : target state (a list containing the target heading, altitude and runtime)
                xplane_state(a list containing the aircraft heading , altitude at present timestep, and the running time)
                Note: if the aircraft crashes then the run time is small, thus the running time captures crashes
        output: Gaussian kernel similarîty between the two inputs. A value between 0 and 1



        '''
        

        data = np.array([target_state,xplane_state])
       
        pairwise_dists = pdist(data,'cosine')
        #print('pairwise distance',pairwise_dists)
        similarity = np.exp(-pairwise_dists ** 2 / sigma ** 2)

        return pairwise_dists


    def step(self, actions):
     

        self.test=False # if true, only test paramters returned. for Model tesing 
        self.ControlParameters.flag = False # for synchronisation of training
        #self.ControlParameters.episodeReward = 0  # for reward in each episode
        #self.ControlParameters.totalReward = 0 # reward for final episode aimed at penalizing crash if it crashes finally 
        
        reward = 0.
        perturbationAllowed = [3.5,15] # pertubation allowed on altitude and heading
        actions_ = []
        
        j=0  # getting simulaion timing measurement
        
        try:
            
            #############################################

            # **********************************************
            ### NOTE:  One could Disable the stability augmentation in XPlane in other to run the simulation without sending pause commands
            #         In that case comment out the send XplaneEnv.CLIENT.pauseSim(False).
            #         Previous action is compared to present action to check that after sending an action the action  
            #         on the controls in the next iteration is same as that which was sent. 
            #         If this is not true then stability augmentation is acting on the controls too -- this gives very unstable
            #         and non smooth flight and the agent will never be able to learn due to constant pertubation
            #         of state by the augmentation system
            #*************************************************


            #############################################


            #############################################
            # chck pevious action is same as the one on the controls
            #print("prevous action",self.actions) # prvious ation
            #print("action on ctrl ...",XplaneEnv.CLIENT.getCTRL()) # action on control surface
            # if this is not sae then there are unaccounted forcs that could affct ainin
            # cnage the sleep time ater actio is sent in odr to ensure that training is synchronise
            # an the actins prined hee are same
            #############################################
            
            #############################################
            i=clock() # get the time up till now
            XplaneEnv.CLIENT.pauseSim(False) # unpause x plane simulation
            XplaneEnv.CLIENT.sendCTRL(actions) # send action
            # sleep(0.0003)  # sleep for a while so that action is executed
            sleep(0.03)#sleep for a while so that action is executed

            self.actions = actions  # set the previous action to current action. 
                                    # This will be compared to action on control in next iteraion
            XplaneEnv.CLIENT.pauseSim(False) #pause simulation so that no other action acts on he aircaft
            # XplaneEnv.CLIENT.pauseSim(True)  pause simulation so that no other action acts on he aircaft
            j=clock() # get the time now, i-j is the time at which the simulation is unpaused and action exeuted
            # fom this point the simulation is paused so that we compute reward and state-action value
            ################################################# 
            
            #################################################
            # tenporary variable for holding state values
            state = [];
            state14 = []
            ################################################
            
            #################################################
            # get the state variabls here . The parameter file has all the required variables
            # we only need to call the client interface and get parameters defined as stateVariable
            # in parameter file as below
            stateVariableTemp = XplaneEnv.CLIENT.getDREFs(self.ControlParameters.stateVariable) 
            # the client interface automaically gets the position paameters
            self.ControlParameters.stateAircraftPosition = list(XplaneEnv.CLIENT.getPOSI());
            # Remove brackets from state variable and store in the dictionary
            self.ControlParameters.stateVariableValue = [i[0] for i in stateVariableTemp]
            # combine the position and other state parameters in temporary variable here
            state =  self.ControlParameters.stateAircraftPosition + self.ControlParameters.stateVariableValue
            ########################################################

            ########################################################
            # **********************************************reward parametera**********************
            # rewardVector : distance to the target . This is distance along the heading and altitude.
            # this is set to motivate he agent to mov forad in time . Accumulate disance
            # rewardVector = XplaneEnv.CLIENT.getDREF(self.ControlParameters.rewardVariable)[0][0]
            rewardVector = XplaneEnv.CLIENT.getDREF(self.ControlParameters.rewardVariable)[0]
            headingReward = 164 # the heading target
            minimumAltitude= 12000 # Targrt Altitude
            minimumRuntime = 210.50 # Target runtime
            # ****************************************************************************************

            # *******************************other training parameters ******************
            # consult https://www.siminnovations.com/xplane/dataref/index.php for full list of possible parameters
            # P = XplaneEnv.CLIENT.getDREF("sim/flightmodel/position/P")[0][0] # moment P
            # Q = XplaneEnv.CLIENT.getDREF("sim/flightmodel/position/Q")[0][0] # moment Q
            # R = XplaneEnv.CLIENT.getDREF("sim/flightmodel/position/R")[0][0]  # moment R
            # hstab = XplaneEnv.CLIENT.getDREF("sim/flightmodel/controls/hstab1_elv2def")[0][0] # horizontal stability : not use for now
            # vstab = XplaneEnv.CLIENT.getDREF("sim/flightmodel/controls/vstab2_rud1def")[0][0] # vertical stability : not used for now
            P = XplaneEnv.CLIENT.getDREF("sim/flightmodel/position/P")[0] # moment P
            Q = XplaneEnv.CLIENT.getDREF("sim/flightmodel/position/Q")[0] # moment Q
            R = XplaneEnv.CLIENT.getDREF("sim/flightmodel/position/R")[0]  # moment R
            hstab = XplaneEnv.CLIENT.getDREF("sim/flightmodel/controls/hstab1_elv2def")[0] # horizontal stability : not use for now
            vstab = XplaneEnv.CLIENT.getDREF("sim/flightmodel/controls/vstab2_rud1def")[0] # vertical stability : not used for now
            # ******************************************************************************
            ################################################################################

            ##############################################################################
            # check that all parameters have been collected. This is done by checking the legth of list
            # It is possible because of network failure that all parameters are not retrieved on UDP
            # In that case previous state / last full state will be used. check the except of this try.
            if len(state) == self.statelength: # this should be true if len(state) is 10

                self.ControlParameters.state14['roll_rate'] = P #  The roll rotation rates (relative to the flight)
                self.ControlParameters.state14['pitch_rate']= Q    # The pitch rotation rates (relative to the flight)
                self.ControlParameters.state14['altitude']= state[2] #  Altitude 
                self.ControlParameters.state14['Pitch']= state[3] # pitch 
                self.ControlParameters.state14['Roll']= state[4]  # roll
                self.ControlParameters.state14['velocity_x']= state[6] # local velocity x  OpenGL coordinates
                self.ControlParameters.state14['velocity_y']= state[7] # local velocity y  OpenGL coordinates              
                self.ControlParameters.state14['velocity_z']= state[8] # local velocity z   OpenGL coordinates
                self.ControlParameters.state14['delta_altitude']= abs(state[2] - minimumAltitude) # difference in altitude
                self.ControlParameters.state14['delta_heading']= abs(state[5] - headingReward) # difference in heading
                self.ControlParameters.state14['yaw_rate']= R # The yaw rotation rates (relative to the flight)
                if self.test :
                    # if testing use append longitude and latitude as  the state variable
                    # The intuition for this is that during testing we need lat and long to be able to project the position of the
                    # aircarft in space. Thus [lat,long,altitude will be relevant]. Lat Long are not relevant during training
                    state.append(R) # if connection fails append R to make sure state is not empty
                    state14 = state # state variable this inclue lat long for ploting 

                else:
                    # lat long have been overriden. The dictionary above is used as normal during training
                    state14 = [i for i in self.ControlParameters.state14.values()]
            ######################################################################
            
            ###########################################################################
            # *******************************reward computation ******************
            # parameters required for reward
            # time is not used here
            # timer =  XplaneEnv.CLIENT.getDREF(self.ControlParameters.timer2)[0][0] # running time of simulation
            timer = XplaneEnv.CLIENT.getDREF(self.ControlParameters.timer2)[0]  # running time of simulation

            target_state = [abs(headingReward),minimumAltitude,0.25]  # taget situation -heading, altitude, and distance
            xplane_state = [ abs(state[5]),state[2],rewardVector]  # present situation -heading, altitude, and distance 
            # if the heading and altitude are within small pertubation set good reward othewise penalize it.
            if  (abs( abs(state[5])-headingReward)) < perturbationAllowed[0] and (state[2]-minimumAltitude) < perturbationAllowed[1]:
                reward = self.rewardCalcul(target_state,xplane_state,sigma=0.85)[0]
                self.ControlParameters.episodeReward = reward
            else:
                reward = self.rewardCalcul(target_state,xplane_state)
                self.ControlParameters.episodeReward = -reward[0]
            self.ControlParameters.episodeStep += 1
            #############################################################################

            ###########################################################################
            # end of episode setting
            # detect crash and penalize the agênt
            # if crash add -3 otherwose reward ramin same
            # if XplaneEnv.CLIENT.getDREFs(self.ControlParameters.on_ground)[0][0] >= 1 or XplaneEnv.CLIENT.getDREFs( self.ControlParameters.crash)[0][0] <=0:
            if XplaneEnv.CLIENT.getDREFs(self.ControlParameters.on_ground)[0][0] >= 1 or XplaneEnv.CLIENT.getDREFs(self.ControlParameters.crash)[0][0] <= 0:
                self.ControlParameters.flag = True # end of episode flag
                self.ControlParameters.reset = False # this checks that duplicate penalizaion is not aplied especiall when sim 
                                                     # frequency is high
                if self.ControlParameters.episodeStep <= 1.:
                    self.ControlParameters.reset = True
                    #print('reset', self.ControlParameters.reset )
                elif not self.ControlParameters.reset:
                    self.episodeReward -=  3.
                    print("crash", self.ControlParameters.episodeReward)
                    self.ControlParameters.totalReward = self.ControlParameters.episodeReward
                    #self.ControlParameters.totalReward -= 1
                else: 
                    pass
            # set flag to true if maximum steps has been achieved. Thus episode is finished.
            # set the maximum episode step to the value you want    
            elif self.ControlParameters.episodeStep > self.max_episode_steps:
                self.ControlParameters.flag = True
                self.ControlParameters.totalReward  = self.ControlParameters.episodeReward
            ###########################################################################
    
            ###########################################################################
            # reset the episode paameters if Flag is true. (since episode has terminated)
            # flag is synchonised with XPlane enviroment
            if self.ControlParameters.flag:
                reward = self.ControlParameters.totalReward 
                #print(reward, 'reward' , self.ControlParameters.totalReward, self.ControlParameters.episodeReward )
                #self.ControlParameters.flag = True
                self.ControlParameters.totalReward=0.
                self.ControlParameters.episodeStep = 0
                #self.episode_steps=0
                self.actions = [0,0,0,0] 
            else:
                reward = self.ControlParameters.episodeReward
            ###########################################################################
            

        except:
            reward = self.ControlParameters.episodeReward
            self.ControlParameters.flag = False
            self.ControlParameters.state14 =  self.ControlParameters.state14
            if self.test:
                state.append(0)
                state14 = state
            else:
                state14 = [i for i in self.ControlParameters.state14.values()]
        #print(reward, 'reward' , self.ControlParameters.totalReward, self.ControlParameters.episodeReward )
        q=clock() # end of loop timer 
        #print("pause estimate", q-j)
        return  np.array(state14),reward,self.ControlParameters.flag,self._get_info() #self.ControlParameters.state14


   

  

    def _get_info(self):
        """Returns a dictionary contains debug info"""
        return {'control Parameters':self.ControlParameters, 'actions':self.action_space }

    def render(self, mode='human', close=False):
        pass


    def reset(self):
        """
        Reset environment and setup for new episode.
        Returns:
            initial state of reset environment.
        """
        self.actions = [0,0,0,0] 
        self.ControlParameters.stateAircraftPosition = []
        self.ControlParameters.stateVariableValue = []
        self.ControlParameters.episodeReward  = 0.
        self.ControlParameters.totalReward  = 0.
        self.ControlParameters.flag = False
        #self.episode_steps = 0
        self.ControlParameters.episodeStep = 0
        self.ControlParameters.state14 = dict.fromkeys(self.ControlParameters.state14.keys(),0)
        #print(self.ControlParameters.state14)
        #stater  = [0,4]
        #state = {'value':  np.array([stater[0], stater[1]]).shape(2,)}
        #print(state)
        #val = 1
        state = np.zeros(11)
        
        return state # self.ControlParameters.state14
