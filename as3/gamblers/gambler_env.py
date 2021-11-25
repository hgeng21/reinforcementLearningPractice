"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018,
  University of Alberta.
  Gambler's problem environment using RLGlue.
"""
from rl_glue import BaseEnvironment
import numpy as np
# import random
import random


class GamblerEnvironment(BaseEnvironment):
    """
    Slightly modified Gambler environment -- Example 4.3 from
    RL book (2nd edition)

    Note: inherit from BaseEnvironment to be sure that your Agent class implements
    the entire BaseEnvironment interface
    """

    def __init__(self):
        """Declare environment variables."""
        self.p_head = None
        self.gamma = None
        self.theta = None
        self.state_values = None
        self.policy = None
        self.state = None

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """
        self.p_head = 0.55
        self.gamma = 1
        self.theta = 0.001
        self.state_action = np.zeros((100,50),dtype="float")
        self.state_action_num = np.zeros((100,50),dtype="float")
        self.state = np.zeros(1)
        ##self.policy = np.zeros(100,1)
        
    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - numpy array
        Hint: Sample the starting state necessary for exploring starts and return.
        """
        ##self.state_action = np.zeros((100,51),dtype="float")
        ##self.state_action_num = np.zeros((100,51),dtype = "float")
        random_int = random.randint(1,99)
        ##self.policy = np.zeros(100,1)
        self.state = np.asarray([random_int])
        ##print("== state = {}".format(self.state[0]))
        return self.state
        

    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - numpy array - terminal - boolean
        Hint: Take a step in the environment based on dynamics; also checking for action validity in
        state may help handle any rogue agents.
        """
        ##if action<1: print("===========================")
        
        coin_flip = np.random.random(1)[0]
        
        # get new state
        if (coin_flip <= self.p_head):
            self.state[0] = self.state[0]+action
        else:
            self.state[0] = self.state[0]-action
            
        ##print("--step-state = {}".format(self.state[0]))
        # get reward and terminal
        if (int(self.state) == 100):
            reward = float(1)
            ##self.state = None
            terminal = True
        elif (int(self.state) == 0):
            reward = float(0)
            ##self.state = None
            terminal = True
        else:
            reward = float(0)
            terminal = False
        
        return reward, self.state, terminal

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass
