
from rl_glue import BaseEnvironment
import numpy as np
import random


class GridEnvironment(BaseEnvironment):

    def __init__(self):
        """Declare environment variables."""
        self.size = None
        self.epsilon = None
        self.start_pt = None
        self.state = None
        self.goal_pt = None
        self.obs = None # store obstacles

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """
        self.size = [5,8] # 9x6
        self.epsilon = 0.1
        self.start_pt = np.asarray((3,0))
        self.state = None
        self.goal_pt = np.asarray((5,8))
        self.obs = set(((1,5),(2,2),(3,2),(3,7),(4,2),(4,7),(5,7)))

        
        
    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - numpy array
        """
        self.start_pt = np.asarray((3,0))
        self.state = self.start_pt
        return self.state
        

    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - numpy array - terminal - boolean
        """
        
        # get new state
        if (action == 0):   # 1: up
            self.state[0] = self.state[0] + 1 
            # check obstacles:
            if tuple(self.state) in self.obs: 
                self.state[0] = self.state[0] - 1    
            ##print("[up]")
        elif (action == 1): # 2: down
            self.state[0] = self.state[0] - 1
            # check obstacles:
            if tuple(self.state) in self.obs: 
                self.state[0] = self.state[0] + 1
            ##print("[down]")
        elif (action == 2): # 3: left
            self.state[1] = self.state[1] - 1
            # check obstacles:
            if tuple(self.state) in self.obs:  
                self.state[1] = self.state[1] + 1
            ##print("[left]")
        elif (action == 3): # 4: right
            self.state[1] = self.state[1] + 1
            # check obstacles:
            if tuple(self.state) in self.obs: 
                self.state[1] = self.state[1] - 1
            ##print("[right]")
        else:
            print("invalid action")
        
        # CHECK BOUNDARIES:
        # check upper boundary
        if self.state[0] > self.size[0]:
            self.state[0] = self.size[0]        
        # check lower boundary
        if self.state[0] < 0:
            self.state[0] = 0           
        # check left boundary
        if self.state[1] < 0:
            self.state[1] = 0 
        # check right boundary
        if self.state[1] > self.size[1]:
            self.state[1] = self.size[1]  
        
        if (self.state[0] == self.goal_pt[0] and self.state[1] == self.goal_pt[1]):
            terminal = True
            reward = float(1)
        else:
            terminal = False
            reward = float(0)
            
        return reward, self.state, terminal


    def env_message(self, in_message):
        pass

###----------------------- TEST ---------------------
#environment = GridEnvironment()
#environment.env_init()
#environment.env_start()
#terminal = False
#while not terminal:
    ##print("current state: {}".format(environment.state))
    #step = int(input("enter a step: "))
    #reward,state,terminal = environment.env_step(step)
    #print("current state: {}\n".format(state))
    
