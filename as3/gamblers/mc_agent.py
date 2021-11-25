"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np


class MonteCarloAgent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.policy = None
        self.action = None
        self.Q = None
        self.episode_path = None

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.Q = np.zeros((100,50),dtype="float")
        self.policy = np.zeros(100,dtype="float")
        self.episode_path = set()
        # starting policy
        self.action = None
        for i in range(1,100):
            self.policy[i] = min(i,100-i)

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        ##self.Q = np.zeros((100,50),dtype="float")
        self.episode_path = set()
        
        ##self.policy = np.zeros((100,1),dtype="float")
        ##for i in range(1,100):
        ##    self.policy[i,0] = min(i,100-i)
        self.action = int(self.policy[int(state)])
        self.episode_path.add((int(state),self.action)) 
        ##print("start action = {}".format(self.action))
        return self.action

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        self.action = int(self.policy[int(state)])
        ##print("action = {}".format(self.action[0]))
        ##print("state = {}".format(state[0]))
        self.episode_path.add((int(state),self.action)) 
        ##print("step episode = {}".format(self.episode_path))
        
        return self.action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        print(self.episode_path)
        G = 0
        gamma = 0.9
        for
        ##print("reward = {}".format(reward))

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'ValueFunction':
            return (np.max(self.Q, axis=1)).tostring()
        else:
            return "I dont know how to respond to this message!!"
