"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
import random

class DynaQ0Agent(BaseAgent):


    def __init__(self):
        """Declare agent variables."""
        self.alpha = None
        self.epsilon = None
        self.gamma = None
        
        self.col = None
        self.row = None
        self.num_action = None 
        self.steps = None
        self.episode = None        
        
        self.Q = None  
        self.model = None
        self.prev_states = None
        self.state_record = None
        
        self.action = None
        #self.action_ = None
        self.state = None
        self.state_ = None
        
        self.step_record = None #delete
        self.time_steps = None
        
        
        

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.alpha = 0.1
        self.epsilon = 0.1
        self.gamma = 0.95
        
        self.col = 9
        self.row = 6
        self.num_action = 4   
        self.episode = 0
        self.n = 0
        
        self.Q = np.zeros((self.row,self.col,self.num_action),dtype="float")
        self.model = np.zeros((self.row,self.col,self.num_action),dtype="object")
        self.prev_states = set()
        self.state_record = np.zeros((self.row,self.col),dtype="object")
        
        self.action = None
        #self.action_ = None
        self.state = None
        self.state_ = None        
        
        


    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        """
        self.step_record = []
        ##np.random.seed(1)
        
        self.state = state  # update self.state
        # choose A using S and Q:
        if (np.random.random(1)[0]<self.epsilon):
            # take random action
            self.action = random.randint(0,self.num_action-1)   # choose a random int from 0 to 3
        else:
            # take greedy action
            self.action = np.argmax(self.Q[self.state[0]][self.state[1]]) # update self.action
        
        return self.action

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        self.action = int(self.action)
        # observe next state and reward(in)
        self.state_ = state
        
        ##print(self.state_record)
        # update self.state_record (add action)
        if tuple(self.state_) not in self.prev_states:
            if self.state_record[self.state_[0]][self.state_[1]] == 0:
                self.state_record[self.state_[0]][self.state_[1]] = set()
            self.state_record[self.state_[0]][self.state_[1]].add(self.action)
        # update self.prev_states (add new state into self.prev_states):
        self.prev_states.add(tuple(self.state_))
        
        # update Q using action, state and state_
        # Q(S,A) <-- Q(S,A)+a*(R+y*max(Q(S',a))-Q(S,A))
 #       print("action = {}".format(self.action))
        self.Q[self.state[0]][self.state[1]][self.action] += self.alpha * (reward + self.gamma * max(self.Q[self.state_[0]][self.state_[1]]) - self.Q[self.state[0]][self.state[1]][self.action])
  #      print("self.Q = {}".format(self.Q))
        
        # update model using reward and new state
        self.model[self.state[0]][self.state[1]][self.action] = (self.state_[0], self.state[1],reward) 
  #      print("model = {}".format(self.model))
        
  #      print("start planning: ")
        # loop n times
        for i in range(self.n):
            # get random state from self.prev_states
  #          print("prev_states = {}".format(self.prev_states))
            s = random.choice(tuple(self.prev_states))
  #          print("s = {}".format(s))
            # get random action at state s, from self.state_record
            a = random.choice(tuple(self.state_record[s[0]][s[1]]))
  #          print("a = {}".format(a))
            # get new state(row and column) and new reward using s and a above, through self.model
            row,col,r = self.model[s[0]][s[1]][a]
   #         print("new row, col, r = {}, {}, {}".format(row,col,r))
            # update Q
            self.Q[s[0]][s[1]][a] += self.alpha * (r + self.gamma * max(self.Q[row][col]) - self.Q[s[0]][s[1]][a])
   #         print("new Q = {}".format())
            
        
        # choose A' from S' using policy derived from Q:
        if (np.random.random(1)[0]<self.epsilon):
            # take random action
            new_action = random.randint(0,self.num_action-1)   # choose a random int from 0 to 3
        else:
            # take greedy action
            # if all action values are zero, random
            if (self.Q[self.state[0]][self.state[1]][0]+self.Q[self.state[0]][self.state[1]][1]+self.Q[self.state[0]][self.state[1]][2]+self.Q[self.state[0]][self.state[1]][3]==0):
                new_action = random.randint(0,self.num_action-1)
            else:
                new_action = np.argmax(self.Q[self.state[0]][self.state[1]]) # update self.action            
            ##new_action = np.argmax(self.Q[self.state[0]][self.state[1]])  

        
        # update S, A:
   #     print("self.state = {}".format)
   #     print("new action = {}".format(new_action))
        self.state = self.state_
        self.action = new_action
    #    print("new step")
        ##print(self.steps)
        return self.action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        ##print("-----{}".format(self.steps))
        self.Q[self.state[0]][self.state[1]][self.action] += self.alpha * (reward - self.Q[self.state[0]][self.state[1]][self.action])
        
        self.episode = self.episode+1
        print("====={}".format(self.time_steps))

                
                
    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'Q':
            return self.steps, self.step_record
    
