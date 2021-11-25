"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
import random

class DynaQAgent(BaseAgent):


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
        ##self.state_record = None
        
        self.action = None
        self.state = None
        self.state_ = None


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
        
        #=======
        self.count = 0
        
        self.Q = np.zeros((self.row,self.col,self.num_action),dtype="float")
        self.model = np.zeros((self.row,self.col,self.num_action),dtype="object")
        self.prev_states = {}
        ##self.state_record = np.zeros((self.row,self.col),dtype="object")
        ## initialize state record to all empty sets
        #for i in range(self.row):
            #for j in range(self.col):
                #self.state_record[i][j] = set()
        
        ##----------------
        #print("init Q = {}, init model = {}, init prev_states = {}, init state_record = {}".format(self.Q, self.model, self.prev_states, self.state_record))
        ##----------------
        
        self.action = None
        self.state = None
        self.state_ = None         


    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        """
        #========
        self.count = 1
        
      
        
        self.state = state  # store current state
        
        # using this state, find an action (epsilon-greedy)
        x = np.random.random(1)[0]
        ##print("x = {}".format(x))
        if (x<self.epsilon):
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
        ##print("self.action = {}".format(self.action))
        
        
        #---------------------------
        # observe next state and reward(in)
        self.state_ = state
        ##print("new state = {}".format(self.state_))
        
        
        if (self.state[0],self.state[1]) not in self.prev_states:
                self.prev_states[(self.state[0],self.state[1])] = set()
        self.prev_states[(self.state[0],self.state[1])].add(self.action)        
        
        
        
        
        ##---------------------------
        ## update self.prev_states (add state)
        #self.prev_states.add(tuple(self.state_))
        ###print("prev states = {}".format(self.prev_states))

        ##---------------------------
        ## update self.state_record (add action)
        #self.state_record[self.state_[0]][self.state_[1]].add(self.action)
        ###print("state record = {}".format(self.state_record))
          
        #---------------------------
        # update Q 
        # Q(S,A) <-- Q(S,A)+a*[R+y*max(Q(S',a))-Q(S,A)]
        temp = reward + self.gamma * max(self.Q[self.state_[0]][self.state_[1]])
        self.Q[self.state[0]][self.state[1]][self.action] += self.alpha * (temp - self.Q[self.state[0]][self.state[1]][self.action])
        ##print("Q(S,A) = {}".format(self.Q[self.state[0]][self.state[1]][self.action]))
        
        
        #---------------------------
        # update model
        # model(S,A) <-- R,S'
        self.model[self.state[0]][self.state[1]][self.action] = (reward, self.state_[0], self.state[1])
        ##print("model = {}".format(self.model))
        
        
        #---------------------------
        # start planning
        ##print("\nstart planning: ")
        
        # loop n times
        for i in range(self.n):
            
            # get random state from self.prev_states
                        
            
            ##print(self.prev_states.keys())
            s = random.choice(list(self.prev_states.keys()))
            a = random.sample(self.prev_states[s], 1)
            
            
            #=============
            ## get random state from self.prev_states
            #s = random.choice(tuple(self.prev_states))
            ###print("s = {}".format(s))
            
            ## get random action at state s, from self.state_record
            #a = random.choice(tuple(self.state_record[s[0]][s[1]])) 
            ###print("a = {}".format(a))
            #=============
            
            # get new reward and new state using s and a above, through self.model
            r_new, s_new_row, s_new_col = self.model[s[0]][s[1]][a[0]]
            ##print("r_new = {}, s_new = {}".format(r_new, s_new))
            
            # update Q
            temp = r_new + self.gamma * max(self.Q[s_new_row][s_new_col])
            self.Q[s[0]][s[1]][a[0]] += self.alpha * (temp - self.Q[s[0]][s[1]][a[0]])
            print(">> new Q(S,A) = {}".format(self.Q[s[0]][s[1]][a]))            
            
            
        
        
        
        
        #---------------------------
        # using this state, find an action (epsilon-greedy)
        x = np.random.random(1)[0]
        ##print("x = {}".format(x))
        if (x<self.epsilon):
            ##print("random")
            # take random action
            new_action = random.randint(0,self.num_action-1)   # choose a random int from 0 to 3 
        else:
            # take greedy action
            ##print("greedy")
            ## if all action values are zero, random
            #if (self.Q[self.state[0]][self.state[1]][0]+self.Q[self.state[0]][self.state[1]][1]+self.Q[self.state[0]][self.state[1]][2]+self.Q[self.state[0]][self.state[1]][3]==0):
                #new_action = random.randint(0,self.num_action-1)
            #else:
                #new_action = np.argmax(self.Q[self.state[0]][self.state[1]]) # update self.action
            new_action = np.argmax(self.Q[self.state[0]][self.state[1]]) # update self.action
        ##print("new action = {}".format(new_action))
        
        #---------------------------
        self.state = self.state_
        self.action = new_action
        ##print("current state = {}".format(self.state_))
        
        ##self.action = int(input("\nenter action: ")) 
        
        self.count += 1
        print(self.count)
        return self.action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        self.Q[self.state[0]][self.state[1]][self.action] += self.alpha * (reward - self.Q[self.state[0]][self.state[1]][self.action])
        ##print("END\n")
        
        self.count += 1
        
        print("final Q = {}".format(self.Q))

                
                
    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'Q':
            return self.steps, self.step_record
        if in_message == 'TimeSteps':
            return self.time_steps
