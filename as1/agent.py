import numpy as np
from rl_glue import BaseAgent


class Agent(BaseAgent): 
    """
    simple random agent, which picks randomly from the ten actions provided

    Note: inheret from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""

        # Your agent may need to remember what the action taken was.
        self.prevAction = None
        self.epsilon = 0.1

        # estimated values
        self.Q = []
        # set a variable to keep track of how many times each action was taken
        self.N = []


    def agent_init(self):
        """Initialize agent variables."""
        self.Q = [0]*10
        self.N = [0]*10

    def _choose_action(self):
        """
        Convenience function.

        You are free to define whatever internal convenience functions
        you want, you just need to make sure that the RLGlue interface
        functions are also defined as well.
        """
        randomNum = np.random.random()
        # epsilon:
        if (randomNum < self.epsilon):
            action = np.random.randint(10)
            return action
        # 1-epsilon:
        else:
            action = self.Q.index(max(self.Q))
            ##print(action)
            return action
        

    def agent_start(self, state):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (state observation): The agent's current state

        Returns:
            The first action the agent takes.
        """
        self.prevAction = self._choose_action()
        ##print("prevAction in agent_start: ", self.prevAction)
        # This agent doesn't care what state it's in
        return self.prevAction

    def agent_step(self, reward, state):
        """
        A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (state observation): The agent's current state
        Returns:
            The action the agent is taking.
        """
        
        # update estimated values in Q
        ##print("self-prevAction in agent_step is: ",self.prevAction)
        prevQ = self.Q[self.prevAction]
        newQ = prevQ + 0.1*(reward - prevQ)
        self.Q[self.prevAction] = newQ
        
        ##print("previousQ = {}".format(prevQ))

        # update values in N
        self.N[self.prevAction] = self.N[self.prevAction] + 1

        # get the action taking
        self.prevAction = self._choose_action()
        
        return self.prevAction

    def agent_end(self, reward):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        pass
    
        
        

    def agent_message(self, message):
        if (message == "greedy"):
            self.epsilon = 0
            self.Q = [5]*10
        else:
            highest = self.N.index(max(self.N))
            ##print("pick arm: {}\n".format(highest))
            return highest
