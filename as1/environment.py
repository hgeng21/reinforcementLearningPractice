import numpy as np
from rl_glue import BaseEnvironment


class Environment(BaseEnvironment):
    """
    Example single-state environment with 10 actions
    """

    def __init__(self):
        """Declare environment variables."""
        super(Environment,self).__init__()
        self.arms = []
        self.optimalArm = None
        self.num_step = 0
        self.max_steps = 1000
        
        # create list containing num of optimal steps within the 1000 steps for 2000 runs
        self.optimal_steps = []

    def env_init(self):
        """
        Initialize environment variables.
        """
        # initialize the 10 arms
        self.arms = []*10
        for x in range(10):
            self.arms.append(np.random.normal(0,1))
        #print("self.arms:")
        #print(self.arms)
#        print("self.optimal: {}".format(self.optimalArm))        
        ##print("the highest arm is {}".format(self.highestArm))
        
        

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        # re-initialize
        self.optimal_steps = [0]*self.max_steps
        self.num_step = 0
##        print(self.arms)
        
        # only one state, which we will represent using 0
        return 0

    def env_step(self, action):
        """
        A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """  

        state = 0
        terminal = False
        trueReward = self.arms[action]
        
        self.optimalArm = self.arms.index(max(self.arms))
        
        # update num_step
        self.num_step += 1
        
        #print(action)


        ##print("action-{} = {}, optimal = {}".format(self.num_step,action,self.optimalArm))
        # update optimal_steps list by checking if the action taken was optimal
        if (action == self.optimalArm):
            self.optimal_steps[self.num_step] += 1

        # return corresponding reward of action given, normal distribution
        reward = np.random.normal(trueReward, 1)
        ##print("R = {}".format(reward))
        
        try:
            return reward, state, terminal
        except NameError:
            m = "Invalid action specified in One-State Environment's " \
                "env_step: {}"
            print(m.format(action))
            #print("Please only return the integers 0 and 1 as actions.\n")
            exit(1)

    def env_message(self, message):
        ##print(self.arms)
        ##print("highest arm: {}".format(highestArm))
        if (message == "getOptimal"):
            ##print(">>>within environment: {}".format(self.optimal_steps))
            return self.optimal_steps
