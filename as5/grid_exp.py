
from rl_glue import RLGlue
from grid_env import GridEnvironment
#from dynaq0_agent import DynaQ0Agent
from sarsa_agent import DynaQAgent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import time
import random

if __name__ == "__main__":
    num_episodes = 20
    max_steps = 2500
    num_runs = 10

    
    # Create and pass agent and environment objects to RLGlue
    environment = GridEnvironment()
    #agent = DynaQ0Agent()
    agent = DynaQAgent()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore
 
    
    ##colors=iter(cm.rainbow(np.linspace(0,1,num_runs))) 
    
    ##t=time.time()
    
    save_for_plot = np.zeros(num_runs, "object")
    ##print(save_for_plot)
    for run in range(num_runs):
        
        ##print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> run {}".format(run))
        # set seed for reproducibility
        np.random.seed(run)
        random.seed(run)
        
        # initialize RL-Glue
        rlglue.rl_init()  
        
        # get data
        time_steps =[]
        for episode in range(num_episodes):
            ##print(">>>>>>>>> episode {}".format(episode))
            # run episode with the allocated steps budget
            rlglue.rl_episode(max_steps)  
            steps = rlglue.num_ep_steps()
            ##print("step = {}".format(steps))
            time_steps.append(steps)
        ##print(time_steps)
        save_for_plot[run] = time_steps
            
    #print(num_episodes)
    ##print(save_for_plot)
    new_plot = [0]*num_episodes
    for i in range(num_episodes):
        temp = 0
        for j in range(num_runs):
            temp += save_for_plot[j][i]
        new_plot[i] = temp/num_runs
        
    ##print(new_plot)
    episodes = list(range(num_episodes)) 
    ##print(episodes)
    #episodes = list(range(len(save_for_plot[0])))  
    
    plt.figure(1)
    plt.plot(episodes, new_plot)
    plt.show()
    ##print("time usage: {}".format(time.time()-t))
