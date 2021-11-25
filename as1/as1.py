"""Assignment1 for CMPUT 366 Fall 2019

This experiment uses the rl_episode() function.

Runs a random agent in a 1D environment. Runs 10 (num_runs) iterations of
100 episodes, and reports the total reward. Each episode is capped at 100 steps.
(max_steps)
"""
# import float division
from __future__ import division

import numpy as np

# import add to add two lists elementwise
from operator import add

import matplotlib.pyplot as plt
from environment import Environment
from agent import Agent
from rl_glue import RLGlue


def experiment1(rlg, num_runs, max_steps):
    
    # create list for optimal percentage of 1000 steps
    optimal_steps = [0]*max_steps
    rewards = np.zeros(num_runs)
    averageRewards = np.zeros(num_runs)
    for run in range(num_runs):
        # set seed for reproducibility
        ##np.random.seed(run)

        # initialize RL-Glue
        rlg.rl_init()

        # example: do 1 episodes using the convenience call rl_episode()
        rlg.rl_episode(max_steps)

        rewards[run] = rlg.total_reward()
        ##print("Experiment total reward: {}".format(rewards[run]))
        
        # find the arm with highest value
        ##highestArm = rlg.rl_env_message("real")
        ##print("highest arm: {}".format(highestArm))
        ##highest = rlg.rl_env_message("est")
        ##print("pick arm: {}\n".format(highest))
        
        # get optimal_steps list from this run
        current_optimal_steps = rlg.rl_env_message("getOptimal")
##        print("optimal steps: {}".format(optimal_steps))
        optimal_steps = list(map(add, current_optimal_steps, optimal_steps))
##        print("run {}:".format(run))
        ##print("current optimal steps after addition: {}".format(optimal_steps))
        
        averageRewards[run] = rewards[run] / max_steps
##        print("average reward for run{} is {}".format(run, averageRewards[run]))

    return averageRewards.mean(), optimal_steps


def experiment2(rlg, num_runs, max_steps):
    
    # create list for optimal percentage of 1000 steps
    optimal_steps = [0]*max_steps
    rewards = np.zeros(num_runs)
    averageRewards = np.zeros(num_runs)
    for run in range(num_runs):

        # initialize RL-Glue
        rlg.rl_init()

        # example: do 1 episodes using the convenience call rl_episode()
        rlg.rl_episode(max_steps)

        rewards[run] = rlg.total_reward()
        
        # get optimal_steps list from this run
        current_optimal_steps = rlg.rl_env_message("getOptimal")
        optimal_steps = list(map(add, current_optimal_steps, optimal_steps))
        
        averageRewards[run] = rewards[run] / max_steps

    return averageRewards.mean(), optimal_steps






def main():
    max_steps = 1000  # max number of steps in an episode
    num_runs = 2000  # number of repetitions of the experiment

    # Create and pass agent and environment objects to RLGlue
    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore

    # run the experiment
    results, optimal_steps = experiment1(rlglue, num_runs, max_steps)
    print("Experiment average reward: {}\n".format(results))
    
    rlglue.rl_agent_message("greedy")
    results2, optimal_steps2 = experiment2(rlglue, num_runs, max_steps)
    print("Experiment2 average reward: {}\n".format(results2))    
    
    
    
##    print(optimal_steps)
    
    # calculate percentage of optimal steps
    for i in range(max_steps):
        ##print(optimal_steps[i])
        optimal_steps[i] = optimal_steps[i] / num_runs
        optimal_steps2[i] = optimal_steps2[i] / num_runs
    # draw graph using the optimal_steps
    ##
    ##print(len(list(range(1,max_steps))))
    ##print(len(optimal_steps))
    
    # plot epsilon-greedy curve
    plt.plot(list(range(max_steps)), optimal_steps, '-')
    plt.axis([-10, max_steps, 0, 1])
    
    plt.plot(list(range(max_steps)), optimal_steps2, '-')
    plt.axis([-10, max_steps, 0, 1])
        
    plt.show()
    ##print(optimal_steps)


if __name__ == '__main__':
    main()
