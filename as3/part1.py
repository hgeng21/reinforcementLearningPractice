import numpy as np
import matplotlib.pyplot as plt

# funtion flip: flip a coin to get result head/tail
def flip(p_head):
    random_num = np.random.random()  # get a random number between 0 and 1
    ##print(random_num)
    if random_num < p_head: return 1
    else: return 0
    
    
# main program
def gamble():
    
    # initialization
    p_head = 0.40
    game_end = False
    gamma = 1
    theta = 0.001
    
    policy = [0]*101    
    state_values = [0]*101    #expected values for each states

    
    # start gamble
    while not game_end:
        terminate = 0
        
        ##p_head = input("probability of getting head (decimal) = ")
        # go through each stage, get value for each stage and store it
        for capital in range(1,100):
            # get the current state value state_values(capital)
            ## v0 = current state value
            v0 = state_values[capital]
            
            # initialize the expected reward
            ## r_max = max expected reward
            r_max = 0
            # set index of r_max
            max_r_index = 0
            
            # go through each action, find reward for each action, get the max
            # state_values(capital) = max(action expected reward)
            ## r = expected reward for current action(bet)
            for bet in range(0,min(capital,100-capital)+1):
                # decide reward for head
                if (capital+bet == 100): head_reward = 1
                else: head_reward = 0
                # decide reward for tail
                if (capital-bet == 100): tail_reward = 1
                else: tail_reward = 0
                
                # calculate head return
                head_return = p_head * (head_reward + gamma*state_values[capital+bet])
                # calculate tail return
                tail_return = (1-p_head) * (tail_reward + gamma*state_values[capital-bet])
                
                # calculate expected action reward
                r = head_return + tail_return
                
                # update r_max and max_r_index
                r_max = max(r,r_max)
                if (r>=r_max):
                    max_r_index = bet
                
            
            
            # store the expected reward for current state
            state_values[capital] = r_max
            policy[capital] = max_r_index
                
            # decide termination
            terminate = max(terminate, abs(r_max - v0))
        if (terminate<theta): 
            game_end = True
            break

    state_values = state_values[:-1]
    ##print(state_values)  
    policy = policy[:-1]
    ##print(policy)
    
    plt.figure(1)
    plt.plot(list(range(len(state_values))), state_values, 'green')
    plt.axis([-1, len(state_values)+1, -0.01, 1])
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')    
    plt.title("Capital vs. Value estimates")
    plt.show()
    plt.savefig('geng1_part1a.png')
    
    
    plt.figure(2)
    plt.plot(list(range(len(policy))), policy, 'green')          
    plt.axis([-1, len(policy)+1, -0.01, 50])
    plt.xlabel('Capital')
    plt.ylabel('Finla policy')    
    plt.title("Capital vs. Final policy")
    plt.show()
    plt.savefig('geng1_part1b.png')     
            





gamble()