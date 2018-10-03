import numpy as np
from collections import defaultdict

class Agent:
    # I used Expected Sarsa with MC Control to get probability, so that we can do exploration and exploitation
    
    def __init__(self, nA=6, epsilon=0.004, alpha=0.1, gamma=1.0, divisor = 400):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.divisor = divisor
    
    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        
        #epsilon = 1.0 / (i_episode + 0.3) #seems returns the highest reward when it sets to this line, and the other epsilon in the other method is set as  epsilon = 1.0 / (i_episode)
        
        epsilon = 1.0 / (i_episode) 

        # adding the value of divisor in the method select_action AND method step, seems will yield to lower average reward. So divisor 800 has a higher reward than 8000
        
        policy = np.ones(self.nA) * epsilon / self.nA
        policy[np.argmax(self.Q[state])] = 1 - epsilon + epsilon / self.nA 
        
        #the function get_probs above (3 lines above), allows us to do exploration-exploitation
        
        # pick next action A
        action =  np.random.choice(np.arange(self.nA), p=policy)
        
        
        return action
        
        
        """
        Epsilon must be a number between 0 and 1. If Epsilon is small, then the agent will be greedy. Conversely, if the Epsilon is big, then the agent will explore various actions
        """
        
        
    def step(self, state, action, reward, next_state, done, i_episode):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        """
        The step-size parameter α\alphaα must satisfy 0<α≤10 < \alpha \leq 10<α≤1. Higher values of α\alphaα will result in faster learning, but values of α\alphaα that are too high can prevent MC control from converging to π∗\pi_*π∗​.
        
        the higher is the gamma, the more the agent cares about the FUTURE REWARD. Conversely, the less is the gamma, the more the agent cares about the MOST IMMEDIATE REWARD
        Gamma value is always between 0 and 1
        """
        
        
        epsilon = 1.0 / (i_episode)
        
        #I think this is the way to select the policy. It is used to obtain the action probabilities corresponding to epsilon-greedy policy.
        next_policy = np.ones(self.nA) * self.epsilon / self.nA 
        
       
        
        next_policy[np.argmax(self.Q[state])] = 1 - epsilon + epsilon / self.nA 
        
        
        #Essentially, next_policy is deemed as the best policy that should be selected by the agent
        
        #In expected sarsa, it uses np.dot(self.Q[next_state], next_policy)
        self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward + (self.gamma * np.dot(self.Q[next_state], next_policy) - self.Q[state][action])))  
        
