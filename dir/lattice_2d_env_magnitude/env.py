import numpy as np
from ..static_env import StaticEnv

START_VECTORS = [ np.array([1,2]), np.array([0,2])]
MAX_STEP = 5

#action list 0
S = 0
T = 1

num_actions = 2

#action list 1
#END = 0
#S = 1
#T = 2
#num_actions = 3

#action list 2
#END=0
#ADD_TO_v1 = 1
#SUBTRACT_FROM_V1 = 2
#ADD_TO_v2 = 3
#SUBTRACT_FROM_V2 = 4

#num_actions = 5

	
#states will be list of 2 np arrays of the same dim and a boolean.

class Env(StaticEnv):
    n_actions= num_actions

    @staticmethod
    def next_state(state, action):
        """
        Given the current state of the environment and the action that is
        performed in that state, returns the resulting state.
        :param state: Current state of the environment.
        :param action: Action that is performed in that state.
        :return: Resulting state.
        """
        if(action== S):
            temp = state[0]
            state[0] = -state[1]
            state[1] = temp
        elif(action == T):
            state[0] = state[0] + state[1]	
        else:
            raise ValueError(f"given action, {action}, is unknown")

#        if(action == END):
#            state[-1] = True
#            return state	
#        elif(action== S):
#            temp = state[0]
#            state[0] = -state[1]
#            state[1] = temp
#        elif(action == T):
#            state[0] = state[0] + state[1]	
#        else:
#            raise ValueError(f"given action, {action}, is unknown")

        return state

# i don't think we can do the step_idx thing for is done if we want to only give reward at the end after the stop button is used
    @staticmethod
    def is_done_state(state, step_idx):
        """
        Given the state and the index of the current step, returns whether
        that state is the end of an episode, i.e. a done state.
        :param state: Current state.
        :param step_idx: Index of the step at which the state occurred.
        :return: True, if the step is a done state, False otherwise.
        """	
        #if (state[2] == True or step_idx >= MAX_STEP): print("done")
        #print(step_idx)
        return state[2] == True or step_idx >= MAX_STEP

    @staticmethod
    def initial_state():
        """
        Returns the initial state of the environment.
        """
        return START_VECTORS+ [False]

    @staticmethod
    def get_obs_for_states(states):
        """
        Some environments distinguish states and observations. An observation
        can be a subset (e.g. in Poker, state is all cards in game, observation
        is cards on player's hand) or superset of the state (i.e. observations
        add additional information).
        :param states: List of states.
        :return: Numpy array of observations.
        """
        x = np.array([ state[0:-1] for state in states],dtype=np.float32)
        return x

    @staticmethod
    def get_return(state, step_idx):
        """
        Returns the return that the agent has achieved so far when he is in
        a given state after a given number of steps.
        :param state: Current state that the agent is in.
        :param step_idx: Index of the step at which the agent reached that
        state.
        :return: Return the agent has achieved so far.
        """

        magnitudes = [np.linalg.norm( v) for v in state[0:-1]]
        return -min(magnitudes)
