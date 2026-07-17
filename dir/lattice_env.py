import numpy as np
from .static_env import StaticEnv

START_VECTORS = [ np.array([100,70]), np.array([50,50])]

MAX_STEP = 30

#action list 0
#S = 0
#T = 1

#num_actions = 2

#action list 1
END = 0
S = 1
T = 2
num_actions = 3



def LagrangeReduce(v1,v2):
    assert np.shape(v1) == (2,), f"LagrangeReduce expects v1 to be shape (2,0), v1 is ${v1} and v2 is ${v2}"
    assert np.shape(v2) == (2,), f"LagrangeReduce expects v1 to be shape (2,0), v1 is ${v1} and v2 is ${v2}"

    # just for concenience for element wise operation notation... probably easier ways to do this
    norm1Squared = np.dot(v1,v1) 
    norm2Squared = np.dot(v2,v2) 

    done = False
    stepCount = 0
    while(not done):
        stepCount+=1

        if(norm1Squared> norm2Squared):
            v1,v2 = v2,v1
            norm1Squared,norm2Squared = norm2Squared,norm1Squared

        u = round( (np.dot(v1,v2))/ norm1Squared)
        v2 = v2-u*v1
        norm2Squared= np.dot(v2,v2) 

        if(norm1Squared<= norm2Squared):
            done = True

    return [v1, v2]






	
#states will be list of...
#2 np arrays of legnth 2 (2 linearly indep vectors of dim 2)
#the magnitude of the smallest vector in the lattice determined by those vectors
#and a boolean 'done' that the agent can set to true to finish the episode


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

		v0, v1,m,done = state 

		if(action == END):
			new_v0 = v0.copy()
			new_v1 = v1.copy()
			done = True
		elif(action== S):
			new_v0 = -v1
			new_v1 = v0.copy()
		elif(action == T):
			new_v0 = v0+v1
			new_v1 = v1.copy()
		else:
			raise ValueError(f"given action, {action}, is unknown")

		return [new_v0, new_v1,m, done]

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
		return state[-1] == True or step_idx >= MAX_STEP

	@staticmethod
	def initial_state():
		"""
		Returns the initial state of the environment.
		"""
		start_vector = START_VECTORS
		smallest_vector = LagrangeReduce(start_vector[0], start_vector[1])[0]
		smallest_m = np.linalg.norm(smallest_vector)


		return START_VECTORS+ [smallest_m, False]

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
		x = np.array([ state[0:-2] for state in states],dtype=np.float32)
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

		magnitudes = [np.linalg.norm(v) for v in state[0:-2]]
		return ( state[-2]/min(magnitudes))



