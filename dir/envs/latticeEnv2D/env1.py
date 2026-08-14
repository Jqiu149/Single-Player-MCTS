import numpy as np
import copy
from ..static_env import StaticEnv
from ..helper import *
from .problem_specific_helpers import *



#will be functions
start_obj_generator= None
apply_step_penalty = None

#will be constant non-negative integers
MAX_STEP = None
HIST_LEN = None


#actions
END = 0
S = 1
T = 2


def get_obs_shape():
	return [2 + HIST_LEN, 2]
	
#states will be list of...
#2 np arrays of legnth 2: 2 linearly indep vectors of dim 2 that the player can act on to produce a new basis
#HIST_LEN many np arrays of length 2: the past HIST_LEN many vectors that were 'removed' from the current state
#		i.e if you apply T, it's the first vector that got added to  and if you apply S it's the second vector that we multipleid by -1
#the magnitude of the smallest vector in the lattice determined by those vectors
#and a boolean 'done' that the agent can set to true to finish the episode


class Env(StaticEnv):
	n_actions= 3

	@staticmethod
	def next_state(state, action):
		"""
		Given the current state of the environment and the action that is
		performed in that state, returns the resulting state.
		:param state: Current state of the environment.
		:param action: Action that is performed in that state.
		:return: Resulting state.
		"""

		v0 = state[0]
		v1 = state[1]
		m = state[-2]
		done = state[-1]

		if(action == END):
			new_v0 = v0.copy()
			new_v1 = v1.copy()
			hist = state[2:-2]
			done = True
		elif(action== S):
			new_v0 = -v1
			new_v1 = v0.copy()
			hist = [v1]+ state[2: -3] if HIST_LEN>0 else []
		elif(action == T):
			new_v0 = v0+v1
			new_v1 = v1.copy()
			hist = [v0]+ state[2: -3] if HIST_LEN>0 else []
		else:
			raise ValueError(f"given action, {action}, is unknown")

		return [new_v0, new_v1] + hist+ [m, done]

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
		
		start_basis= start_obj_generator()
		smallest_vector = LagrangeReduce(start_basis[0], start_basis[1])[0]
		smallest_m = np.linalg.norm(smallest_vector)

		
		return start_basis + [np.zeros(2)]*HIST_LEN + [smallest_m, False]

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

		min_magnitude= min([np.linalg.norm(v) for v in state[0:2]])
		score = apply_step_penalty ( 
                    pre_penalty_reward = ( state[-2]/min_magnitude)**2 ,
                    step_count =  step_idx
                )

		return	score



