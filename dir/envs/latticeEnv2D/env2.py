import numpy as np
import random
import	copy
from functools import partial
from scipy.stats import loguniform
from ..static_env import StaticEnv

from ..helper import *
from .problem_specific_helpers import *


start_obj_generator= None
MAX_STEP = None
STEP_PENALTY = None
HIST_LEN = None


#actions
END = 0
S = 1
T = 2

#actions
END = 0
ADD_V0= 1
SUB_V0= 2
ADD_V1= 3
SUB_V1= 4



def get_obs_shape():
	return [3 + HIST_LEN, 2]

#another np array of length 2 that is the 'current' vector that will be asses for the score
#HIST_LEN many np arrays of length 2: the past HIST_LEN many vectors
#	the magnitude of the smallest vector in the lattice
#and a boolean 'done' that the agent can set to true to finish the episode


class Env(StaticEnv):
	n_actions= 5
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
		curr = state[2]

		m = state[-2]
		done = state[-1]

		if(action == END):
			new_curr= curr.copy()
			hist = state[3:-2]
			done = True
		elif(action== ADD_V0):
			new_curr = curr + v0
			hist = [curr]+ state[3: -3] if HIST_LEN>0 else []
		elif(action == SUB_V0):
			new_curr = curr - v0
			hist = [curr]+ state[3: -3] if HIST_LEN>0 else []
		elif(action== ADD_V1):
			new_curr = curr + v1
			hist = [curr]+ state[3: -3] if HIST_LEN>0 else []
		elif(action == SUB_V1):
			new_curr = curr - v1
			hist = [curr]+ state[3: -3] if HIST_LEN>0 else []
		else:
			raise ValueError(f"given action, {action}, is unknown")

		return [v0, v1, new_curr] + hist+ [m, done]

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

		
		return start_basis + [np.zeros(2)]*(HIST_LEN +1)+ [smallest_m, False]

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

		current_magnitude= np.linalg.norm(state[2])

		if current_magnitude == 0:
			return -50	
		score = ( state[-2]/current_magnitude)**2 - step_idx*STEP_PENALTY
		#score = ( state[-2]/current_magnitude)**2 * (1- step_idx*STEP_PENALTY)
		return	score



