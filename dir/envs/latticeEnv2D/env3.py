import numpy as np
import random
import	copy
from functools import partial
from scipy.stats import loguniform
from ..static_env import StaticEnv
from .. helper import parse_init_method
from . init_helpers import *


def select_init_method(method, custom_list): 
	global basis_generator

	method, args = parse_init_method(method)

	if method == "default":
		basis_generator = pick_from_basis_list
	elif method == "random_generator":
		basis_generator = partial(random_basis, **args)
	elif method== "custom_list":
		assert np.shape(custom_list)[1:] == (2,2), f"custom_list shape is {np.shape(custom_list)}"
		custom_list = [ [np.array(vector) for vector in vector_list ] for vector_list in custom_list]

		assert all( pairVectorsR2LinearIndep(basis_vectors[0],basis_vectors[1]) for basis_vectors in custom_list)

		global basis_list
		basis_list = custom_list

		print("basis_list is:", basis_list)

		basis_generator = pick_from_basis_list
	else:
		ValueError (f"method chosen isn't one of the options, given {method}")




basis_generator= pick_from_basis_list
MAX_STEP = 300
STEP_PENALTY = 1e-5
HIST_LEN = 0


#actions
END = 0

ADD_V0= 1
SUB_V0= 2

ADD_V1= 3
SUB_V1= 4


#states will be list of...
#2 np arrays of legnth 2: 2 linearly indep vectors of dim 2 defining the lattice
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

		m = state[-2]
		done = state[-1]

		if(action == END):
			hist = state[2:-2]
			done = True
		elif(action== ADD_V0):
			new_v0 = v0
			new_v1 = v1 + v0
			hist = [v1]+ state[2: -3] if HIST_LEN>0 else []
		elif(action == SUB_V0):
			new_v0 = v0
			new_v1 = v1-v0
			hist = [v1]+ state[2: -3] if HIST_LEN>0 else []
		elif(action== ADD_V1):
			new_v0 = v0+v1
			new_v1 = v1
			hist = [v0]+ state[2: -3] if HIST_LEN>0 else []
		elif(action == SUB_V1):
			new_v0 = v0 -v1
			new_v1 = v1
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
		start_basis= basis_generator()
		smallest_vector = LagrangeReduce(start_basis[0], start_basis[1])[0]
		smallest_m = np.linalg.norm(smallest_vector)

		
		return start_basis + [np.zeros(2)]*HIST_LEN+ [smallest_m, False]

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

		score = ( state[-2]/current_magnitude)**2 - step_idx*STEP_PENALTY


		#score = ( state[-2]/current_magnitude)**2 * (1- step_idx*STEP_PENALTY)

		return	score



