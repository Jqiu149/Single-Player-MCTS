import numpy as np
import copy
from scipy.linalg import block_diag

from ..static_env import StaticEnv
from ..helper import *
from .problem_specific_helpers import *



#idk for now jsut writing this as a variable we'll change manually.....  ,maybe later figure out how to make it like editable through command line
N=2 
q= 251
DIM = 2*N


#will be functions, assigned by program using function in problem specific helpers
#start_obj_generator= None #probably not using this one for this?
apply_step_penalty = None
#will be constant non-negative integers, also assigned by program
MAX_STEP = None

def get_obs_shape():
	return [DIM, DIM]





#.... should my actions be like doing things modulo q?
#i think so....? 



def create_E_ij_func(n, i,j, sign):
  assert sign == "+" or sign == "-", f"expected sign to be + or -, got {sign}"
  assert n>j, f"expected n to be greater than or equal to j, got n={n}, j={j}"
  assert j>i, f"expected i to be less than j, got i={i}, j={j}"
  assert i>=0, f"expected i to be greater than or equal to 1, got i={i}"

  E_ij = np.identity(n)
  if sign == "+":
    E_ij[i,j] = 1
  else:
    E_ij[i,j] = -1


  #function ig will assume it's taking in DIM by DIM np array?
  def fun(state):
    resulting_matrix = np.matmul(E_ij,state[0])
    return [resulting_matrix, state[1], state[2]]

  return fun


# this is 0-indexed and basically our 'middle' block will start at index i. 
def create_S_i(n,i):
  assert i<n-1, f"expected i to be less than n-1, got i={i}, n={n}"
  assert i>=0, f"expected i to be greater than or equal to 0, got i={i}"

  upper_I= np.identity(i) if i>0 else None
  lower_I = np.identity(n-i-2)
  middle_block = np.array([[0,-1], [1,0]])

  S_i = block_diag(upper_I, middle_block, lower_I) if i>0 else block_diag( middle_block, lower_I)


  def fun(state):
    resulting_matrix = np.matmul(S_i,state[0])
    return [resulting_matrix, state[1], state[2]]

  return fun



action_list = []
#list of functions 

def end_game_action(state):
  return [state[0], state[1], True]
action_list.append(end_game_action)

for i in range(DIM):
  for j in range(i+1,DIM):
    for sign in ["+","-"]:
      action_list.append(create_E_ij_func(DIM,i,j,sign))

for i in range(DIM-1):
  action_list.append(create_S_i(n,i))





#states will bdimension  list of...
# a np matrix of dimension DIMxDIM
#the magnitude of the smallest vector in the lattice determined by those vectors
#and a boolean 'done' that the agent can set to true to finish the episode

#observations ig are us doing grahm matrix?

class Env(StaticEnv):
	n_actions=DIM**2

	@staticmethod
	def next_state(state, action):
		"""
		Given the current state of the environment and the action that is
		performed in that state, returns the resulting state.
		:param state: Current state of the environment.
		:param action: Action that is performed in that state.
		:return: Resulting state.
		"""

		return action_list[action](state)

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
		def inital_basis(n,q):
			zero_block = np.zeros((n,n),dtype=np.int64)

			I_block = np.identity(n,dtype=np.int64)

			rng = np.random.default_rng()
			A_block = rng.integers(0,q , size=(n,n),dtype=np.int64)
			result = np.block([[q*I_block, zero_block], [A_block, I_block]]) 

			return result

		start_basis= inital_basis(DIM,q)


		A=IntegerMatrix.from_matrix(start_basis)
		Reduced = LLL.reduction(A) #should check what LLL paramater values we're runing this with ig? 
		reduced_numpy =np.empty( (2*n,2*n) ,dtype=np.int64)
		Reduced.to_matrix(reduced_numpy)

		magnitudes = [np.linalg.norm(v) for v in reduced_numpy] 

		smallest_m = min(magnitudes)

		return start_basis + [smallest_m, False]

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
		x = np.array([ state[0] @ state[0].T for state in states],dtype=np.float32)
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

		min_magnitude= min([np.linalg.norm(v) for v in state[0]])
		score = apply_step_penalty (
					pre_penalty_reward = ( state[-2]/min_magnitude)**2 ,
					step_count =  step_idx
				)
		return	score


statistic_functions= {
        "reached_min_magnitude": reached_min_mag
      }
