import numpy as np
import random
from .static_env import StaticEnv



#what do we want? 
# we want to be able to specify either a specific starting basis, or a list of bases to be the starting state for training?
#or a function for specifying vectors to choose as basis 


#for if we have a specific list we want to take from

#set this to the list you want to be taking from, the value set below is more of an example / default ig
#should be a list of lists of 2 of np arrays of 2 integers
basis_list = [
		[np.array([1,2]), np.array([0,2])],
		[np.array([1,2]), np.array([3,4])],
		[np.array([100, 70]), np.array([50,50])],
		[np.array([349,-300]), np.array([49,-50])]
		]

def pick_from_basis_list():
	return random.choice(basis_list)


def pairVectorsR2LinearIndep(v1,v2):
		return	v1[0]*v2[1] != v2[0]*v1[1]

def polarToCartesian( angle, magnitude): 
	return [np.cos(angle)*magnitude, np.sin(angle)*magnitude]


def logUniformReal(maxint):
	return	10** (np.log10(maxint)*np.random.rand())

#for if you want to choose a basis randomly
#will generate a pair of linearly independent 2d integer vectors 
# we'll probably need to work on this to like check if we're hapy with the distribution this guves but... for now it will probably maybe run?
def random_basis(m=100,minAngleDiff=0,maxAngleDiff=2*np.pi):
	m1 = logUniformReal(m) 
	m2 = logUniformReal(m)
	a1 = random.uniform(0, 2*np.pi)
	a2 = a1+random.uniform(minAngleDiff,maxAngleDiff)


	v1 = [int(x) for x in polarToCartesian(a1, m1)]
	v2 = [int(x) for x in polarToCartesian(a2, m2)]

	if(v1 == [0,0]):
	  v1[0] = 1

	counter = 0
	while( not pairVectorsR2LinearIndep(v1, v2)):
		m2 = logUniformReal(m)
		a2 = a1+random.uniform(minAngleDiff, maxAngleDiff) 
		if random.uniform(0,1)> 0.5 :
			a2 += np.pi
		v2 = [int(x) for x in polarToCartesian(a2, m2)]
		counter+=1
		if(counter >1000):
			raise Exception(f"okay we generated more than 1000 lineraly dependent vectors in a row, something is probably wrong")

	return [np.array(v1), np.array(v2)]


#used to set the value of basis generator
# for method...
# give "default_list" to be using the existing basis_list defined above
# "random_generator" to use the random_basis function
# "custom_list" to use the list 
def select_init_method(method, custom_list): 
	if method == "default":
		basis_generator = pick_from_basis_list
	elif method == "random_generator":
		basis_generator = random_basis
	elif method == "custom_list":
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
MAX_STEP = 75



def LagrangeReduce(v1,v2):
	assert np.shape(v1) == (2,), f"LagrangeReduce expects v1 to be shape (2,0), v1 is ${v1} and v2 is ${v2}"
	assert np.shape(v2) == (2,), f"LagrangeReduce expects v1 to be shape (2,0), v1 is ${v1} and v2 is ${v2}"
	assert pairVectorsR2LinearIndep(v1,v2)

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




END = 0
S = 1
T = 2

	
#states will be list of...
#2 np arrays of legnth 2 (2 linearly indep vectors of dim 2)
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
		
		start_basis= basis_generator()
		smallest_vector = LagrangeReduce(start_basis[0], start_basis[1])[0]
		smallest_m = np.linalg.norm(smallest_vector)


		return start_basis+ [smallest_m, False]

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
		return ( state[-2]/min(magnitudes))**2 -0.0001*step_idx



