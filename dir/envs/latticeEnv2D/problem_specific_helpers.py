import numpy as np
import random
from functools import partial
from scipy.stats import loguniform
from .. helper import *


def pairVectorsR2LinearIndep(v1,v2):
	return	v1[0]*v2[1] != v2[0]*v1[1]


def polarToCartesian( angle, magnitude): 
	return [np.cos(angle)*magnitude, np.sin(angle)*magnitude]


#generator stuff

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


#for if you want to choose a basis randomly
#will generate a pair of linearly independent 2d integer vectors 
# we'll probably need to work on this to like check if we're hapy with the distribution this guves but... for now it will probably maybe run?

#min magnitude is going to be 10 ig b/c we're doing integers and i think after rounding it gives a distribution i like more this way...
def random_basis(m=10000,minAngleDiff=1e-4*2*np.pi, maxAngleDiff=2*np.pi):
	
	m1 = random.uniform(1,m/10)
	m2 = random.uniform(1,m/10)
	a1 = random.uniform(0, 2*np.pi)
	a2 = a1+loguniform.rvs(minAngleDiff,maxAngleDiff)

	v1 = [int(10*x) for x in polarToCartesian(a1, m1)]
	v2 = [int(10*x) for x in polarToCartesian(a2, m2)]

	if(v1 == [0,0]):
	  v1[0] = 1

	counter = 0
	while( not pairVectorsR2LinearIndep(v1, v2)):
		m2 = random.uniform(1,m/10)
		a2 = a1+loguniform.rvs(minAngleDiff,maxAngleDiff)
		if random.uniform(0,1)> 0.5 :
			a2 += np.pi
		v2 = [int(x*10) for x in polarToCartesian(a2, m2)]
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
	method, args = parse_init_method(method)
	if method == "default":
		return pick_from_basis_list

	elif method == "random_generator":
		return partial(random_basis, **args)
		
	elif method== "custom_list":
		assert np.shape(custom_list)[1:] == (2,2), f"custom_list shape is {np.shape(custom_list)}"
		custom_list = [ [np.array(vector) for vector in vector_list ] for vector_list in custom_list]
		assert all( pairVectorsR2LinearIndep(basis_vectors[0],basis_vectors[1]) for basis_vectors in custom_list)

		global basis_list
		basis_list = custom_list

		print("basis_list is:", basis_list)

		return pick_from_basis_list

	else:
		ValueError (f"method chosen isn't one of the options, given {method}")









def LagrangeReduce(v1,v2, returnSteps= False):
	assert np.shape(v1) == (2,), f"LagrangeReduce expects v1 to be shape (2,0), v1 is ${v1} and v2 is ${v2}"
	assert np.shape(v2) == (2,), f"LagrangeReduce expects v1 to be shape (2,0), v1 is ${v1} and v2 is ${v2}"
	assert pairVectorsR2LinearIndep(v1,v2), f"v1:{v1}, v2:{v2}"

	norm1Squared = np.dot(v1,v1) 
	norm2Squared = np.dot(v2,v2) 

	done = False
	swap_count = 0
	subtract_count = 0
	while(not done):

		if(norm1Squared> norm2Squared):
			v1,v2 = v2,v1
			norm1Squared,norm2Squared = norm2Squared,norm1Squared

			swap_count+=1

		u = round( (np.dot(v1,v2))/ norm1Squared)
		v2 = v2-u*v1
		norm2Squared= np.dot(v2,v2) 

		subtract_count += abs(u)

		if(norm1Squared<= norm2Squared):
			done = True

	if returnSteps:
		return [swap_count, subtract_count]

	return [v1, v2]





# statistic functions
#plan is for each problem to write any additional statistic functions that are problem or env specific in the problem specific helper function or in the env file 
#for now i think we'll just assume that taking in observation list and final state should be enough, if more comes up we'll add to them
#outputs will be either a single numeric value or a list of numeric values. the function will be run on everything in the evaluation set and then also averaged over the evaluation set

#each env file will contain a dictionary statistic functions with keys being the name of the statistic and value being the statistic function




#assumes observations store the basis for the lattice as the first two members of an observation
def lagrange_step_counts(obs_list, final_state):
	return LagrangeReduce(obs_list[0][0].astype(int), obs_list[0][1].astype(int), returnSteps = True)


#assumes state second last member is the magnitude of the minimum size vector
def reached_min_mag(obs_list, final_state):
	min_magnitude= min([np.linalg.norm(v) for v in final_state[0:2]])
	return final_state[-2] == min_magnitude
   
