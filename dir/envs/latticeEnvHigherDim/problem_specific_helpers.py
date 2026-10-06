import numpy as np
import random
from functools import partial
from scipy.stats import loguniform
from .. helper import *



def reached_min_mag(obs_list, start_state, final_state):
	min_magnitude= min([np.linalg.norm(v) for v in final_state[0]])
	return final_state[-2] == min_magnitude
 
