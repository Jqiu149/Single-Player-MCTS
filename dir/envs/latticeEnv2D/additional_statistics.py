import numpy as np
from problem_specific_helpers import *
#plan is for each problem to write any additional statistics that are problem specific in a file in the folder. in additon to like helper functions used for them i guess


#i think will need to assume all take in the same info so somethign to think about a little
#i think game states /observations should be enough but idk
##issue is that like..... okay if passing in obs list, we kinda need to assume that structure of observations is the same in everything.... or at least the parts relevent to the statistic(s)


#also need to standardize output format somehow...
#i think for now we'll assume everything is number or list of numbers for like giving multiple stats with one function ig

#and i think we wnat like names for statistics...

statistic_functions = {}


def lagrange_step_counts(obs_list):
    return LagrangeReduce(obs_list[0][0], obs_list[0][1])

statistics_functions["lagrange_step_counts"] = (lagrange_step_counts)
