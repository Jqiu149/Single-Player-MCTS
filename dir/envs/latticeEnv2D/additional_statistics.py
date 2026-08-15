#plan is for each problem to write any additional statistic functions that are problem or env specific in the problem specific helper function or in the env file 
#for now i think we'll just assume that taking in observation list and final state should be enough, if more comes up we'll add to them
#outputs will be either a single numeric value or a list of numeric values. the function will be run on everything in the evaluation set and then also averaged over the evaluation set

#each env file will contain a dictionary statistic functions with keys being the name of the statistic and value being the statistic function




def lagrange_step_counts(obs_list):
    return LagrangeReduce(obs_list[0][0].astype(int), obs_list[0][1].astype(int), returnSteps = True)

statistic_functions= {
        "lagrange_step_counts": lagrange_step_counts
        }
