from functools import partial


#used for your select init method in your environment
#output is the string selecting the method, and a dictionary of the paramater name and corresponding value to be passed in to the function selected
#!!! right now the argument values for the selected function must be floats...
def parse_init_method(input):
  a = input.split(",")
  method = a[0]

  args = {}

  for arg_val_str in a[1:]:
    arg, val = arg_val_str.split("=")
    args[arg] = float(val)
  
  return method, args




def apply_flat_step_penalty(pre_penalty_reward, step_count,step_penalty):
    return pre_penalty_reward - step_count*step_penalty

def apply_percent_step_penalty(pre_penalty_reward, step_count,step_penalty):
    return pre_penalty_reward *(1- step_count*step_penalty)

def select_step_penalty(selection_string):
    penalty_type,args= parse_init_method(selection_string)

    if penalty_type == "flat":
        return partial(apply_flat_step_penalty, step_penalty = args["step_penalty"])
    elif penalty_type == "percent":
        return partial(apply_percent_step_penalty, step_penalty = args["step_penalty"])
    else:
        ValueError("penalty type string not known, got {penalty_type}")

