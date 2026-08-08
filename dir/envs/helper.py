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
