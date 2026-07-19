import argparse
import json
parser = argparse.ArgumentParser()

def get_input():
    parser.add_argument("--dump_path", type=str, default="", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",help="Experiment ID")

    parser.add_argument('--eval_only', action='store_true')

    parser.add_argument("--init_method", type=str, default="default", 
        help="""specify the method used to choose inputs/starting positions for the agent. look for the select_init method in the env for the options
        """)
    parser.add_argument('--custom_init_list', type = json.loads, help ="use with --init_method set to 2. pass in your input as a list of possible valid starting states. And the list should be surrounded with quotes. so eg \"[ [[1,2], [3,4]], [[5,6],[7,8]] ]\" ")

    parser.add_argument('--custom_init_list_path', type ="str", help ="use with --init_method set to 3. pass in the path to a file where each line is a possible starting input")
    parser.add_argument("--max_step", type=int, default=50, 
        help="the max step number the agent can take before we end the episode. honestly i don't know if steps start from 0 or 1 right now...")



    parser.add_argument("--num_simulations", type=int, default=200, help="number of simulations before a step is taken in MCTS")

    parser.add_argument("--c_puct", type= float, default = 1.38)


    parser.add_argument("--reload_model", type=str, default="",help="path to model to be loaded. ignored if it's an empty string")

    parser.add_argument("--save_periodic", type=int, default=0,help="Save the model periodically (0 to disable)")

    parser.add_argument("--memory_size", type=int, default=200, help="number of most recent datapoints from MCTS to keep to train neural net with")


    parser.add_argument("--eval_freq", type=int, default=50, help="how often should we evaluate the model")
    parser.add_argument("--num_eval_iterations", type=int, default=1, help="how many iterations to run when evaluating the model")



    parser.add_argument("--batch_size", type=int, default=32, help="Number of datapoints per batch in training the neural net")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate used to train the neural net")

    #probably smth about the optimizer and/or decay?

    parser.add_argument("--architecture", type=str, default="encoder", help="architecture of the eural net used for the policy and value estimates")
    parser.add_argument("--n_layers", type=int, default=6, help="number of layers used in neural net used for policy and value estimates")


    return parser.parse_args()

