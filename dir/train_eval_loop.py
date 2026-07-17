import numpy as np
import matplotlib.pyplot as plt
import torch
import pathlib

from .trainer import Trainer 
from .replay_memory import ReplayMemory
from .mcts import execute_episode
from .mcts import execute_episode_eval


import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--dump_path", type=str, default="", help="Experiment dump path")
parser.add_argument("--exp_name", type=str, default="",help="Experiment name")
parser.add_argument("--exp_id", type=str, default="",help="Experiment ID")


parser.add_argument("--reload_model", type=str, default="",help="path to model to be loaded. ignored if it's an empty string")

parser.add_argument("--save_periodic", type=int, default=0,help="Save the model periodically (0 to disable)")

parser.add_argument("--num_simulations", type=int, default=300, help="number of simulations before a step is taken in MCTS")


parser.add_argument("--memory_size", type=int, default=200, help="number of most recent datapoints from MCTS to keep to train neural net with")


parser.add_argument("--eval_freq", type=int, default=50, help="how often should we evaluate the model")
parser.add_argument("--num_eval_iterations", type=int, default=1, help="how many iterations to run when evaluating the model")



parser.add_argument("--batch_size", type=int, default=32, help="Number of datapoints per batch in training the neural net")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate used to train the neural net")


parser.add_argument("--architecture", type=str, default="encoder", help="architecture of the eural net used for the policy and value estimates")
parser.add_argument("--n_layers", type=int, default=6, help="number of layers used in neural net used for policy and value estimates")


args = parser.parse_args()


assert args.dump_path != "" and args.exp_name != "" and args.exp_id !="", "one of dump_path, exp_name, exp_id wasn't specified"

assert args.batch_size < args.memory_size

model_save_state_dir= args.dump_path + "/" + args.exp_name + "/" + args.exp_id  + '/'
model_save_state_path = model_save_state_dir + "checkpoint.pth"
model_load_path = args.reload_path if args.reload_model!="" else model_save_state_path



# setup traininer ( which is just choose your policy and pass in paramaters to it
# -------------
#ig we'll just change this whenever we want to test different thing?



#++++++++++++++

#from .hill_climbing_example.policy import HillClimbingPolicy as Policy
#from .hill_climbing_example.hill_climbing_env import HillClimbingEnv as Env
#n_actions = 4
#n_obs = 49 #... only needed for thier policy netowrk...
#
#trainer = Trainer(lambda: Policy(n_obs, 20, n_actions)) # how we initalize netowrk dependson our netwrok.....
#

#obs_shape = []

#+++++++++++++



#+++++++++++++++

from .lattice_encoder_policy import Policy
from .lattice_env import Env

n_enc_layers = 6  
n_vectors = 2
vector_dim = 2 
encoder_nhead=2 # needs to divide vector_dim...
n_actions=3

trainer=trainer=Trainer( lambda: Policy(n_enc_layers, vector_dim, encoder_nhead, n_actions), model_path=model_load_path )

obs_shape = [n_vectors, vector_dim]
#++++++++++++



network = trainer.step_model



#----------------

# actual train_eval loop stuff is below here ig





num_eval_iterations = 1

mem = ReplayMemory(args.memory_size,
                   { "ob": np.long,
                     "pi": np.float32,
                     "return": np.float32},
                   { "ob":obs_shape,
                     "pi": [n_actions],
                     "return": []},
                   batch_size = args.batch_size)


def test_agent(num_iterations):
    network.eval()
    total_reward= 0

    for i in range(num_iterations):
        obs, pis, returns, reward, done_state, action_list= execute_episode_eval(network,
                                                                 args.num_simulations,
                                                                 Env )
        print("observation list:")
        print(obs)
        print(action_list)
        print(f"final state: {done_state}")

        total_reward+=reward
    

    print(f"avg_reward:{total_reward/num_iterations}")
    
          

def loop():
    value_losses = []
    policy_losses = []

    for i in range(1,20001):
        if i % args.eval_freq== 0:
            test_agent(num_eval_iterations)
            
            
#            plt.plot(value_losses, label="value loss")
#            plt.plot(policy_losses, label="policy loss")
#            plt.legend()
#            plt.show()

        obs, pis, returns, total_reward, done_state = execute_episode(network,
                                                                 args.num_simulations,
                                                                 Env)
        mem.add_all({"ob": obs, "pi": pis, "return": returns})

        batch = mem.get_minibatch()
        vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
        value_losses.append(vl)
        policy_losses.append(pl)

    

    pathlib.Path(model_save_state_dir).mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), model_save_state_path)
