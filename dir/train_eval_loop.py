import numpy as np
import matplotlib.pyplot as plt

from .trainer import Trainer 
from .replay_memory import ReplayMemory
from .mcts import execute_episode
from .mcts import execute_episode_eval


import argparse
parser = argparse.ArgumentParser()




parser.add_argument("--dump_path", type=str, default="", help="Experiment dump path")
parser.add_argument("--exp_name", type=str, default="debug",help="Experiment name")
parser.add_argument("--save_periodic", type=int, default=0,help="Save the model periodically (0 to disable)")
parser.add_argument("--exp_id", type=str, default="",help="Experiment ID")


parser.add_argument("--batch_size", type=int, default=32, help="Number of datapoints per batch in training the neural net")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate used to train the neural net")


parser.add_argument("--architecture", type=str, default="encoder", help="architecture of the eural net used for the policy and value estimates")

parser.add_argument("--n_layers", type=int, default=6, help="number of layers used in neural net used for policy and value estimates")





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


#from .lattice_ffn_policy import Policy
#from .lattice_2d_env_magnitude.env import Env
#
#n_vectors = 2
#vector_dim = 2
#n_hidden_dim= 20
#n_actions = 3
#
#trainer=Trainer( lambda: Policy(n_vectors, vector_dim, n_hidden_dim, n_actions))
#
#
#obs_shape = [n_vectors, vector_dim]
#+++++++++++++++

from .lattice_encoder_policy import Policy
from .lattice_2d_env_magnitude.env import Env

n_enc_layers = 6  
n_vectors = 2
vector_dim = 2 
encoder_nhead=2 # needs to divide vector_dim...
n_actions=3

trainer=Trainer( lambda: Policy(n_enc_layers, vector_dim, encoder_nhead, n_actions))


obs_shape = [n_vectors, vector_dim]
#++++++++++++









network = trainer.step_model

#----------------

# actual train_eval loop stuff is below here ig


num_simulations = 15
memory_size = 200
num_eval_iterations = 1




mem = ReplayMemory(memory_size,
                   { "ob": np.long,
                     "pi": np.float32,
                     "return": np.float32},
                   { "ob":obs_shape,
                     "pi": [n_actions],
                     "return": []},
                   batch_size = 32)


def test_agent():
    obs, pis, returns, reward, done_state, action_list= execute_episode_eval(network,
                                                                 num_simulations,
                                                                 Env )
    print("observation list:")
    print(obs)
    print(action_list)
    print(f"final state: {done_state}")
    return reward


def loop():
    value_losses = []
    policy_losses = []

    for i in range(1,10000):
        if i % 50== 0:
            total_reward = 0
            for j in range(num_eval_iterations): 
                total_reward+= test_agent()
            
            print(f"step{i}, avg reward: {total_reward/num_eval_iterations}")
            
#            plt.plot(value_losses, label="value loss")
#            plt.plot(policy_losses, label="policy loss")
#            plt.legend()
#            plt.show()

        obs, pis, returns, total_reward, done_state = execute_episode(network,
                                                                 num_simulations,
                                                                 Env)
        mem.add_all({"ob": obs, "pi": pis, "return": returns})

        batch = mem.get_minibatch()
        vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
        value_losses.append(vl)
        policy_losses.append(pl)

