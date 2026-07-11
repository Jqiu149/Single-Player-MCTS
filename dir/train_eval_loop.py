import numpy as np
import matplotlib.pyplot as plt

from .trainer import Trainer 
from .replay_memory import ReplayMemory
from .mcts import execute_episode
from .mcts import execute_episode_eval






# setup traininer ( which is just choose your policy and pass in paramaters to it
# -------------
#ig we'll just change this whenever we want to test different thing?

#from .hill_climbing_example.policy import HillClimbingPolicy as Policy
#from .hill_climbing_example.hill_climbing_env import HillClimbingEnv as Env
#n_obs = 49 #... only needed for thier policy netowrk...

#trainer = Trainer(lambda: Policy(n_obs, 20, n_actions)) # how we initalize netowrk dependson our netwrok.....



from .lattice_ffn_policy import Policy
from .lattice_2d_env_magnitude.env import Env

num_vectors = 2
vector_dim = 2
num_encoder_layers = 6
encoder_nhead= 2 # needs to divide the vector dim ig?
n_actions = 3

trainer=Trainer( lambda: Policy(num_vectors, vector_dim, encoder_nhead, n_actions))



#from .lattice_ffn_policy import Policy
#from .2d_lattice_env_0.env import Env







network = trainer.step_model

#----------------

# actual train_eval loop stuff is below here ig



num_simulations = 40
memory_size = 200




obs_shape = [num_vectors, vector_dim]
mem = ReplayMemory(memory_size,
                   { "ob": np.float32,
                     "pi": np.float32,
                     "return": np.float32},
                   { "ob":obs_shape,
                     "pi": [n_actions],
                     "return": []})


def test_agent(iteration):
    obs, pis, returns, total_reward, done_state, action_list= execute_episode_eval(network,
                                                                 num_simulations,
                                                                 Env )
    return total_reward


def loop():
    value_losses = []
    policy_losses = []

    for i in range(10000):
        if i % 100 == 0:
            n=100
            total_reward = 0
            for j in range(n):
                total_reward+=test_agent(i)
            
            print(f"step{i}, avg reward: {total_reward/n}")
            
            plt.plot(value_losses, label="value loss")
            plt.plot(policy_losses, label="policy loss")
            plt.legend()
            plt.show()

        obs, pis, returns, total_reward, done_state = execute_episode(network,
                                                                 num_simulations,
                                                                 Env)
        mem.add_all({"ob": obs, "pi": pis, "return": returns})

        batch = mem.get_minibatch()


        vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
        value_losses.append(vl)
        policy_losses.append(pl)

