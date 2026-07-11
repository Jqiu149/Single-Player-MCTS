import numpy as np
import matplotlib.pyplot as plt

from .trainer import Trainer 
from .replay_memory import ReplayMemory
from .mcts import execute_episode
from .mcts import execute_episode_eval






# setup traininer ( which is just choose your policy and pass in paramaters to it
# -------------
#ig we'll just change this whenever we want to test different thing?
from .hill_climbing_example.policy import HillClimbingPolicy as Policy
from .hill_climbing_example.hill_climbing_env import HillClimbingEnv as Env
#from .lattice_ffn_policy import Policy
#from .2d_lattice_env_0.env import Env

n_obs = 49 #... only needed for thier policy netowrk...

n_actions = 4
trainer = Trainer(lambda: Policy(n_obs, 20, n_actions)) # how we initalize netowrk dependson our netwrok.....

network = trainer.step_model

#----------------

# actual train_eval loop stuff is below here ig




memory_size = 200


mem = ReplayMemory(memory_size,
                   { "ob": np.long,
                     "pi": np.float32,
                     "return": np.float32},
                   { "ob": [],
                     "pi": [n_actions],
                     "return": []})


def test_agent(iteration):
    obs, pis, returns, total_reward, done_state, action_list= execute_episode_eval(network,
                                                                 32,
                                                                Env)
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
                                                                 32,
                                                                 Env)
        mem.add_all({"ob": obs, "pi": pis, "return": returns})

        batch = mem.get_minibatch()


        vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
        value_losses.append(vl)
        policy_losses.append(pl)

