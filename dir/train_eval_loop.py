"""
Example program that uses the single-player MCTS algorithm to train an agent
to master the HillClimbingEnvironment, in which the agent has to reach the
highest point on a map.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from .trainer import Trainer 
from .replay_memory import ReplayMemory
from .mcts import execute_episode
from .mcts import execute_episode_eval


from .hill_climbing_example.policy import HillClimbingPolicy
from .hill_climbing_example.hill_climbing_env import HillClimbingEnv

def log(test_env, iteration, step_idx, total_rew):
    """
    Logs one step in a testing episode.
    :param test_env: Test environment that should be rendered.
    :param iteration: Number of training iterations so far.
    :param step_idx: Index of the step in the episode.
    :param total_rew: Total reward collected so far.
    """
    print()
    print(f"Training Episodes: {iteration}")
    test_env.render()
    print(f"Step: {step_idx}")
    print(f"Return: {total_rew}")




n_actions = 4
n_obs = 49

trainer = Trainer(lambda: HillClimbingPolicy(n_obs, 20, n_actions))
network = trainer.step_model

mem = ReplayMemory(200,
                   { "ob": np.long,
                     "pi": np.float32,
                     "return": np.float32},
                   { "ob": [],
                     "pi": [n_actions],
                     "return": []})

def test_agent(iteration):
    test_env = HillClimbingEnv()
    obs, pis, returns, total_reward, done_state, action_list= execute_episode_eval(network,
                                                                 32,
                                                                 HillClimbingEnv)
     

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
            print(f"avg reward: {total_reward/n}")
            logits, p,v = network(np.array([i for i in range(49)]))
            for i in range(7):
                print(v[7*i:7*(i+1)])
            
            plt.plot(value_losses, label="value loss")
            plt.plot(policy_losses, label="policy loss")
            plt.legend()
            plt.show()

        obs, pis, returns, total_reward, done_state = execute_episode(network,
                                                                 32,
                                                                 HillClimbingEnv)
        mem.add_all({"ob": obs, "pi": pis, "return": returns})

        batch = mem.get_minibatch()


        vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
        value_losses.append(vl)
        policy_losses.append(pl)

