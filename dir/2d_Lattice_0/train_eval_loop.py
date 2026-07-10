"""
Example program that uses the single-player MCTS algorithm to train an agent
to master the HillClimbingEnvironment, in which the agent has to reach the
highest point on a map.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from ..trainer import Trainer
from .2dlattice_policy-0 import policy
from ..replay_memory import ReplayMemory
from .2dlattice_env-0 0import env
from .mcts import execute_episode


def log(test_env, iteration, step_idx, total_rew):
    """
    Logs one step in a testing episode.
    :param test_env: Test environment that should be rendered.
    :param iteration: Number of training iterations so far.
    :param step_idx: Index of the step in the episode.
    :param total_rew: Total reward collected so far.
    """
    time.sleep(0.3)
    print()
    print(f"Training Episodes: {iteration}")
    test_env.render()
    print(f"Step: {step_idx}")
    print(f"Return: {total_rew}")


if __name__ == '__main__':
    n_actions = 3
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
        total_rew = 0
