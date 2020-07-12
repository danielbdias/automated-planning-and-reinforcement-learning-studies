#!/usr/bin/env python
"""Implement Neural Fitted Q-Iteration.

http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf


Running
-------
You can train the NFQ agent on CartPole Regulator with the inluded
configuration file with the below command:
```
python train_eval.py -c cartpole.conf
```

For a reproducible run, use the RANDOM_SEED flag.
```
python train_eval.py -c cartpole.conf --RANDOM_SEED=1
```

To save a trained agent, use the SAVE_PATH flag.
```
python train_eval.py -c cartpole.conf --SAVE_PATH=saves/cartpole.pth
```

To load a trained agent, use the LOAD_PATH flag.
```
python train_eval.py -c cartpole.conf --LOAD_PATH=saves/cartpole.pth
```

To enable logging to TensorBoard or W&B, use appropriate flags.
```
python train_eval.py -c cartpole.conf --USE_TENSORBOARD --USE_WANDB
```


Logging
-------
1. You can view runs online via Weights & Biases (wandb):
https://app.wandb.ai/seungjaeryanlee/implementations-nfq/runs

2. You can use TensorBoard to view runs offline:
```
tensorboard --logdir=tensorboard_logs --port=2223
```


Glossary
--------
env : Environment
obs : Observation
"""
import configargparse
import torch
import torch.optim as optim

from reinforcement_learning.algorithms.sample_efficient.environments import CartPoleRegulatorEnv
from reinforcement_learning.algorithms.sample_efficient.nfq.agents import NFQAgent
from reinforcement_learning.algorithms.sample_efficient.nfq.networks import NFQNetwork
from reinforcement_learning.algorithms.sample_efficient.utils import get_logger, load_models, make_reproducible, save_models

from collections import namedtuple
import numpy as np

from reinforcement_learning.structures import EpisodeStats

AlgorithmConfig = namedtuple("AlgorithmConfig", [ "EPOCH", "TRAIN_ENV_MAX_STEPS", "EVAL_ENV_MAX_STEPS", "DISCOUNT", "INIT_EXPERIENCE", "INCREMENT_EXPERIENCE",
                                                  "HINT_TO_GOAL", "RANDOM_SEED", "TRAIN_RENDER", "EVAL_RENDER", "SAVE_PATH", "LOAD_PATH", "USE_TENSORBOARD", "USE_WANDB" ])

def neuro_fitt_q(epoch, train_env_max_steps, eval_env_max_steps, discount, init_experience = 0, seed = None):
    """Run NFQ."""
    CONFIG = AlgorithmConfig(
        EPOCH = epoch,
        TRAIN_ENV_MAX_STEPS = train_env_max_steps,
        EVAL_ENV_MAX_STEPS = eval_env_max_steps,
        DISCOUNT = discount,
        INIT_EXPERIENCE = init_experience,
        INCREMENT_EXPERIENCE = True,
        HINT_TO_GOAL = True,
        RANDOM_SEED = seed,
        TRAIN_RENDER = False,
        EVAL_RENDER = False,
        SAVE_PATH = "",
        LOAD_PATH = "",
        USE_TENSORBOARD = False,
        USE_WANDB = False,
    )

    # Log to File, Console, TensorBoard, W&B
    logger = get_logger()

    # Setup environment
    train_env = CartPoleRegulatorEnv(mode="train", max_steps=train_env_max_steps)
    eval_env = CartPoleRegulatorEnv(mode="eval", max_steps=eval_env_max_steps)

    # Fix random seeds
    if CONFIG.RANDOM_SEED is not None:
        make_reproducible(CONFIG.RANDOM_SEED, use_numpy=True, use_torch=True)
        train_env.seed(CONFIG.RANDOM_SEED)
        eval_env.seed(CONFIG.RANDOM_SEED)
    else:
        logger.warning("Running without a random seed: this run is NOT reproducible.")

    # Setup agent
    nfq_net = NFQNetwork()
    optimizer = optim.Rprop(nfq_net.parameters())
    nfq_agent = NFQAgent(nfq_net, optimizer)

    # Load trained agent
    # if CONFIG.LOAD_PATH:
    #     load_models(CONFIG.LOAD_PATH, nfq_net=nfq_net, optimizer=optimizer)

    # NFQ Main loop
    # A set of transition samples denoted as D
    all_rollouts = []
    total_cost = 0

    if CONFIG.INIT_EXPERIENCE:
        for _ in range(CONFIG.INIT_EXPERIENCE):
            rollout, episode_cost = train_env.generate_rollout(
                None, render=CONFIG.TRAIN_RENDER
            )
            all_rollouts.extend(rollout)
            total_cost += episode_cost

    stats = EpisodeStats(episode_lengths=np.zeros(CONFIG.EPOCH), episode_rewards=np.zeros(CONFIG.EPOCH))

    for epoch in range(CONFIG.EPOCH + 1):
        # Variant 1: Incermentally add transitions (Section 3.4)
        # TODO(seungjaeryanlee): Done before or after training?
        if CONFIG.INCREMENT_EXPERIENCE:
            new_rollout, episode_cost = train_env.generate_rollout(
                nfq_agent.get_best_action, render=CONFIG.TRAIN_RENDER
            )
            all_rollouts.extend(new_rollout)
            total_cost += episode_cost

        state_action_b, target_q_values = nfq_agent.generate_pattern_set(all_rollouts)

        # Variant 2: Clamp function to zero in goal region
        # TODO(seungjaeryanlee): Since this is a regulator setting, should it
        #                        not be clamped to zero?
        if CONFIG.HINT_TO_GOAL:
            goal_state_action_b, goal_target_q_values = train_env.get_goal_pattern_set()
            goal_state_action_b = torch.FloatTensor(goal_state_action_b)
            goal_target_q_values = torch.FloatTensor(goal_target_q_values)
            state_action_b = torch.cat([state_action_b, goal_state_action_b], dim=0)
            target_q_values = torch.cat([target_q_values, goal_target_q_values], dim=0)

        loss = nfq_agent.train((state_action_b, target_q_values))

        # TODO(seungjaeryanlee): Evaluation should be done with 3000 episodes
        eval_episode_length, eval_success, eval_episode_cost = nfq_agent.evaluate(
            eval_env, CONFIG.EVAL_RENDER
        )

        if eval_success:
            break

        #stats.episode_rewards[epoch] = eval_episode_cost
        stats.episode_rewards[epoch] = eval_episode_length + 1
        stats.episode_lengths[epoch] = eval_episode_length

    train_env.close()
    eval_env.close()

    return stats
