import gym
import numpy as np

from reinforcement_learning.algorithms.value_based import q_learning, sarsa

env = gym.make('CartPole-v1')

def discretize_state(state, truncate_digits = 4):
    values = []

    for state_value in state:
        truncated_state_value = int(np.trunc(state_value * (10 ** truncate_digits)))
        values.append(str(truncated_state_value))

    return "_".join(values)

episodes = 2000

#Q, stats = q_learning(env, episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, discretize_state_function=discretize_state)
Q, stats = q_learning(env, episodes, discount_factor=0.9, alpha=0.5, epsilon=0.5, discretize_state_function=discretize_state)

#Q, stats = sarsa(env, episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, discretize_state_function=discretize_state)
# Q, stats = sarsa(env, episodes, discount_factor=0.9, alpha=0.5, epsilon=0.5, discretize_state_function=discretize_state)

print("\n")
print(f"Episode Lenghts: {np.mean(stats.episode_lengths)}")
print(f"Episode Lenghts (First 10): {stats.episode_lengths[0:10]}")
print(f"Episode Lenghts (Last 10): {stats.episode_lengths[-10:]}")

print()
print(f"Episode Rewards: {np.mean(stats.episode_rewards)}")
print(f"Episode Rewards (First 10): {stats.episode_rewards[0:10]}")
print(f"Episode Rewards (Last 10): {stats.episode_rewards[-10:]}")

env.close()
