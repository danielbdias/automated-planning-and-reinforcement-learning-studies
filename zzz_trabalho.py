import gym
import numpy as np
import time
import pickle

from reinforcement_learning.algorithms.value_based import q_learning, sarsa
from reinforcement_learning.algorithms.policy_search import reinforce

# environment setup
env = gym.make('CartPole-v1')

def discretize_state(state, truncate_digits = 4):
    values = []

    for state_value in state:
        truncated_state_value = int(np.trunc(state_value * (10 ** truncate_digits)))
        values.append(str(truncated_state_value))

    return "_".join(values)

def save_statistics(stats, file_name):
    pickle.dump(stats.to_tuple(), open(file_name, "wb"))

episodes = 5000

# Q-Learning (Tabular)
print("Running Q-Learning...")
# start_time = time.perf_counter()
# _, q_learning_d_10_alpha_09_epsilon01_stats = q_learning(env, episodes, discount_factor=1.0, alpha=0.9, epsilon=0.1, discretize_state_function=discretize_state)
# _, q_learning_d_10_alpha_05_epsilon05_stats = q_learning(env, episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5, discretize_state_function=discretize_state)
# _, q_learning_d_10_alpha_01_epsilon09_stats = q_learning(env, episodes, discount_factor=1.0, alpha=0.1, epsilon=0.9, discretize_state_function=discretize_state)
# _, q_learning_d_09_alpha_09_epsilon01_stats = q_learning(env, episodes, discount_factor=0.9, alpha=0.9, epsilon=0.1, discretize_state_function=discretize_state)
# _, q_learning_d_09_alpha_05_epsilon05_stats = q_learning(env, episodes, discount_factor=0.9, alpha=0.5, epsilon=0.5, discretize_state_function=discretize_state)
# _, q_learning_d_09_alpha_01_epsilon09_stats = q_learning(env, episodes, discount_factor=0.9, alpha=0.1, epsilon=0.9, discretize_state_function=discretize_state)
# _, q_learning_d_05_alpha_09_epsilon01_stats = q_learning(env, episodes, discount_factor=0.5, alpha=0.9, epsilon=0.1, discretize_state_function=discretize_state)
# _, q_learning_d_05_alpha_05_epsilon05_stats = q_learning(env, episodes, discount_factor=0.5, alpha=0.5, epsilon=0.5, discretize_state_function=discretize_state)
# _, q_learning_d_05_alpha_01_epsilon09_stats = q_learning(env, episodes, discount_factor=0.5, alpha=0.1, epsilon=0.9, discretize_state_function=discretize_state)
# elapsed_time = time.perf_counter() - start_time
# print(f"Elapsed time: {elapsed_time:0.4f} seconds")

print("Saving Q-Learning results...")
save_statistics(q_learning_d_10_alpha_09_epsilon01_stats, "./pickles/q_learning_d_10_alpha_09_epsilon01_stats.pickle")
save_statistics(q_learning_d_10_alpha_05_epsilon05_stats, "./pickles/q_learning_d_10_alpha_05_epsilon05_stats.pickle")
save_statistics(q_learning_d_10_alpha_01_epsilon09_stats, "./pickles/q_learning_d_10_alpha_01_epsilon09_stats.pickle")
save_statistics(q_learning_d_09_alpha_09_epsilon01_stats, "./pickles/q_learning_d_09_alpha_09_epsilon01_stats.pickle")
save_statistics(q_learning_d_09_alpha_05_epsilon05_stats, "./pickles/q_learning_d_09_alpha_05_epsilon05_stats.pickle")
save_statistics(q_learning_d_09_alpha_01_epsilon09_stats, "./pickles/q_learning_d_09_alpha_01_epsilon09_stats.pickle")
save_statistics(q_learning_d_05_alpha_09_epsilon01_stats, "./pickles/q_learning_d_05_alpha_09_epsilon01_stats.pickle")
save_statistics(q_learning_d_05_alpha_05_epsilon05_stats, "./pickles/q_learning_d_05_alpha_05_epsilon05_stats.pickle")
save_statistics(q_learning_d_05_alpha_01_epsilon09_stats, "./pickles/q_learning_d_05_alpha_01_epsilon09_stats.pickle")
print("Q-Learning saved.")

# SARSA (Tabular)
# print("Running SARSA...")
# start_time = time.perf_counter()
# _, sarsa_d_10_alpha_09_epsilon01_stats = sarsa(env, episodes, discount_factor=1.0, alpha=0.9, epsilon=0.1, discretize_state_function=discretize_state)
# _, sarsa_d_10_alpha_05_epsilon05_stats = sarsa(env, episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5, discretize_state_function=discretize_state)
# _, sarsa_d_10_alpha_01_epsilon09_stats = sarsa(env, episodes, discount_factor=1.0, alpha=0.1, epsilon=0.9, discretize_state_function=discretize_state)
# _, sarsa_d_09_alpha_09_epsilon01_stats = sarsa(env, episodes, discount_factor=0.9, alpha=0.9, epsilon=0.1, discretize_state_function=discretize_state)
# _, sarsa_d_09_alpha_05_epsilon05_stats = sarsa(env, episodes, discount_factor=0.9, alpha=0.5, epsilon=0.5, discretize_state_function=discretize_state)
# _, sarsa_d_09_alpha_01_epsilon09_stats = sarsa(env, episodes, discount_factor=0.9, alpha=0.1, epsilon=0.9, discretize_state_function=discretize_state)
# _, sarsa_d_05_alpha_09_epsilon01_stats = sarsa(env, episodes, discount_factor=0.5, alpha=0.9, epsilon=0.1, discretize_state_function=discretize_state)
# _, sarsa_d_05_alpha_05_epsilon05_stats = sarsa(env, episodes, discount_factor=0.5, alpha=0.5, epsilon=0.5, discretize_state_function=discretize_state)
# _, sarsa_d_05_alpha_01_epsilon09_stats = sarsa(env, episodes, discount_factor=0.5, alpha=0.1, epsilon=0.9, discretize_state_function=discretize_state)
# elapsed_time = time.perf_counter() - start_time
# print(f"Elapsed time: {elapsed_time:0.4f} seconds")

print("Saving SARSA results...")
save_statistics(sarsa_d_10_alpha_09_epsilon01_stats, "./pickles/sarsa_d_10_alpha_09_epsilon01_stats.pickle")
save_statistics(sarsa_d_10_alpha_05_epsilon05_stats, "./pickles/sarsa_d_10_alpha_05_epsilon05_stats.pickle")
save_statistics(sarsa_d_10_alpha_01_epsilon09_stats, "./pickles/sarsa_d_10_alpha_01_epsilon09_stats.pickle")
save_statistics(sarsa_d_09_alpha_09_epsilon01_stats, "./pickles/sarsa_d_09_alpha_09_epsilon01_stats.pickle")
save_statistics(sarsa_d_09_alpha_05_epsilon05_stats, "./pickles/sarsa_d_09_alpha_05_epsilon05_stats.pickle")
save_statistics(sarsa_d_09_alpha_01_epsilon09_stats, "./pickles/sarsa_d_09_alpha_01_epsilon09_stats.pickle")
save_statistics(sarsa_d_05_alpha_09_epsilon01_stats, "./pickles/sarsa_d_05_alpha_09_epsilon01_stats.pickle")
save_statistics(sarsa_d_05_alpha_05_epsilon05_stats, "./pickles/sarsa_d_05_alpha_05_epsilon05_stats.pickle")
save_statistics(sarsa_d_05_alpha_01_epsilon09_stats, "./pickles/sarsa_d_05_alpha_01_epsilon09_stats.pickle")
print("SARSA saved.")

# REINFORCE (Policy search)
print("Running REINFORCE...")
start_time = time.perf_counter()
reinforce_d_10_stats = reinforce(env, episodes, discount_factor=1.0)
reinforce_d_09_stats = reinforce(env, episodes, discount_factor=0.9)
reinforce_d_05_stats = reinforce(env, episodes, discount_factor=0.5)
elapsed_time = time.perf_counter() - start_time
print(f"Elapsed time: {elapsed_time:0.4f} seconds")

print("Saving REINFORCE results...")
save_statistics(reinforce_d_10_stats, "./pickles/reinforce_d_10_stats.pickle")
save_statistics(reinforce_d_09_stats, "./pickles/reinforce_d_09_stats.pickle")
save_statistics(reinforce_d_05_stats, "./pickles/reinforce_d_05_stats.pickle")
print("REINFORCE saved.")
