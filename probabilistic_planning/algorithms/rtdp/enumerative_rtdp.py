from probabilistic_planning.structures import EnumerativeValueFunction

import numpy as np

def compute_maximum_residual(mdp, first_value_function, second_value_function):
    state_residual = lambda state: abs(first_value_function[state] - second_value_function[state])
    residuals = map(state_residual, mdp.states)
    return max(residuals)

def compute_bellman_backup(state, mdp, gamma, value_function):
    qualities = []

    for action in mdp.actions:
        qualities.append(
            mdp.compute_infinite_horizon_quality(state, action, gamma, value_function)
        )

    return max(qualities)

def compute_greedy_action(state, mdp, gamma, value_function):
    policy = {}

    best_action = None
    max_value_for_best_action = float("-inf ")

    for action in mdp.actions:
        quality = mdp.compute_infinite_horizon_quality(state, action, gamma, value_function)

        if quality > max_value_for_best_action:
            max_value_for_best_action = quality
            best_action = action

    return best_action

def sample_next_state(state, action, mdp):
    sampled_probability = np.random.random_sample()
    cummulative_probability = 0.0

    for next_state in mdp.states:
        probability = mdp.transition(state, action, next_state)
        cummulative_probability = cummulative_probability + probability

        if sampled_probability < cummulative_probability:
            return next_state

    return mdp.states[-1] # return last state

def compute_policy(mdp, gamma, value_function):
    policy = {}

    for state in mdp.states:
        policy[state] = compute_greedy_action(state, mdp, gamma, value_function)

    return policy

def residual(state, mdp, gamma, value_function):
    action = compute_greedy_action(state, mdp, gamma, value_function)
    quality = mdp.compute_infinite_horizon_quality(state, action, gamma, value_function)

    return abs(value_function[state] - quality)

def initial_states_residuals(mdp, gamma, value_function):
    residuals = []

    for initial_state in mdp.initial_states:
        residuals.append(residual(initial_state, mdp, gamma, value_function))

    return residuals

def all_initial_states_converged(mdp, gamma, value_function, epsilon):
    residuals = initial_states_residuals(mdp, gamma, value_function)

    for residual in residuals:
        if residual > epsilon:
            return False

    return True

def enumerative_rtdp(mdp, gamma, max_trials, max_depth, epsilon = None, initial_value_function = None, seed = None):
    """Executes the Real Time Dynamic Programming algorithm.

    Parameters:
    mdp (EnumerativeMDP): enumerative Markov Decison Problem to be solved
    gamma (float): discount factor applied to solve this MDP (assumes infinite on indefinite horizon)
    max_trials (int): maximum number of trials to execute (ignored if an epsilon is informed)
    max_depth (int): max depth to search (used to avoid infinite loops on deadends)
    epsilon (float): optional maximum residual allowed between V_k and V_{k+1}
    initial_value_function (EnumerativeValueFunction): initial value function to start the algorithm. If this value
                                                       is ommited, the algorithm will consider a initial value function that
                                                       that returns 0 (zero) for all states.
    seed (int): optional seed used to initialize random number generator

    Returns:
    policy (dict): resulting policy computed for a mdp, represented as a dict that maps a state to an action
    value_function (dict): value function found by this algorithm, represented as a dict that maps a state to a float
    statistics (dict): dictionary containing some statistics about the algorithm execution. We have three statistics here:
                      "iterations" that is equal to the horizon parameter, "bellman_backups_done" that is the overall
                      number of Bellman backups executed and "maximum_residuals" that is the maximum residual found in
                      each iteration.
    """
    value_function = initial_value_function

    if value_function is None:
        value_function = EnumerativeValueFunction(mdp.states, lambda state: 0) # value function with zeroes

    if seed is not None:
        np.random.seed(seed)

    bellman_backups_done = 0
    trials = 0
    maximum_residuals = []

    while True:
        trials = trials + 1
        visited_states = []
        initial_state = np.random.choice(mdp.initial_states)

        state = initial_state

        while state not in mdp.goal_states:
            visited_states.append(state)

            value_function[state] = compute_bellman_backup(state, mdp, gamma, value_function)
            bellman_backups_done = bellman_backups_done + 1

            next_action = compute_greedy_action(state, mdp, gamma, value_function)
            state = sample_next_state(state, next_action, mdp)

            if len(visited_states) > max_depth:
                break

        maximum_residuals.append(max(initial_states_residuals(mdp, gamma, value_function)))

        if trials >= max_trials and epsilon is None:
            break

        if epsilon is not None and all_initial_states_converged(mdp, gamma, value_function, epsilon):
            break

    # compute policy
    policy = compute_policy(mdp, gamma, value_function)

    statistics = {
        "iterations": trials,
        "bellman_backups_done": bellman_backups_done,
        "maximum_residuals": maximum_residuals
    }

    return policy, value_function, statistics
