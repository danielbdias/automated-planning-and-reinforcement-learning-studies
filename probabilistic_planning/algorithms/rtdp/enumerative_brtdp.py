from probabilistic_planning.structures import EnumerativeValueFunction

import numpy as np

def compute_maximum_residual(mdp, first_value_function, second_value_function):
    state_residual = lambda state: abs(first_value_function[state] - second_value_function[state])
    residuals = map(state_residual, mdp.states)
    return max(residuals)

def compute_quality(state, action, mdp, gamma, value_function):
    pondered_expected_values = []

    for next_state in mdp.states:
        pondered_expected_values.append(
            mdp.transition(state, action, next_state) * value_function[next_state]
        )

    return mdp.reward(state) + gamma * sum(pondered_expected_values)

def compute_bellman_backup(state, mdp, gamma, value_function):
    qualities = []

    for action in mdp.actions:
        qualities.append(
            compute_quality(state, action, mdp, gamma, value_function)
        )

    return max(qualities)

def compute_greedy_action(state, mdp, gamma, value_function):
    policy = {}

    best_action = None
    max_value_for_best_action = float("-inf ")

    for action in mdp.actions:
        quality = compute_quality(state, action, mdp, gamma, value_function)

        if quality > max_value_for_best_action:
            max_value_for_best_action = quality
            best_action = action

    return best_action

def sample_next_state(normalizing_constant, bounded_weighted_probabilities):
    sampled_probability = np.random.random_sample() * normalizing_constant
    cummulative_probability = 0.0

    states = list(bounded_weighted_probabilities.keys())

    for next_state in states:
        bounded_probability = bounded_weighted_probabilities[next_state]
        cummulative_probability = cummulative_probability + bounded_probability

        if sampled_probability < cummulative_probability:
            return next_state

    return states[-1] # return last state

def compute_policy(mdp, gamma, value_function):
    policy = {}

    for state in mdp.states:
        policy[state] = compute_greedy_action(state, mdp, gamma, value_function)

    return policy

def compute_bounded_weighted_probabilities(state, action, mdp, lower_value_function, upper_value_function):
    normalizing_constant = 0
    bounded_weighted_probabilities = {}

    for next_state in mdp.states:
        probability = mdp.transition(state, action, next_state)
        bound = upper_value_function[next_state] - lower_value_function[next_state]

        bounded_weighted_probability = probability * bound
        bounded_weighted_probabilities[next_state] = bounded_weighted_probability
        normalizing_constant = normalizing_constant + bounded_weighted_probability

    return normalizing_constant, bounded_weighted_probabilities

def all_states_reached_converged(states, epsilon, lower_value_function, upper_value_function):
    for state in states:
        if (upper_value_function[state] - lower_value_function[state]) > epsilon:
            return False

    return True

def all_states_reached_bound(states, normalizing_constant, tau, lower_value_function, upper_value_function):
    for state in states:
        bound = (upper_value_function[state] - lower_value_function[state]) / tau

        if bound >= normalizing_constant:
            return False

    return True

def enumerative_brtdp(mdp, gamma, max_depth, epsilon, tau, initial_lower_value_function = None, initial_upper_value_function = None, seed = None):
    """Executes the Bounded Real Time Dynamic Programming algorithm.

    Parameters:
    mdp (EnumerativeMDP): enumerative Markov Decison Problem to be solved
    gamma (float): discount factor applied to solve this MDP (assumes infinite on indefinite horizon)
    trials (int): number of trials to execute
    max_depth (int): max depth to search (used to avoid infinite loops on deadends)
    epsilon (float): maximum residual allowed between V_upper and V_lower
    tau (float): adaptative criterion, used to check and limit V_upper and V_lower differences
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
    lower_value_function = initial_lower_value_function

    if lower_value_function is None:
        lower_value_function = EnumerativeValueFunction(mdp.states, lambda state: 0) # value function with zeroes

    upper_value_function = initial_upper_value_function

    if upper_value_function is None:
        upper_value_function = EnumerativeValueFunction(mdp.states, lambda state: 1 + epsilon) # value function with ones

    if seed is not None:
        np.random.seed(seed)

    bellman_backups_done = 0
    trials = 0

    while not all_states_reached_converged(mdp.initial_states, epsilon, lower_value_function, upper_value_function):
        trials = trials + 1
        visited_states = []
        initial_state = np.random.choice(mdp.initial_states)

        state = initial_state

        while True:
            visited_states.append(state)

            upper_value_function[state] = compute_bellman_backup(state, mdp, gamma, upper_value_function)
            bellman_backups_done = bellman_backups_done + 1

            next_action = compute_greedy_action(state, mdp, gamma, lower_value_function)
            lower_value_function[state] = compute_quality(state, next_action, mdp, gamma, lower_value_function)

            normalizing_constant, bounded_weighted_probabilities = compute_bounded_weighted_probabilities(state, next_action, mdp, lower_value_function, upper_value_function)

            if all_states_reached_bound(mdp.initial_states, normalizing_constant, tau, lower_value_function, upper_value_function):
                break

            state = sample_next_state(normalizing_constant, bounded_weighted_probabilities)

            if len(visited_states) > max_depth:
                break

        while len(visited_states) != 0:
            state = visited_states.pop()

            lower_value_function[state] = compute_bellman_backup(state, mdp, gamma, lower_value_function)
            upper_value_function[state] = compute_bellman_backup(state, mdp, gamma, upper_value_function)
            bellman_backups_done = bellman_backups_done + 2

    # compute policy
    policy = compute_policy(mdp, gamma, lower_value_function)

    statistics = {
        "iterations": trials,
        "bellman_backups_done": bellman_backups_done
    }

    return policy, lower_value_function, statistics
