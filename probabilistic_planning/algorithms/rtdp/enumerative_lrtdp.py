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

def all_states_solved(states, solved_states):
    for state in states:
        if state not in solved_states:
            return False

    return True

def residual(state, mdp, gamma, value_function):
    action = compute_greedy_action(state, mdp, gamma, value_function)
    quality = compute_quality(state, action, mdp, gamma, value_function)

    return abs(value_function[state] - quality)

def reachable_states(state, action, mdp):
    states = []

    for next_state in mdp.states:
        probability = mdp.transition(state, action, next_state)
        if probability > 0:
            states.append(next_state)

    return states

def check_solved(root_state, epsilon, solved_states, mdp, gamma, value_function):
    solved = True
    open_states = []
    closed_states = []
    bellman_backups_done = 0

    if root_state not in solved_states:
        open_states.append(root_state)

    while len(open_states) != 0:
        state = open_states.pop()
        closed_states.append(state)

        if residual(state, mdp, gamma, value_function) > epsilon:
            solved = False
            continue

        action = compute_greedy_action(state, mdp, gamma, value_function)

        for next_state in reachable_states(state, action, mdp):
            is_solved = next_state in solved_states
            is_open = next_state in open_states
            is_closed = next_state in closed_states

            if not (is_solved or is_closed or is_open):
                open_states.append(next_state)

    if solved:
        for closed_state in closed_states:
            solved_states.append(closed_state)
    else:
        while len(closed_states) != 0:
            closed_state = closed_states.pop()

            value_function[closed_state] = compute_bellman_backup(closed_state, mdp, gamma, value_function)
            bellman_backups_done = bellman_backups_done + 1

    return solved, bellman_backups_done

def enumerative_lrtdp(mdp, gamma, max_depth, epsilon, initial_value_function = None, seed = None):
    """Executes the Labeled Real Time Dynamic Programming algorithm.

    Parameters:
    mdp (EnumerativeMDP): enumerative Markov Decison Problem to be solved
    gamma (float): discount factor applied to solve this MDP (assumes infinite on indefinite horizon)
    max_depth (int): max depth to search (used to avoid infinite loops on deadends)
    epsilon (float): maximum residual allowed between V_k and V_{k+1}
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
        value_function = EnumerativeValueFunction(lambda state: 0) # value function with zeroes

    if seed is not None:
        np.random.seed(seed)

    solved_states = []
    bellman_backups_done = 0
    trials = 0

    while not all_states_solved(mdp.initial_states, solved_states):
        trials = trials + 1
        visited_states = []
        initial_state = np.random.choice(mdp.initial_states)

        state = initial_state

        while state not in solved_states:
            visited_states.append(state)

            if state in mdp.goal_states:
                break

            value_function[state] = compute_bellman_backup(state, mdp, gamma, value_function)
            bellman_backups_done = bellman_backups_done + 1

            next_action = compute_greedy_action(state, mdp, gamma, value_function)
            state = sample_next_state(state, next_action, mdp)

            if len(visited_states) > max_depth:
                break

        while len(visited_states) != 0:
            state = visited_states.pop()

            solved, bellman_backups = check_solved(state, epsilon, solved_states, mdp, gamma, value_function)
            bellman_backups_done = bellman_backups_done + bellman_backups

            if not solved:
                break

    # compute policy
    policy = compute_policy(mdp, gamma, value_function)

    statistics = {
        "iterations": trials,
        "bellman_backups_done": bellman_backups_done
    }

    return policy, value_function, statistics