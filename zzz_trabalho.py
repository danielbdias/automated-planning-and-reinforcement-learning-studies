import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from probabilistic_planning.problems.reader import read_problem_file
from probabilistic_planning.algorithms.value_iteration import enumerative_value_iteration
from probabilistic_planning.algorithms.rtdp import enumerative_rtdp, enumerative_lrtdp

epsilon = 1e-8
gamma = 0.9

def test_algorithm(algorithm_name, mdp, algorithm):
    start_time = time.time()

    policy, value_function, statistics = algorithm(mdp)

    elapsed_time = time.time() - start_time
    elapsed_time_as_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    return {
        "algorithm_name": algorithm_name,
        "elapsed_time": elapsed_time,
        "elapsed_time_as_string": elapsed_time_as_string,
        "iterations": statistics["iterations"],
        "bellman_backups_done": statistics["bellman_backups_done"],
        "maximum_residuals": statistics["maximum_residuals"],
        "last_residual": float(statistics["maximum_residuals"][-1])
    }

mdp = read_problem_file(problem_file="probabilistic_planning/problems/files/enumerative/river_traversal_02.net")

result = test_algorithm("IV", mdp, lambda mdp: enumerative_value_iteration(mdp, gamma=0.9, epsilon=1e-8))
print(result["algorithm_name"])
print(result["iterations"])
print(result["elapsed_time"])

result = test_algorithm("LRTDP", mdp, lambda mdp: enumerative_lrtdp(mdp, gamma=0.9, epsilon=1e-8, max_depth=20_000))
print(result["algorithm_name"])
print(result["iterations"])
print(result["elapsed_time"])

result = test_algorithm("RTDP", mdp, lambda mdp: enumerative_rtdp(mdp, gamma=0.9, max_trials=4_500, max_depth=20_000))
print(result["algorithm_name"])
print(result["iterations"])
print(result["elapsed_time"])

mdp = read_problem_file(problem_file="probabilistic_planning/problems/files/enumerative/river_traversal_03.net")

result = test_algorithm("IV", mdp, lambda mdp: enumerative_value_iteration(mdp, gamma=0.9, epsilon=1e-8))
print(result["algorithm_name"])
print(result["iterations"])
print(result["elapsed_time"])

result = test_algorithm("LRTDP", mdp, lambda mdp: enumerative_lrtdp(mdp, gamma=0.9, epsilon=1e-8, max_depth=200_000))
print(result["algorithm_name"])
print(result["iterations"])
print(result["elapsed_time"])

result = test_algorithm("RTDP", mdp, lambda mdp: enumerative_rtdp(mdp, gamma=0.9, max_trials=10_000, max_depth=200_000))
print(result["algorithm_name"])
print(result["iterations"])
print(result["elapsed_time"])
