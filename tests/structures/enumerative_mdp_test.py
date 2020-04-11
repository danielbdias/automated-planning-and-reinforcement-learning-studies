from ..context import probabilistic_planning
from probabilistic_planning.structures import EnumerativeMDP

import unittest

from nose2.tools import such

with such.A("Enumerative MDP class representation") as it:
    with it.having("constructor called"):
        @it.should("raise error when all provided parameters are None")
        def test_all_constructor_parameters_are_none(self):
            with it.assertRaisesRegex(ValueError, "The states should be defined"):
                EnumerativeMDP(states=None, reward_function=None, transition_function=None, initial_states=None, goal_states=None)

        @it.should("raise error when only state list is None")
        def test_states_is_none(self):
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            with it.assertRaisesRegex(ValueError, "The states should be defined"):
                EnumerativeMDP(states=None, reward_function=reward_function, transition_function=transition_function,
                               initial_states=initial_states, goal_states=goal_states)

        @it.should("raise error when state list is empty")
        def test_states_is_empty(self):
            states = []
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            with it.assertRaisesRegex(ValueError, "The states should have at least one state"):
                EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                               initial_states=initial_states, goal_states=goal_states)

        @it.should("raise error when state list has duplicated states")
        def test_states_has_duplicated_states(self):
            states = [ "state01", "state01" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            with it.assertRaisesRegex(ValueError, "There is a repeated state identifier in the states"):
                EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                               initial_states=initial_states, goal_states=goal_states)

        @it.should("raise error when reward function is None")
        def test_reward_function_is_none(self):
            states = ["state01", "state02"]
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            with it.assertRaisesRegex(ValueError, "The reward function should be defined"):
                EnumerativeMDP(states=states, reward_function=None, transition_function=transition_function,
                               initial_states=initial_states, goal_states=goal_states)

        @it.should("raise error when reward function does not define a reward for each state")
        def test_reward_function_not_defines_all_states(self):
            states = ["state01", "state02"]
            reward_function = {
                "state01": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            with it.assertRaisesRegex(ValueError, "The reward function must be have a value for each state"):
                EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                               initial_states=initial_states, goal_states=goal_states)

        @it.should("raise error when reward function defines a reward to an invalid state")
        def test_reward_function_not_defines_all_states(self):
            states = ["state01", "state02"]
            reward_function = {
                "state01": 1,
                "invalid-state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            with it.assertRaisesRegex(ValueError, "Invalid state \[invalid\-state02\] defined in reward function"):
                EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                               initial_states=initial_states, goal_states=goal_states)

        @it.should("raise error when transition function is None")
        def test_transition_function_is_none(self):
            states = ["state01", "state02"]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            with it.assertRaisesRegex(ValueError, "The transition function should be defined"):
                EnumerativeMDP(states=states, reward_function=reward_function, transition_function=None,
                               initial_states=initial_states, goal_states=goal_states)

        # other transition function validations are done inside TransitionFunction class and tested there

        @it.should("raise error when initial state list has duplicated states")
        def test_initial_states_has_duplicated_states(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01", "state01"]
            goal_states = ["state02"]

            with it.assertRaisesRegex(ValueError, "There is a repeated state identifier in the initial states"):
                EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                               initial_states=initial_states, goal_states=goal_states)

        @it.should("raise error when initial state list has invalid states")
        def test_initial_states_has_invalid_states(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01", "invalid-state01"]
            goal_states = ["state02"]

            with it.assertRaisesRegex(ValueError, "Unrecognized state \[invalid\-state01\] in initial states"):
                EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                               initial_states=initial_states, goal_states=goal_states)

        @it.should("raise error when goal state list has duplicated states")
        def test_goal_states_has_duplicated_states(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02", "state02"]

            with it.assertRaisesRegex(ValueError, "There is a repeated state identifier in the goal states"):
                EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                               initial_states=initial_states, goal_states=goal_states)

        @it.should("raise error when goal state list has invalid states")
        def test_goal_states_has_invalid_states(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02","invalid-state02"]

            with it.assertRaisesRegex(ValueError, "Unrecognized state \[invalid\-state02\] in goal states"):
                EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                               initial_states=initial_states, goal_states=goal_states)

        @it.should("return successfully when all parameters are passed")
        def test_all_constructor_parameters_are_informed(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                           initial_states=initial_states, goal_states=goal_states)

        @it.should("return successfully when all parameters are passed, except initial and goal state list")
        def test_all_constructor_parameters_are_informed_except_initial_and_goal_states(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }

            EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function)

    with it.having("states attribute called"):
        @it.should("return a valid set")
        def test_state_attribute(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            mdp = EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                                 initial_states=initial_states, goal_states=goal_states)

            it.assertSetEqual(mdp.states, { "state01", "state02" })

    with it.having("reward_function attribute called"):
        @it.should("return a valid dict")
        def test_reward_function_attribute(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            mdp = EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                                 initial_states=initial_states, goal_states=goal_states)

            it.assertDictEqual(mdp.reward_function, { "state01": 1, "state02": 1 })

    with it.having("transition_function attribute called"):
        @it.should("return a valid object")
        def test_transition_function_attribute(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            mdp = EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                                 initial_states=initial_states, goal_states=goal_states)

            it.assertListEqual(
                list(mdp.transition_function.get_transition_matrix("some-action").flat),
                [ 0.9, 0.1, 0.0, 1.0 ]
            )

    with it.having("initial_states attribute called"):
        @it.should("return a valid set")
        def test_initial_states_attribute(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            mdp = EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                                 initial_states=initial_states, goal_states=goal_states)

            it.assertSetEqual(mdp.initial_states, { "state01" })

    with it.having("goal_states attribute called"):
        @it.should("return a valid set")
        def test_goal_states_attribute(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            mdp = EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                                 initial_states=initial_states, goal_states=goal_states)

            it.assertSetEqual(mdp.goal_states, { "state02" })

    with it.having("reward method called"):
        @it.should("call the internal reward function")
        def test_reward_method_attribute(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            mdp = EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                                 initial_states=initial_states, goal_states=goal_states)

            it.assertEqual(mdp.reward("state01"), 1)
            it.assertEqual(mdp.reward("state02"), 1)

    with it.having("transition method called"):
        @it.should("call the internal transition function")
        def test_transition_method_attribute(self):
            states = [ "state01", "state02" ]
            reward_function = {
                "state01": 1,
                "state02": 1
            }
            transition_function = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            initial_states = ["state01"]
            goal_states = ["state02"]

            mdp = EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                                 initial_states=initial_states, goal_states=goal_states)

            it.assertEqual(mdp.transition("state01", "some-action", "state01"), 0.9)
            it.assertEqual(mdp.transition("state01", "some-action", "state02"), 0.1)
            it.assertEqual(mdp.transition("state02", "some-action", "state02"), 1.0)

    it.createTests(globals())
