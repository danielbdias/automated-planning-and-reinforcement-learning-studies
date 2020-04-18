from ..context import probabilistic_planning
from probabilistic_planning.structures import EnumerativeMDP

import unittest

class EnumerativeMDPTests(unittest.TestCase):

    # Constructor tests
    def test_constructor_call_with_all_parameters_none(self):
        with self.assertRaisesRegex(ValueError, "The states should be defined"):
            EnumerativeMDP(states=None, reward_function=None, transition_function=None, initial_states=None, goal_states=None)

    def test_constructor_call_with_states_none(self):
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

        with self.assertRaisesRegex(ValueError, "The states should be defined"):
            EnumerativeMDP(states=None, reward_function=reward_function, transition_function=transition_function,
                            initial_states=initial_states, goal_states=goal_states)

    def test_constructor_call_with_states_empty(self):
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

        with self.assertRaisesRegex(ValueError, "The states should have at least one state"):
            EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                            initial_states=initial_states, goal_states=goal_states)

    def test_constructor_call_with_states_having_duplicated_states(self):
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

        with self.assertRaisesRegex(ValueError, "There is a repeated state identifier in the states"):
            EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                            initial_states=initial_states, goal_states=goal_states)

    def test_constructor_call_with_reward_function_none(self):
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

        with self.assertRaisesRegex(ValueError, "The reward function should be defined"):
            EnumerativeMDP(states=states, reward_function=None, transition_function=transition_function,
                            initial_states=initial_states, goal_states=goal_states)

    def test_constructor_call_with_reward_function_having_not_defined_all_states(self):
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

        with self.assertRaisesRegex(ValueError, "The reward function must be have a value for each state"):
            EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                            initial_states=initial_states, goal_states=goal_states)

    def test_constructor_call_with_reward_function_having_invalid_state(self):
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

        with self.assertRaisesRegex(ValueError, "Invalid state \[invalid\-state02\] defined in reward function"):
            EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                            initial_states=initial_states, goal_states=goal_states)

    def test_constructor_call_with_transition_function_none(self):
        states = ["state01", "state02"]
        reward_function = {
            "state01": 1,
            "state02": 1
        }
        initial_states = ["state01"]
        goal_states = ["state02"]

        with self.assertRaisesRegex(ValueError, "The transition function should be defined"):
            EnumerativeMDP(states=states, reward_function=reward_function, transition_function=None,
                            initial_states=initial_states, goal_states=goal_states)

    def test_constructor_call_with_initial_states_having_duplicated_states(self):
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

        with self.assertRaisesRegex(ValueError, "There is a repeated state identifier in the initial states"):
            EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                            initial_states=initial_states, goal_states=goal_states)

    def test_constructor_call_with_initial_states_having_invalid_states(self):
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

        with self.assertRaisesRegex(ValueError, "Unrecognized state \[invalid\-state01\] in initial states"):
            EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                            initial_states=initial_states, goal_states=goal_states)

    def test_constructor_call_with_goal_states_having_duplicated_states(self):
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

        with self.assertRaisesRegex(ValueError, "There is a repeated state identifier in the goal states"):
            EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                            initial_states=initial_states, goal_states=goal_states)

    def test_constructor_call_with_goal_states_having_invalid_states(self):
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

        with self.assertRaisesRegex(ValueError, "Unrecognized state \[invalid\-state02\] in goal states"):
            EnumerativeMDP(states=states, reward_function=reward_function, transition_function=transition_function,
                            initial_states=initial_states, goal_states=goal_states)

    def test_constructor_call_with_all_valid_parameters(self):
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

    def test_constructor_call_with_all_valid_parameters_except_initial_and_goal_states(self):
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

    # Attributes tests
    def test_state_attribute_call(self):
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

        self.assertSetEqual(mdp.states, { "state01", "state02" })

    def test_reward_function_attribute_call(self):
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

        self.assertDictEqual(mdp.reward_function, { "state01": 1, "state02": 1 })

    def test_transition_function_attribute_call(self):
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

        self.assertListEqual(
            list(mdp.transition_function.get_transition_matrix("some-action").flat),
            [ 0.9, 0.1, 0.0, 1.0 ]
        )

    def test_initial_states_attribute_call(self):
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

        self.assertSetEqual(mdp.initial_states, { "state01" })

    def test_goal_states_attribute_call(self):
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

        self.assertSetEqual(mdp.goal_states, { "state02" })

    # Method tests
    def test_reward_call_with_valid_parameters(self):
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

        self.assertEqual(mdp.reward("state01"), 1)
        self.assertEqual(mdp.reward("state02"), 1)

    def test_transition_call_with_valid_parameters(self):
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

        self.assertEqual(mdp.transition("state01", "some-action", "state01"), 0.9)
        self.assertEqual(mdp.transition("state01", "some-action", "state02"), 0.1)
        self.assertEqual(mdp.transition("state02", "some-action", "state02"), 1.0)
