from ..context import probabilistic_planning
from probabilistic_planning.structures import TransitionFunction

import unittest
import numpy as np

class TransitionFunctionTests(unittest.TestCase):

    # Constructor tests

    def test_constructor_call_with_all_parameters_none(self):
        with self.assertRaisesRegex(ValueError, "The states should be defined"):
            TransitionFunction(transition_function=None, actions=None, states=None)

    def test_constructor_call_with_state_set_ommited(self):
        transition_function = {
            "some-action": {
                ( "state01", "state01" ): 1.0
            }
        }
        actions = {"some-action"}

        with self.assertRaisesRegex(ValueError, "The states should be defined"):
            TransitionFunction(transition_function=transition_function, actions=actions, states=None)

    def test_constructor_call_with_transition_function_ommited(self):
        actions = {"some-action"}
        states = {"state01"}

        with self.assertRaisesRegex(ValueError, "The transition function should be defined"):
            TransitionFunction(transition_function=None, actions=actions, states=states)

    def test_constructor_call_with_transition_function_without_actions(self):
        transition_function = {}
        actions = {}
        states = {"state01"}

        with self.assertRaisesRegex(ValueError, "The transition function must have at least one action transition matrix"):
            TransitionFunction(transition_function=transition_function, actions=actions, states=states)

    def test_constructor_call_with_transition_function_having_incorrect_number_of_actions(self):
        transition_function = {
            "some-action": {}
        }
        actions = {"some-action01", "some-action02"}
        states = {"state01"}

        with self.assertRaisesRegex(ValueError, "The transition function must have one transition matrix per action"):
            TransitionFunction(transition_function=transition_function, actions=actions, states=states)

    def test_constructor_call_with_transition_function_having_undefined_action(self):
        transition_function = {
            "some-weird-action": {}
        }
        actions = {"some-action"}
        states = {"state01"}

        with self.assertRaisesRegex(ValueError, "The some-weird-action action is not defined in action list"):
            TransitionFunction(transition_function=transition_function, actions=actions, states=states)

    def test_constructor_call_with_transition_function_having_action_with_no_transitions(self):
        transition_function = {
            "some-action": {}
        }
        actions = {"some-action"}
        states = {"state01"}

        with self.assertRaisesRegex(ValueError, "The action \[some-action\] should have at least one transition defined"):
            TransitionFunction(transition_function=transition_function, actions=actions, states=states)

    def test_constructor_call_with_transition_function_having_action_with_transition_to_an_invalid_state(self):
        transition_function_with_invalid_to_state = {
            "some-action": {
                ("invalid-state01", "state01"): 1.0
            }
        }
        transition_function_with_invalid_from_state = {
            "some-action": {
                ("state01", "invalid-state01"): 1.0
            }
        }
        actions = {"some-action"}
        states = {"state01"}

        with self.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
            TransitionFunction(transition_function=transition_function_with_invalid_to_state, actions=actions, states=states)

        with self.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
            TransitionFunction(transition_function=transition_function_with_invalid_to_state, actions=actions, states=states)

    def test_constructor_call_with_transition_function_having_action_with_invalid_transition_distributions(self):
        transition_function_with_transition_greater_than_one = {
            "some-action": {
                ("state01", "state01"): 0.9,
                ("state01", "state02"): 0.2,
                ("state02", "state02"): 1.0
            }
        }
        transition_function_with_transition_lesser_than_one = {
            "some-action": {
                ("state01", "state01"): 1.0,
                ("state02", "state02"): 0.1,
            }
        }
        actions = {"some-action"}
        states = {"state01","state02"}

        with self.assertRaisesRegex(ValueError, "Invalid probability distribution on \[state01\] transition in action \[some-action\]. The sum of all transitions probabilities from this state to others must be 1 \(one\)"):
            TransitionFunction(transition_function=transition_function_with_transition_greater_than_one, actions=actions, states=states)

        with self.assertRaisesRegex(ValueError, "Invalid probability distribution on \[state02\] transition in action \[some-action\]. The sum of all transitions probabilities from this state to others must be 1 \(one\)"):
            TransitionFunction(transition_function=transition_function_with_transition_lesser_than_one, actions=actions, states=states)

    def test_constructor_call_with_transition_function_having_action_with_invalid_transition_distributions(self):
        transition_function_as_dict = {
            "some-action": {
                ("state01", "state01"): 0.9,
                ("state01", "state02"): 0.1,
                ("state02", "state02"): 1.0
            }
        }
        actions = {"some-action"}
        states = {"state01","state02"}

        TransitionFunction(transition_function=transition_function_as_dict, actions=actions, states=states)

    # get_transition_probability method tests

    def test_get_transition_probability_call_with_invalid_state(self):
        transition_function_as_dict = {
            "some-action": {
                ("state01", "state01"): 0.9,
                ("state01", "state02"): 0.1,
                ("state02", "state02"): 1.0
            }
        }
        actions = {"some-action"}
        states = {"state01","state02"}

        transition_function = TransitionFunction(transition_function=transition_function_as_dict, actions=actions, states=states)

        with self.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
            transition_function.get_transition_probability("invalid-state01", "some-action", "state01")

        with self.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
            transition_function.get_transition_probability("state01", "some-action", "invalid-state01")

    def test_get_transition_probability_call_with_invalid_action(self):
        transition_function_as_dict = {
            "some-action": {
                ("state01", "state01"): 0.9,
                ("state01", "state02"): 0.1,
                ("state02", "state02"): 1.0
            }
        }
        actions = {"some-action"}
        states = {"state01","state02"}

        transition_function = TransitionFunction(transition_function=transition_function_as_dict, actions=actions, states=states)

        with self.assertRaisesRegex(ValueError, "Action \[some\-invalid\-action\] not found"):
            transition_function.get_transition_probability("state01", "some-invalid-action", "state01")

    def test_get_transition_probability_call_with_valid_parameters(self):
        transition_function_as_dict = {
            "some-action": {
                ("state01", "state01"): 0.9,
                ("state01", "state02"): 0.1,
                ("state02", "state02"): 1.0
            }
        }
        actions = {"some-action"}
        states = {"state01","state02"}

        transition_function = TransitionFunction(transition_function=transition_function_as_dict, actions=actions, states=states)

        self.assertEqual(transition_function.get_transition_probability("state01", "some-action", "state01"), 0.9)
        self.assertEqual(transition_function.get_transition_probability("state01", "some-action", "state02"), 0.1)
        self.assertEqual(transition_function.get_transition_probability("state02", "some-action", "state02"), 1.0)

    # get_transition_matrix method tests

    def test_get_transition_matrix_call_with_invalid_action(self):
        transition_function_as_dict = {
            "some-action": {
                ("state01", "state01"): 0.9,
                ("state01", "state02"): 0.1,
                ("state02", "state02"): 1.0
            }
        }
        actions = {"some-action"}
        states = {"state01","state02"}

        transition_function = TransitionFunction(transition_function=transition_function_as_dict, actions=actions, states=states)

        with self.assertRaisesRegex(ValueError, "Action \[some\-invalid\-action\] not found"):
            transition_function.get_transition_matrix("some-invalid-action")

    def test_get_transition_matrix_call_with_valid_parameters(self):
        transition_function_as_dict = {
            "some-action": {
                ("state01", "state01"): 0.9,
                ("state01", "state02"): 0.1,
                ("state02", "state02"): 1.0
            }
        }
        actions = {"some-action"}
        states = {"state01","state02"}

        transition_function = TransitionFunction(transition_function=transition_function_as_dict, actions=actions, states=states)

        expected_matrix = np.matrix([[0.9, 0.1], [0.0, 1.0]])
        obtained_matrix = transition_function.get_transition_matrix("some-action")

        self.assertListEqual(list(obtained_matrix.flat), list(expected_matrix.flat))
