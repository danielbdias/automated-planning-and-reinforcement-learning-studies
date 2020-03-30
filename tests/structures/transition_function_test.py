from ..context import probabilistic_planning
from probabilistic_planning.structures import TransitionFunction

import unittest
import numpy as np

from nose2.tools import such

with such.A("Transition function representation") as it:
    with it.having("constructor called"):
        @it.should("raise error when all parameters are None")
        def test_all_constructor_parameters_are_none(self):
            with it.assertRaisesRegex(ValueError, "The states should be defined"):
                TransitionFunction(transition_function=None, states=None)

        @it.should("raise error when only a transition function is informed")
        def test_only_transition_function_is_informed(self):
            transition_function = {
                "some-action": {
                    ( "state01", "state01" ): 1.0
                }
            }

            with it.assertRaisesRegex(ValueError, "The states should be defined"):
                TransitionFunction(transition_function=transition_function, states=None)

        @it.should("raise error when only a state set is informed")
        def test_only_state_set_is_informed(self):
            states = {"state01"}

            with it.assertRaisesRegex(ValueError, "The transition function should be defined"):
                TransitionFunction(transition_function=None, states=states)

        @it.should("raise error when the transition function have no actions")
        def test_transition_function_without_actions(self):
            transition_function = {}
            states = {"state01"}

            with it.assertRaisesRegex(ValueError, "The transition function must have at least one action transition matrix"):
                TransitionFunction(transition_function=transition_function, states=states)

        @it.should("raise error when the transition function has an action with no transitions")
        def test_transition_function_has_action_with_no_transitions(self):
            transition_function = {
                "some-action": {}
            }
            states = {"state01"}

            with it.assertRaisesRegex(ValueError, "The action \[some-action\] should have at least one transition defined"):
                TransitionFunction(transition_function=transition_function, states=states)

        @it.should("raise error when the transition function has an action with transitions to invalid states")
        def test_transition_function_has_action_with_transition_to_an_invalid_state(self):
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
            states = {"state01"}

            with it.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
                TransitionFunction(transition_function=transition_function_with_invalid_to_state, states=states)

            with it.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
                TransitionFunction(transition_function=transition_function_with_invalid_to_state, states=states)

        @it.should("raise error when the transition function has an action with invalid transition distributions")
        def test_transition_function_has_action_with_invalid_transition_distributions(self):
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
            states = {"state01","state02"}

            with it.assertRaisesRegex(ValueError, "Invalid probability distribution on \[state01\] transition in action \[some-action\]. The sum of all transitions probabilities from this state to others must be 1 \(one\)"):
                TransitionFunction(transition_function=transition_function_with_transition_greater_than_one, states=states)

            with it.assertRaisesRegex(ValueError, "Invalid probability distribution on \[state02\] transition in action \[some-action\]. The sum of all transitions probabilities from this state to others must be 1 \(one\)"):
                TransitionFunction(transition_function=transition_function_with_transition_lesser_than_one, states=states)

        @it.should("return successfully when the transition function is valid")
        def test_transition_function_has_action_with_invalid_transition_distributions(self):
            transition_function_as_dict = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            states = {"state01","state02"}

            TransitionFunction(transition_function=transition_function_as_dict, states=states)

    with it.having("get_transition_probability method called"):
        @it.should("raise error when an invalid state is informed")
        def test_invalid_state_informed_to_get_transition_probability(self):
            transition_function_as_dict = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            states = {"state01","state02"}

            transition_function = TransitionFunction(transition_function=transition_function_as_dict, states=states)

            with it.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
                transition_function.get_transition_probability("invalid-state01", "some-action", "state01")

            with it.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
                transition_function.get_transition_probability("state01", "some-action", "invalid-state01")

        @it.should("raise error when an invalid action is informed")
        def test_invalid_action_informed_to_get_transition_probability(self):
            transition_function_as_dict = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            states = {"state01","state02"}

            transition_function = TransitionFunction(transition_function=transition_function_as_dict, states=states)

            with it.assertRaisesRegex(ValueError, "Action \[some\-invalid\-action\] not found"):
                transition_function.get_transition_probability("state01", "some-invalid-action", "state01")

        @it.should("return correct probabilities when valid states and actions are informed")
        def test_valid_parameters_informed_to_get_transition_probability(self):
            transition_function_as_dict = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            states = {"state01","state02"}

            transition_function = TransitionFunction(transition_function=transition_function_as_dict, states=states)

            it.assertEqual(transition_function.get_transition_probability("state01", "some-action", "state01"), 0.9)
            it.assertEqual(transition_function.get_transition_probability("state01", "some-action", "state02"), 0.1)
            it.assertEqual(transition_function.get_transition_probability("state02", "some-action", "state02"), 1.0)

    with it.having("get_transition_matrix method called"):
        @it.should("raise error when an invalid action is informed")
        def test_invalid_action_informed_to_get_transition_matrix(self):
            transition_function_as_dict = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            states = {"state01","state02"}

            transition_function = TransitionFunction(transition_function=transition_function_as_dict, states=states)

            with it.assertRaisesRegex(ValueError, "Action \[some\-invalid\-action\] not found"):
                transition_function.get_transition_matrix("some-invalid-action")

        @it.should("return transition matrix when a valid action is informed")
        def test_valid_parameters_informed_to_get_transition_probability(self):
            transition_function_as_dict = {
                "some-action": {
                    ("state01", "state01"): 0.9,
                    ("state01", "state02"): 0.1,
                    ("state02", "state02"): 1.0
                }
            }
            states = {"state01","state02"}

            transition_function = TransitionFunction(transition_function=transition_function_as_dict, states=states)

            expected_matrix = np.matrix([[0.9, 0.1], [0.0, 1.0]])
            obtained_matrix = transition_function.get_transition_matrix("some-action")

            it.assertListEqual(list(obtained_matrix.flat), list(expected_matrix.flat))

    it.createTests(globals())
