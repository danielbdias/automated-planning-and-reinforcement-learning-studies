from ..context import probabilistic_planning
from probabilistic_planning.problems.reader import read_problem_file

import unittest
from unittest import mock

from nose2.tools import such

def mock_file_content(lines):
    file_content = "\n".join(lines)
    mock_open = mock.mock_open(read_data=file_content)
    return mock.patch("probabilistic_planning.problems.reader.open", mock_open)


with such.A("Problem Reader function") as it:

    @it.should("fail when the problem file does not define any sections")
    def test_file_with_any_section(self):
        file_content = [
            ""
        ]

        with mock_file_content(file_content), \
             it.assertRaisesRegex(ValueError, "The states should be defined"):
            read_problem_file("some_file.txt")

    @it.should("fail when the problem file does not define reward and action sections")
    def test_file_without_reward_and_action_sections(self):
        file_content = [
            "states",
            "   state01",
            "endstates"
        ]

        with mock_file_content(file_content), \
             it.assertRaisesRegex(ValueError, "The reward function should be defined"):
            read_problem_file("some_file.txt")

    @it.should("fail when the problem file does not define an action section")
    def test_file_without_action_section(self):
        file_content = [
            "states",
            "   state01",
            "endstates",
            "reward",
            "\tstate01 1.0",
            "endreward"
        ]

        with mock_file_content(file_content), \
             it.assertRaisesRegex(ValueError, "The transition function must have at least one defined action"):
            read_problem_file("some_file.txt")

    @it.should("succeed when the problem file has all required sections")
    def test_file_with_required_sections(self):
        file_content = [
            "states",
            "   state01",
            "endstates",
            "reward",
            "\tstate01 1.0",
            "endreward",
            "action first-action",
            "\tstate01 state01 1.0",
            "endaction"
        ]

        with mock_file_content(file_content):
            mdp = read_problem_file("some_file.txt")

            it.assertListEqual(mdp.states, ["state01"])
            it.assertDictEqual(mdp.reward_function, { "state01": 1.0 })

            it.assertIn("first-action", mdp.transition_function.keys())
            it.assertDictEqual(mdp.transition_function["first-action"], { "state01": { "state01": 1.0 } })

    it.createTests(globals())