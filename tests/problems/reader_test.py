from probabilistic_planning.problems.reader import read_problem_file

import unittest


class ProblemReaderTestSuite(unittest.TestCase):
    """Test cases for MDP problem reader."""

    def test_thoughts(self):
        read_problem_file('bla.txt')


if __name__ == '__main__':
    unittest.main()