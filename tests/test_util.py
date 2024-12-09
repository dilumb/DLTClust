""" Unit tests for Util module """

import unittest
from pathlib import Path
from src.dsm import DSM
import src.util as util

MOD_PATH = str(Path(__file__).parent)
INPUT_FILE_NAME = MOD_PATH + "/test_dsm.csv"
OUTPUT_FILE_NAME = MOD_PATH + "/test_dsm_clusters.csv"
ALPHA = 0.8116
BETA = 0.1102
P_C = 0.5
P_M = 0.25
POPULATION_SIZE = 10
OFFSPRING_SIZE = 10
GENERATION_LIMIT = 10
GENERATION_LIMIT_WITHOUT_IMPROVEMENT = 2


class TestUtil(unittest.TestCase):
    """ Unit tests for Util module """

    def test_str_2_bool(self):
        self.assertEqual(util.str_2_bool('true'), True)
        self.assertEqual(util.str_2_bool('True'), True)
        self.assertEqual(util.str_2_bool('tRUE'), True)
        self.assertEqual(util.str_2_bool('false'), False)
        self.assertEqual(util.str_2_bool('False'), False)
        self.assertEqual(util.str_2_bool('fALSe'), False)
        with self.assertRaises(ValueError):
            util.str_2_bool('true1')
        with self.assertRaises(ValueError):
            util.str_2_bool('1false')
        with self.assertRaises(ValueError):
            util.str_2_bool('fa1se')

    def test_dsm_to_graph(self):
        num_square_clusters = 4
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)
        util.dsm_to_graph(dsm, './dsm_to_graph.txt')

    def test_build_chromosome(self):
        names = ['A', 'B', 'C', 'D', 'E', 'F']
        clusters = [['C', 'B', 'A'], ['D', 'E'], ['F', 'E']]
        num_clusters = 4
        c = util.build_chromosome(clusters, names, num_clusters)
        self.assertEqual(len(c), 24)
        expected_c = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1,
                      1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        self.assertEqual(c, expected_c)
