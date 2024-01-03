""" Unit tests for DMM class """
import unittest
from pathlib import Path
from src.dsm import DSM
from src.datatypes import ClusterType

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


class TestDSM(unittest.TestCase):
    """ Unit tests for DMM class """

    def test_init(self):
        """ Test init method """
        num_square_clusters = 4
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        n = 5
        self.assertEqual(dsm.n, n)
        self.assertEqual(dsm.output_file_name, OUTPUT_FILE_NAME)
        self.assertEqual(dsm.num_clusters, num_square_clusters + num_busses)
        self.assertEqual(dsm.num_busses, num_busses)
        self.assertEqual(dsm.num_sinks, num_sinks)
        self.assertEqual(dsm.num_sources, num_sources)
        self.assertEqual(dsm.num_square_clusters, num_square_clusters)
        self.assertEqual(dsm.population_size, POPULATION_SIZE)
        self.assertEqual(dsm.offspring_size, OFFSPRING_SIZE)
        self.assertEqual(dsm.p_c, P_C)
        self.assertEqual(dsm.p_m, P_M)
        self.assertEqual(dsm.generation_limit, GENERATION_LIMIT)
        self.assertEqual(dsm.generation_limit_without_improvement,
                         GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        self.assertEqual(dsm.mdl.n_n, n)
        self.assertEqual(dsm.mdl.n_c, num_square_clusters + num_busses)
        self.assertEqual(dsm.mdl.alpha, ALPHA)
        self.assertEqual(dsm.mdl.beta, BETA)
        self.assertEqual(dsm.mdl.gamma, 1 - ALPHA - BETA)

    def test_init_fail(self):
        """ Test failing init method """
        num_square_clusters = -1
        num_busses = -2
        num_sinks = -3
        num_sources = -4

        self.assertRaises(ValueError, DSM, INPUT_FILE_NAME, OUTPUT_FILE_NAME,
                          num_square_clusters, num_busses, num_sinks,
                          num_sources, ALPHA, BETA, POPULATION_SIZE,
                          OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                          GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

    def test_read_DSM(self):
        """ Test read_DSM """
        num_square_clusters = 4
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)
        n = 5
        d = [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [
            0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0]]
        d_dash = [[0] * n] * n
        self.assertEqual(len(dsm.D), n)
        self.assertEqual(dsm.D, d)
        self.assertEqual(len(dsm.D_dash), n)
        self.assertEqual(dsm.D_dash, d_dash)

    def test_all_pairs(self):
        """ Test all_pairs method """
        num_clusters = 5
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        pairs = [[1, 3], [1, 2], [1, 4], [3, 2], [3, 4], [2, 4]]
        self.assertEqual(dsm._DSM__all_pairs([1, 3, 2, 4]), pairs)

    def test_build_D_dash(self):
        """ Test build_D_dash method """
        num_square_clusters = 2
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom1 = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        d_dash1 = [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [
            0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 5)
        self.assertEqual(dsm.D_dash, d_dash1)

        chrom2 = [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
        d_dash2 = [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [
            1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom2), 6)
        self.assertEqual(dsm.D_dash, d_dash2)

        chrom3 = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        d_dash3 = [[0, 1, 0, 1, 1], [1, 0, 1, 1, 1], [
            0, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom3), 4)
        self.assertEqual(dsm.D_dash, d_dash3)

        chrom4 = [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]
        self.assertEqual(dsm._DSM__build_D_dash(chrom4), 8)
        self.assertEqual(dsm.D_dash, d_dash2)

    def test_build_D_dash_read_only(self):
        """ Test build_D_dash method for read_only cluster """
        num_square_clusters = 2
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT, True)
        # Change some values from test DSM
        dsm.D[3][0] = 0
        dsm.D[3][4] = 0
        chrom1 = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        d_dash1 = [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [
            0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 5)
        self.assertEqual(dsm.D_dash, d_dash1)

        dsm.D[3][0] = 1
        dsm.D[3][4] = 1
        dsm.D[0][3] = 0
        dsm.D[0][4] = 0
        d_dash2 = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [
            0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 5)
        self.assertEqual(dsm.D_dash, d_dash2)

        dsm.D[0][4] = 1
        d_dash3 = [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [
            0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 5)
        self.assertEqual(dsm.D_dash, d_dash3)

        dsm.D[0][3] = 1
        dsm.D[0][4] = 0
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 5)
        self.assertEqual(dsm.D_dash, d_dash3)

        dsm.D[3][0] = 0
        dsm.D[3][4] = 0
        dsm.D[0][4] = 1
        dsm.D[3][2] = 1
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 5)
        self.assertEqual(dsm.D_dash, d_dash1)

        dsm.D[3][1] = 1
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 5)
        self.assertEqual(dsm.D_dash, d_dash1)

        dsm.D[3][4] = 1
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 5)
        self.assertEqual(dsm.D_dash, d_dash3)

        dsm.D[3][4] = 0
        dsm.D[3][0] = 1
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 5)
        self.assertEqual(dsm.D_dash, d_dash3)

    def test_build_D_dash_partial_bus(self):
        """ Test build_D_dash method for a partial bus"""
        num_square_clusters = 2
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT, False, True)
        # Change some values from test DSM to make a bus
        dsm.D[0][1] = 1
        dsm.D[3][1] = 1
        dsm.D[4][1] = 1
        dsm.D[1][0] = 1
        dsm.D[1][3] = 1
        dsm.D[1][4] = 1

        chrom1 = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0]
        d_dash1 = [[0, 1, 0, 1, 1], [1, 0, 1, 1, 1], [
            0, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 6)
        self.assertEqual(dsm.D_dash, d_dash1)

        dsm.D[1][0] = 0
        d_dash2 = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [
            0, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 6)
        self.assertEqual(dsm.D_dash, d_dash2)

        dsm.D[4][1] = 0
        d_dash3 = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0], [
            0, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 0, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 6)
        self.assertEqual(dsm.D_dash, d_dash3)

    def test_build_D_dash_partial_sinks(self):
        """ Test build_D_dash method for a partial sources and sinks"""
        num_square_clusters = 2
        num_busses = 0
        num_sinks = 1
        num_sources = 1

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT, False, False, True, True)
        # Change some values from test DSM
        dsm.D[0][2] = 1
        dsm.D[3][2] = 1
        dsm.D[4][2] = 1

        chrom1 = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        d_dash1 = [[0, 0, 1, 1, 1], [0, 0, 1, 0, 0], [
            0, 1, 0, 0, 0], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 6)
        self.assertEqual(dsm.D_dash, d_dash1)

        dsm.D[0][2] = 0  # partial reader
        d_dash2 = [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [
            0, 1, 0, 0, 0], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom1), 6)
        self.assertEqual(dsm.D_dash, d_dash2)

        dsm.D[0][1] = 1
        dsm.D[4][1] = 1
        chrom2 = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        d_dash3 = [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [
            0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom2), 7)
        self.assertEqual(dsm.D_dash, d_dash3)

        dsm.D[0][2] = 1
        dsm.D[1][2] = 0
        chrom3 = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        d_dash4 = [[0, 1, 1, 1, 1], [0, 0, 0, 0, 0], [
            0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom3), 5)
        self.assertEqual(dsm.D_dash, d_dash4)

        dsm.D[1][2] = 1
        dsm.D[3][2] = 0
        dsm.D[4][2] = 0
        dsm.D[1][0] = 1
        dsm.D[1][3] = 1
        dsm.D[1][4] = 1

        chrom4 = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        d_dash5 = [[0, 0, 0, 1, 1], [1, 0, 1, 1, 1], [
            0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom4), 6)
        self.assertEqual(dsm.D_dash, d_dash5)

        dsm.D[1][3] = 0  # Partial writer
        d_dash6 = [[0, 0, 0, 1, 1], [1, 0, 1, 0, 1], [
            0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom4), 6)
        self.assertEqual(dsm.D_dash, d_dash6)

        dsm.D[0][2] = 0
        dsm.D[1][3] = 1
        chrom5 = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        d_dash7 = [[0, 1, 0, 1, 1], [1, 0, 0, 1, 1], [
            0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom5), 5)
        self.assertEqual(dsm.D_dash, d_dash7)

        dsm.D[0][1] = 0
        dsm.D[0][2] = 1
        d_dash8 = [[0, 0, 1, 1, 1], [1, 0, 1, 1, 1], [
            0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom5), 5)
        self.assertEqual(dsm.D_dash, d_dash8)

    def test_calculate_MDL_value(self):
        """ Test calculate_MDL_value method """
        num_square_clusters = 2
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        self.assertAlmostEqual(
            dsm._DSM__calculate_MDL_value(chrom), 1.452598216)

        chrom = [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
        self.assertAlmostEqual(
            dsm._DSM__calculate_MDL_value(chrom), 56.60081719)

        chrom = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        self.assertAlmostEqual(
            dsm._DSM__calculate_MDL_value(chrom), 28.754345540)

        chrom = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.assertAlmostEqual(
            dsm._DSM__calculate_MDL_value(chrom), 20.837144077)

    def test_fitness(self):
        """ Test fitness method """
        num_square_clusters = 2
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom1 = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        chrom2 = [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
        chrom3 = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        chrom4 = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        chrom = [[chrom1, 0.0], [chrom2, 0.0], [chrom3, 0.0], [chrom4, 0.0]]

        result = [[chrom1, 1.452598216], [chrom2, 56.60081719],
                  [chrom3, 28.754345540], [chrom4, 20.837144077]]
        res = dsm._DSM__fitness(chrom)
        self.assertEqual(len(res), len(result))
        self.assertEqual(res[0][0], result[0][0])
        self.assertEqual(res[1][0], result[1][0])
        self.assertEqual(res[2][0], result[2][0])
        self.assertEqual(res[3][0], result[3][0])
        self.assertAlmostEqual(res[0][1], result[0][1])
        self.assertAlmostEqual(res[1][1], result[1][1])
        self.assertAlmostEqual(res[2][1], result[2][1])
        self.assertAlmostEqual(res[3][1], result[3][1])

    def test_no_busses(self):
        """ Test no_busses method """
        num_square_clusters = 3
        num_busses = 0
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
        d_dash = [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [
            0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0]]
        self.assertEqual(dsm._DSM__build_D_dash(chrom), 5)
        self.assertEqual(dsm.D_dash, d_dash)

        self.assertAlmostEqual(
            dsm._DSM__calculate_MDL_value(chrom), 1.452598216)

        chrom = [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
        self.assertEqual(dsm.D_dash, d_dash)

        self.assertAlmostEqual(
            dsm._DSM__calculate_MDL_value(chrom), 2.6965041203)

    def test_save_clustered_matrix(self):
        """ Test save_clustered_matrix method """
        num_square_clusters = 3
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        dsm.clusters = [['B', 'E', 'F', 'G', 'I'], [
            'C', 'D', 'I'], ['A', 'F', 'H'], ['G']]
        dsm.D_out_header = [i for cluster in dsm.clusters for i in cluster]
        dsm.D_out = [['.', 'x', 'x', 'x', 'x', '', '', '', '', '', '', 'x'],
                     ['x', '.', 'x', 'x', 'x', '', '', '', '', '', '', 'x'],
                     ['x', 'x', '.', 'x', 'x', '', '', '', '', '', '', 'x'],
                     ['x', 'x', 'x', '.', 'x', '', '', '', '', '', '', 'x'],
                     ['x', 'x', 'x', 'x', '.', '', '', '', '', '', '', 'x'],
                     ['', '', '', '', '', '.', 'x', 'x', '', '', '', 'x'],
                     ['', '', '', '', '', 'x', '.', 'x', '', '', '', 'x'],
                     ['', '', '', '', '', 'x', 'x', '.', '', '', '', 'x'],
                     ['', '', '', '', '', '', '', '', '.', 'x', 'x', 'x'],
                     ['', '', '', '', '', '', '', '', 'x', '.', 'x', 'x'],
                     ['', '', '', '', '', '', '', '', 'x', 'x', '.', 'x'],
                     ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', '.']]
        dsm._DSM__save_clustered_matrix(1.567)

    def test_add_cluster_to_matrix(self):
        """ Test add_cluster_to_matrix method """
        num_square_clusters = 2
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        clusters = [(ClusterType.SQUARE, ['C', 'A', 'B']),
                    (ClusterType.SQUARE, ['D', 'E']), (ClusterType.BUS, [])]
        dsm.D_out = [['' for i in range(5)] for j in range(5)]

        dsm._DSM__add_cluster_to_matrix(chrom[0:5], ClusterType.SQUARE)
        self.assertEqual(dsm.D_out_header, clusters[0][1])
        self.assertEqual(dsm.clusters, [clusters[0]])
        self.assertEqual(dsm.D_out_start_idx, 3)

        dsm._DSM__add_cluster_to_matrix(chrom[5:10], ClusterType.SQUARE)
        self.assertEqual(dsm.D_out_header, clusters[0][1] + clusters[1][1])
        self.assertEqual(dsm.clusters, clusters[:2])
        self.assertEqual(dsm.D_out_start_idx, 5)

        dsm._DSM__add_cluster_to_matrix(chrom[10:15], ClusterType.BUS)
        self.assertEqual(
            dsm.D_out_header, clusters[0][1] + clusters[1][1] + clusters[2][1])
        self.assertEqual(dsm.clusters, clusters)
        self.assertEqual(dsm.D_out_start_idx, 5)

        D_out = [['.', 'x', 'x', '', ''],
                 ['x', '.', 'x', '', ''],
                 ['x', 'x', '.', '', ''],
                 ['', '', '', '.', 'x'],
                 ['', '', '', 'x', '.']]
        self.assertEqual(dsm.D_out, D_out)

        dsm.D_out_start_idx = 0
        dsm.D_out_header = []
        dsm.clusters = []
        chrom = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
        clusters = [(ClusterType.SQUARE, ['C', 'A']),
                    (ClusterType.SQUARE, ['D', 'B']), (ClusterType.BUS, ['E'])]
        dsm.D_out = [['' for i in range(5)] for j in range(5)]

        dsm._DSM__add_cluster_to_matrix(chrom[0:5], ClusterType.SQUARE)
        self.assertEqual(dsm.D_out_header, clusters[0][1])
        self.assertEqual(dsm.clusters, [clusters[0]])
        self.assertEqual(dsm.D_out_start_idx, 2)

        dsm._DSM__add_cluster_to_matrix(chrom[5:10], ClusterType.SQUARE)
        self.assertEqual(dsm.D_out_header, clusters[0][1] + clusters[1][1])
        self.assertEqual(dsm.clusters, clusters[:2])
        self.assertEqual(dsm.D_out_start_idx, 4)

        dsm._DSM__add_cluster_to_matrix(chrom[10:15], ClusterType.BUS)
        self.assertEqual(
            dsm.D_out_header, clusters[0][1] + clusters[1][1] + clusters[2][1])
        self.assertEqual(dsm.clusters, clusters)
        self.assertEqual(dsm.D_out_start_idx, 5)

        D_out = [['.', 'x', '', '', 'x'],
                 ['x', '.', '', '', 'x'],
                 ['', '', '.', 'x', 'x'],
                 ['', '', 'x', '.', 'x'],
                 ['x', 'x', 'x', 'x', '.']]
        self.assertEqual(dsm.D_out, D_out)

    def test_build_clustered_DSM(self):
        """ Test build_clustered_DSM method """
        num_square_clusters = 3
        num_busses = 0
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        dsm._DSM__build_clustered_DSM(chrom)
        D_out = [['.', 'x', 'x', '', ''],
                 ['x', '.', 'x', '', ''],
                 ['x', 'x', '.', '', ''],
                 ['', '', '', '.', 'x'],
                 ['', '', '', 'x', '.']]
        self.assertEqual(dsm.D_out, D_out)

        dsm.D_out_start_idx = 0
        dsm.D_out_header = []
        dsm.clusters = []
        chrom = [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
        dsm._DSM__build_clustered_DSM(chrom)
        D_out = [['.', 'x', 'x', '', '', ''],
                 ['x', '.', 'x', '', '', ''],
                 ['x', 'x', '.', '', '', ''],
                 ['', '', '', '.', 'x', 'x'],
                 ['', '', '', 'x', '.', 'x'],
                 ['', '', '', 'x', 'x', '.']]
        self.assertEqual(dsm.D_out, D_out)

        dsm.D_out_start_idx = 0
        dsm.D_out_header = []
        dsm.clusters = []
        chrom = [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1]
        dsm._DSM__build_clustered_DSM(chrom)
        D_out = [['.', 'x', 'x', '', '', ''],
                 ['x', '.', 'x', '', '', ''],
                 ['x', 'x', '.', '', '', ''],
                 ['', '', '', '.', 'x', 'x'],
                 ['', '', '', 'x', '.', 'x'],
                 ['', '', '', 'x', 'x', '.']]
        self.assertEqual(dsm.D_out, D_out)

        dsm.D_out_start_idx = 0
        dsm.D_out_header = []
        dsm.clusters = []
        chrom = [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
        dsm._DSM__build_clustered_DSM(chrom)
        D_out = [['.', 'x', 'x', '', '', ''],
                 ['x', '.', 'x', '', '', ''],
                 ['x', 'x', '.', '', '', ''],
                 ['', '', '', '.', 'x', 'x'],
                 ['', '', '', 'x', '.', 'x'],
                 ['', '', '', 'x', 'x', '.']]
        self.assertEqual(dsm.D_out, D_out)

        dsm.D_out_start_idx = 0
        dsm.D_out_header = []
        dsm.clusters = []
        chrom = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0,  0, 1, 1, 0, 0]
        dsm._DSM__build_clustered_DSM(chrom)
        D_out = [['.', 'x', 'x', '', ''],
                 ['x', '.', 'x', '', ''],
                 ['x', 'x', '.', '', ''],
                 ['', '', '', '.', 'x'],
                 ['', '', '', 'x', '.']]
        self.assertEqual(dsm.D_out, D_out)

    def test_build_clustered_DSM_with_bus(self):
        """ Test build_clustered_DSM method with_bus """
        num_square_clusters = 2
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        dsm._DSM__build_clustered_DSM(chrom)
        D_out = [['.', 'x', 'x', '', ''],
                 ['x', '.', 'x', '', ''],
                 ['x', 'x', '.', '', ''],
                 ['', '', '', '.', 'x'],
                 ['', '', '', 'x', '.']]
        self.assertEqual(dsm.D_out, D_out)

        dsm.D_out_start_idx = 0
        dsm.D_out_header = []
        dsm.clusters = []
        chrom = [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
        dsm._DSM__build_clustered_DSM(chrom)
        D_out = [['.', 'x', 'x', '', 'x'],
                 ['x', '.', 'x', '', 'x'],
                 ['x', 'x', '.', '', 'x'],
                 ['', '', '', '.', 'x'],
                 ['x', 'x', 'x', 'x', '.']]
        self.assertEqual(dsm.D_out, D_out)

    def test__cleanup_clusters(self):
        """ Test cleanup_clusters method """
        num_square_clusters = 4
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)
        dsm.n = 7
        clusters = [[[1, 1, 1, 1, 0, 0, 0], 4],
                    [[0, 0, 1, 0, 1, 0, 0], 2],
                    [[0, 0, 1, 1, 1, 0, 0], 3],
                    [[0, 0, 0, 0, 0, 1, 1], 2],
                    [[1, 1, 1, 0, 0, 0, 0], 3]]
        [num_overlap, overlap] = dsm._DSM__cleanup_clusters(clusters)
        num_overlap_expected_result = [[[0, 0, 0, 0, 0, 1, 1], 2]]
        overlap_expected_result = [[[1, 1, 1, 1, 0, 0, 0], 4],
                                   [[0, 0, 1, 1, 1, 0, 0], 3]]
        self.assertEqual(num_overlap, num_overlap_expected_result)
        self.assertEqual(overlap, overlap_expected_result)

    def test__arrange_D_using_BEA(self):
        """ Test arrange_D_using_BEA method """
        num_square_clusters = 4
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)
        dsm.n = 7
        dsm.column_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        dsm.D = [[0, 1, 1, 1, 0, 1, 0],
                 [1, 0, 1, 1, 0, 0, 0],
                 [1, 1, 0, 1, 1, 0, 0],
                 [1, 1, 1, 0, 1, 0, 0],
                 [0, 0, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1, 0]]
        nonoverlapping_clusters = [[[0, 0, 0, 0, 0, 1, 1], 2]]
        overlapping_clusters = [
            [[1, 1, 1, 1, 0, 0, 0], 4], [[0, 0, 1, 1, 1, 0, 0], 3]]
        dsm._DSM__arrange_D_using_BEA(nonoverlapping_clusters,
                                      overlapping_clusters)
        CA = [['.', 'x', 'x', '', '', '', ''],
              ['x', '.', 'x', 'x', 'x', '', ''],
              ['x', 'x', '.', 'x', 'x', '', ''],
              ['', 'x', 'x', '.', 'x', 'x', ''],
              ['', 'x', 'x', 'x', '.', '', ''],
              ['', '', '', '', '', '.', 'x'],
              ['', '', '', '', '', 'x', '.']]
        self.assertEqual(dsm.CA, CA)
        self.assertEqual(dsm.CA_header, ['E', 'D', 'C', 'A', 'B', 'F', 'G'])

    def test_D_2_AA(self):
        """ Test D_2_AA method """
        D = [[0, 1, 1, 1, 0, 0, 0],
             [1, 0, 1, 1, 0, 1, 0],
             [1, 1, 0, 1, 1, 0, 0],
             [1, 1, 1, 0, 1, 0, 0],
             [0, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 1, 0]]
        num_square_clusters = 4
        num_busses = 1
        num_sinks = 0
        num_sources = 0

        dsm = DSM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_square_clusters,
                  num_busses, num_sinks, num_sources, ALPHA, BETA,
                  POPULATION_SIZE, OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT,
                  GENERATION_LIMIT_WITHOUT_IMPROVEMENT)
        dsm.n = 7
        dsm.D = D
        res = dsm._DSM__D_2_AA()
        D_column = [[1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1]]
        self.assertEqual(res, D_column)


if __name__ == '__main__':
    unittest.main()
