""" Unit tests for DMM class"""
import unittest
from pathlib import Path
from src.dmm import DMM

MOD_PATH = str(Path(__file__).parent)
INPUT_FILE_NAME = MOD_PATH + "/test_dmm.csv"
OUTPUT_FILE_NAME = MOD_PATH + "/test_dmm_clusters.csv"
ALPHA = 0.8116
BETA = 0.1102
P_C = 0.5
P_M = 0.25
POPULATION_SIZE = 10
OFFSPRING_SIZE = 10
GENERATION_LIMIT = 10
GENERATION_LIMIT_WITHOUT_IMPROVEMENT = 2


class TestDMM(unittest.TestCase):
    """ Unit tests for DMM class"""

    def test_init(self):
        """ Test init method """
        num_clusters = 5

        dmm = DMM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_clusters, ALPHA, BETA, POPULATION_SIZE,
                  OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT, GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        n = 7
        m = 5
        self.assertEqual(dmm.n, n)
        self.assertEqual(dmm.m, m)
        self.assertEqual(dmm.output_file_name, OUTPUT_FILE_NAME)
        self.assertEqual(dmm.num_clusters, num_clusters)
        self.assertEqual(dmm.population_size, POPULATION_SIZE)
        self.assertEqual(dmm.offspring_size, OFFSPRING_SIZE)
        self.assertEqual(dmm.p_c, P_C)
        self.assertEqual(dmm.p_m, P_M)
        self.assertEqual(dmm.generation_limit, GENERATION_LIMIT)
        self.assertEqual(dmm.generation_limit_without_improvement,
                         GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        self.assertEqual(dmm.mdl.n_n, n * m)
        self.assertEqual(dmm.mdl.n_c, num_clusters)
        self.assertEqual(dmm.mdl.alpha, ALPHA)
        self.assertEqual(dmm.mdl.beta, BETA)
        self.assertEqual(dmm.mdl.gamma, 1 - ALPHA - BETA)

    def test_read_DMM(self):
        """ Test read_DMM method """
        num_clusters = 5

        dmm = DMM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_clusters, ALPHA, BETA, POPULATION_SIZE,
                  OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT, GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        n = 7
        m = 5
        d = [[0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [1, 1, 1, 1, 0], [
            0, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 0, 0, 1, 0], [1, 1, 1, 1, 0]]
        D_dash = [[0] * m] * n
        self.assertEqual(len(dmm.D), n)
        self.assertEqual(len(dmm.D[0]), m)
        self.assertEqual(dmm.D, d)
        self.assertEqual(len(dmm.D_dash), n)
        self.assertEqual(len(dmm.D_dash[0]), m)
        self.assertEqual(dmm.D_dash, D_dash)

        points = [[0, 3], [1, 1], [2, 0], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [
            3, 3], [4, 0], [4, 2], [4, 4], [5, 0], [5, 3], [6, 0], [6, 1], [6, 2], [6, 3]]
        self.assertEqual(dmm.points, points)

    def test_all_pairs(self):
        """ Test all_pairs method """
        num_clusters = 5

        dmm = DMM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_clusters, ALPHA, BETA, POPULATION_SIZE,
                  OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT, GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        pairs = [[1, 3], [1, 2], [1, 4], [3, 3], [3, 2], [3, 4]]
        self.assertEqual(dmm._DMM__all_pairs([1, 3], [3, 2, 4]), pairs)

    def test_build_D_dash(self):
        """ Test build_D_dash method """
        num_clusters = 3

        dmm = DMM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_clusters, ALPHA, BETA, POPULATION_SIZE,
                  OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT, GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        D_dash = [[0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 0], [
            0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 0]]
        self.assertEqual(dmm._DMM__build_D_dash(chrom), 22)
        self.assertEqual(dmm.D_dash, D_dash)

        chrom = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        D_dash = [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 0], [
            0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 0]]
        self.assertEqual(dmm._DMM__build_D_dash(chrom), 16)
        self.assertEqual(dmm.D_dash, D_dash)

        chrom = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        D_dash = [[0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 0], [
            1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 1], [1, 1, 1, 1, 0]]
        self.assertEqual(dmm._DMM__build_D_dash(chrom), 22)
        self.assertEqual(dmm.D_dash, D_dash)

    def test_calculate_MDL_value(self):
        """ Test calculate_MDL_value method """
        num_clusters = 3

        dmm = DMM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_clusters, ALPHA, BETA, POPULATION_SIZE,
                  OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT, GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        self.assertAlmostEqual(
            dmm._DMM__calculate_MDL_value(chrom), 33.26542859)

        chrom = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        self.assertAlmostEqual(
            dmm._DMM__calculate_MDL_value(chrom), 25.44339874)

        chrom = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        self.assertAlmostEqual(
            dmm._DMM__calculate_MDL_value(chrom), 49.05894502)

    def test_fitness(self):
        """ Test fitness method """
        num_clusters = 3

        dmm = DMM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_clusters, ALPHA, BETA, POPULATION_SIZE,
                  OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT, GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom1 = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        chrom2 = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        chrom3 = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                  1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        chrom = [[chrom1, 0.0], [chrom2, 0.0], [chrom3, 0.0]]

        result = [[chrom1, 33.26542859], [
            chrom2, 25.44339874], [chrom3, 49.05894502]]
        res = dmm._DMM__fitness(chrom)
        self.assertEqual(len(res), len(result))
        self.assertEqual(res[0][0], result[0][0])
        self.assertEqual(res[1][0], result[1][0])
        self.assertEqual(res[2][0], result[2][0])
        self.assertAlmostEqual(res[0][1], result[0][1])
        self.assertAlmostEqual(res[1][1], result[1][1])
        self.assertAlmostEqual(res[2][1], result[2][1])

    def test_save_matrix(self):
        """ Test save_matrix method """
        num_clusters = 3

        dmm = DMM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_clusters, ALPHA, BETA, POPULATION_SIZE,
                  OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT, GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        dmm.clusters = [[['D2', 'P2'], ['D2', 'P1'], ['D2', 'P3'], ['D2', 'P5'], ['D3', 'P2'], \
                         ['D3', 'P1'], ['D3', 'P3'], ['D3', 'P5']],
                        [['D1', 'P4'], ['D2', 'P4'], ['D6', 'P4'], ['D7', 'P4'], ['D3', 'P4']]]
        dmm.D_out_header_column = ['P2', 'P1', 'P3', 'P5', 'P4']
        dmm.D_out_header_row = ['D2', 'D3', 'D1', 'D2', 'D6', 'D7', 'D3']
        dmm.D_out = [['x', 'x', 'x', 'x', ''],
                    ['x', 'x', 'x', 'x', ''],
                    ['', '', '', '', 'x'],
                    ['', '', '', '', 'x'],
                    ['', '', '', '', 'x'],
                    ['', '', '', '', 'x'],
                    ['', '', '', '', 'x']]
        dmm._DMM__save_matrix(25.44339874)

    def test_add_cluster_to_matrix(self):
        """ Test add_cluster_to_matrix method """
        num_clusters = 3

        dmm = DMM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_clusters, ALPHA, BETA, POPULATION_SIZE,
                  OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT, GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom = [[[2, 0], [2, 1], [2, 2], [2, 3], [6, 0], [6, 1], [6, 2], [6, 3]],
                 [[0, 3], [2, 3], [3, 3], [5, 3], [6, 3]]]
        clusters = [[['D2', 'P2'], ['D2', 'P1'], ['D2', 'P3'], ['D2', 'P5'], \
                     ['D3', 'P2'], ['D3', 'P1'], ['D3', 'P3'], ['D3', 'P5']],
                    [['D1', 'P5'], ['D2', 'P5'], ['D6', 'P5'], ['D7', 'P5'], \
                     ['D3', 'P5']]]
        header_row = ['D2', 'D3', 'D1', 'D2', 'D6', 'D7', 'D3']
        header_column = ['P2', 'P1', 'P3', 'P5', 'P5']
        dmm.D_out = [['' for i in range(5)] for j in range(7)]

        dmm._DMM__add_cluster_to_matrix(chrom[0])
        self.assertEqual(dmm.D_out_header_row, header_row[0:2])
        self.assertEqual(dmm.D_out_header_column, header_column[0:4])
        self.assertEqual(dmm.clusters, [clusters[0]])
        self.assertEqual(dmm.D_out_start_idx_row, 2)
        self.assertEqual(dmm.D_out_start_idx_column, 4)

        dmm._DMM__add_cluster_to_matrix(chrom[1])
        self.assertEqual(dmm.clusters, clusters[:2])
        self.assertEqual(dmm.D_out_header_row, header_row)
        self.assertEqual(dmm.D_out_header_column, header_column)
        self.assertEqual(dmm.D_out_start_idx_row, 7)
        self.assertEqual(dmm.D_out_start_idx_column, 5)

        D_out = [['x', 'x', 'x', 'x', ''],
                ['x', 'x', 'x', 'x', ''],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x']]
        self.assertEqual(dmm.D_out, D_out)

    def test_build_clustered_DMM(self):
        """ Test build_clustered_DMM method """
        num_clusters = 3

        dmm = DMM(INPUT_FILE_NAME, OUTPUT_FILE_NAME, num_clusters, ALPHA, BETA, POPULATION_SIZE,
                  OFFSPRING_SIZE, P_C, P_M, GENERATION_LIMIT, GENERATION_LIMIT_WITHOUT_IMPROVEMENT)

        chrom = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        dmm._DMM__build_clustered_DMM(chrom)
        D_out = [['x', 'x', 'x', 'x', ''],
                ['x', 'x', 'x', 'x', ''],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x']]
        self.assertEqual(dmm.D_out, D_out)

        dmm.D_out_start_idx_row = 0
        dmm.D_out_start_idx_column = 0
        dmm.D_out_header_row = []
        dmm.D_out_header_column = []
        dmm.clusters = []

        chrom = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        dmm._DMM__build_clustered_DMM(chrom)
        D_out = [['x', 'x', 'x', 'x', ''],
                ['x', 'x', 'x', 'x', ''],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x']]
        self.assertEqual(dmm.D_out, D_out)

        dmm.D_out_start_idx_row = 0
        dmm.D_out_start_idx_column = 0
        dmm.D_out_header_row = []
        dmm.D_out_header_column = []
        dmm.clusters = []

        chrom = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        dmm._DMM__build_clustered_DMM(chrom)
        D_out = [['x', 'x', 'x', 'x', ''],
                ['x', 'x', 'x', 'x', ''],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x'],
                ['', '', '', '', 'x']]
        self.assertEqual(dmm.D_out, D_out)

        dmm.D_out_start_idx_row = 0
        dmm.D_out_start_idx_column = 0
        dmm.D_out_header_row = []
        dmm.D_out_header_column = []
        dmm.clusters = []

        chrom = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        dmm._DMM__build_clustered_DMM(chrom)
        D_out = [['x', '', '', '', '', ''],
                ['x', '', '', '', '', ''],
                ['', 'x', 'x', 'x', 'x', ''],
                ['', 'x', 'x', 'x', 'x', ''],
                ['', '', '', '', '', 'x'],
                ['', '', '', '', '', 'x'],
                ['', '', '', '', '', 'x'],
                ['', '', '', '', '', 'x'],
                ['', '', '', '', '', 'x']]
        self.assertEqual(dmm.D_out, D_out)

        dmm.D_out_start_idx_row = 0
        dmm.D_out_start_idx_column = 0
        dmm.D_out_header_row = []
        dmm.D_out_header_column = []
        dmm.clusters = []

        chrom = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        dmm._DMM__build_clustered_DMM(chrom)
        D_out = [['x', 'x', 'x', 'x', '', ''],
                ['x', 'x', 'x', 'x', '', ''],
                ['', '', '', '', 'x', ''],
                ['', '', '', '', 'x', ''],
                ['', '', '', '', 'x', ''],
                ['', '', '', '', 'x', ''],
                ['', '', '', '', 'x', ''],
                ['', '', '', '', '', 'x'],
                ['', '', '', '', '', 'x'],
                ['', '', '', '', '', 'x']]
        self.assertEqual(dmm.D_out, D_out)


if __name__ == '__main__':
    unittest.main()
