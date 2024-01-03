""" Unit tests for MDL class """
import unittest
import math

from src.mdl import MDL

class TestMDL(unittest.TestCase):
    """ Unit tests for MDL class """

    def test_init(self):
        """ Test init method """
        num_elements = 10
        num_clusters = 5
        alpha = 0.8116
        beta = 0.1102

        mdl = MDL(num_elements, num_clusters, alpha, beta)

        self.assertEqual(mdl.n_n, num_elements)
        self.assertEqual(mdl.n_c, num_clusters)
        log_n_n = math.log2(num_elements)
        self.assertEqual(mdl.log_n_n, log_n_n)
        self.assertEqual(mdl.n_c_log_n_n, num_clusters * log_n_n)
        self.assertEqual(mdl.two_log_n_n_plus_1, 2 * log_n_n + 1)
        self.assertEqual(mdl.alpha, alpha)
        self.assertEqual(mdl.beta, beta)
        self.assertEqual(mdl.gamma, 1.0 - alpha - beta)

    def test_value1(self):
        """ Test value method """
        num_elements = 10
        num_clusters = 5
        alpha = 0.8116
        beta = 0.1102
        sum_c_i = 12
        type_1_size = 2
        type_2_size = 3

        mdl = MDL(num_elements, num_clusters, alpha, beta)

        self.assertAlmostEqual(
            mdl.value(sum_c_i, type_1_size, type_2_size), 19.35073743)
        type_1_size = 5
        self.assertAlmostEqual(
            mdl.value(sum_c_i, type_1_size, type_2_size), 37.96199848)
        type_2_size = 4
        self.assertAlmostEqual(
            mdl.value(sum_c_i, type_1_size, type_2_size), 38.80435144)
        sum_c_i = 15
        self.assertAlmostEqual(
            mdl.value(sum_c_i, type_1_size, type_2_size), 39.58367577)

    def test_value2(self):
        """ Test value method for different combination """
        num_elements = 50
        num_clusters = 10
        alpha = 0.5
        beta = 0.25
        sum_c_i = 75
        type_1_size = 8
        type_2_size = 5

        mdl = MDL(num_elements, num_clusters, alpha, beta)

        self.assertAlmostEqual(
            mdl.value(sum_c_i, type_1_size, type_2_size), 184.442434)
        type_1_size = 13
        self.assertAlmostEqual(
            mdl.value(sum_c_i, type_1_size, type_2_size), 215.161715)
        type_2_size = 7
        self.assertAlmostEqual(
            mdl.value(sum_c_i, type_1_size, type_2_size), 221.3055712)
        sum_c_i = 88
        self.assertAlmostEqual(
            mdl.value(sum_c_i, type_1_size, type_2_size), 239.6481038)


if __name__ == '__main__':
    unittest.main()
