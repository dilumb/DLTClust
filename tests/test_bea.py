""" Unit tests for BEA class"""
import unittest
import copy
import math
from src.bea import BEA

# Matrix in column order
AA = [[45, 0, 45, 0],
      [0, 80, 5, 75],
      [45, 5, 53, 3],
      [0, 75, 3, 78]]
AA_column_names = ['a', 'b', 'c', 'd']


class TestBEA(unittest.TestCase):
    """ Unit tests for BEA class"""

    def test_init(self):
        """ Test init method """
        column_names = ['A', 'B', 'C', 'D', 'E']
        clusters = [[0, 1, 2,], [2, 3], [2, 3, 4]]
        bea = BEA(AA, column_names, clusters)
        self.assertEqual(bea.n, len(AA))
        self.assertEqual(bea.AA, AA)
        self.assertEqual(bea.column_names, column_names)
        self.assertEqual(bea.clusters, clusters)

    def test_bond(self):
        """ Test bond method """
        bea = BEA(AA, AA_column_names, [])
        bea.CA = copy.deepcopy(AA)
        self.assertEqual(bea._BEA__bond(0, 1), 45*5)
        self.assertEqual(bea._BEA__bond(1, 2), 80*5 + 5*53 + 75*3)
        self.assertEqual(bea._BEA__bond(2, 3), 5*75 + 3*53 + 3*78)
        self.assertEqual(bea._BEA__bond(0, 2), 45*45 + 45*53)
        self.assertEqual(bea._BEA__bond(1, 3), 80*75 + 5*3 + 75*78)

    def test_cont(self):
        """ Test const method   """
        bea = BEA(AA, AA_column_names, [])
        bea.CA = copy.deepcopy(AA)
        bea._BEA__column_order = [0, 1]
        self.assertEqual(bea._BEA__cont(-1, 2, 0), 8820)
        self.assertEqual(bea._BEA__cont(0, 2, 1), 10150)
        self.assertEqual(bea._BEA__cont(1, 2, 3), 1780)

        AA2 = [[0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 0, 0],
               [1, 0, 0, 1, 0, 0, 0],
               [1, 0, 0, 1, 1, 0, 0],
               [1, 0, 0, 1, 1, 1, 0]]
        column_names = ['C', 'D', 'E', 'A', 'B', 'F', 'G']
        clusters = [[3, 4, 0], [2, 1], [4, 5], [6, 4]]
        bea2 = BEA(AA2, column_names, clusters)
        bea2.CA = copy.deepcopy(AA2)
        bea2._BEA__column_order = [0, 1]
        self.assertEqual(bea2._BEA__cont(-1, 2, 0), -1*math.inf)
        self.assertEqual(bea2._BEA__cont(0, 2, 1), 0)
        self.assertEqual(bea2._BEA__cont(1, 2, 3), 0)
        bea2._BEA__column_order.append(2)
        self.assertEqual(bea2._BEA__cont(-1, 3, 0), 2)
        self.assertEqual(bea2._BEA__cont(0, 3, 1), 2)
        self.assertEqual(bea2._BEA__cont(1, 3, 2), -1*math.inf)
        self.assertEqual(bea2._BEA__cont(2, 3, 4), -1*math.inf)

        self.assertEqual(bea2._BEA__cont(-1, 4, 0), 2)
        self.assertEqual(bea2._BEA__cont(0, 4, 1), 2)
        self.assertEqual(bea2._BEA__cont(1, 4, 2), -1*math.inf)
        self.assertEqual(bea2._BEA__cont(2, 4, 5), 0)
        bea2._BEA__column_order = [4] + bea2._BEA__column_order
        self.assertEqual(bea2._BEA__cont(-1, 5, 0), 4)
        self.assertEqual(bea2._BEA__cont(0, 5, 1),  -1*math.inf)
        self.assertEqual(bea2._BEA__cont(1, 5, 2), -1*math.inf)
        self.assertEqual(bea2._BEA__cont(2, 5, 6),  -1*math.inf)

    def test_order_rows(self):
        """ Test order_rows method """
        bea = BEA(AA, AA_column_names, [])
        bea.CA = copy.deepcopy(AA)
        bea._BEA__column_order = [0, 1, 2, 3]
        bea._BEA__order_rows()
        self.assertEqual(bea.CA, AA)
        tmpAA = copy.deepcopy(AA)
        tmpAA[1] = [0, 75, 5, 80]
        tmpAA[2] = [45, 3, 53, 5]
        tmpAA[3] = [0, 78, 3, 75]
        bea._BEA__column_order = [0, 3, 2, 1]
        bea._BEA__order_rows()
        self.assertEqual(bea.CA, tmpAA)

    def test_is_splitting_cluster(self):
        """ Test is_splitting_cluster method """
        AA2 = [[0, 0, 0, 1, 1],
               [0, 0, 1, 0, 0],
               [0, 1, 0, 0, 0],
               [1, 0, 0, 0, 1],
               [1, 0, 0, 1, 0]]
        column_names = ['C', 'D', 'E', 'A', 'B']
        clusters = [[3, 4, 0], [2, 1]]
        bea = BEA(AA2, column_names, clusters)
        bea.CA = copy.deepcopy(AA)
        bea._BEA__column_order = [0, 1]
        self.assertEqual(bea._BEA__k_is_splitting_cluster(0, 2, 1), False)
        self.assertEqual(bea._BEA__k_is_splitting_cluster(0, 4, 1), False)

        bea._BEA__column_order = [0, 4, 1]
        self.assertEqual(bea._BEA__k_is_splitting_cluster(0, 3, 1), False)
        self.assertEqual(bea._BEA__k_is_splitting_cluster(0, 2, 1), True)
        self.assertEqual(bea._BEA__k_is_splitting_cluster(1, 3, 2), False)
        self.assertEqual(bea._BEA__k_is_splitting_cluster(1, 2, 2), False)

        clusters2 = [[1, 2, 3], [1, 2, 0]]
        bea2 = BEA(AA2, column_names, clusters2)
        bea2._BEA__column_order = [1, 2]
        self.assertEqual(bea2._BEA__k_is_splitting_cluster(0, 3, 1), False)
        self.assertEqual(bea2._BEA__k_is_splitting_cluster(0, 0, 1), False)
        self.assertEqual(bea2._BEA__k_is_splitting_cluster(0, 4, 1), True)
        bea2._BEA__column_order = [1, 2, 0]
        self.assertEqual(bea2._BEA__k_is_splitting_cluster(0, 3, 1), True)

    def test_is_adjacent_to_cluster_member(self):
        """ Test is_adjacent_to_cluster_member method """
        AA2 = [[0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 0, 0],
               [1, 0, 0, 1, 0, 0, 0],
               [1, 0, 0, 1, 1, 0, 0],
               [1, 0, 0, 1, 1, 1, 0]]
        column_names = ['C', 'D', 'E', 'A', 'B', 'F', 'G']
        clusters = [[3, 4, 0], [2, 1], [4, 5], [6, 4]]
        bea = BEA(AA2, column_names, clusters)
        bea.CA = copy.deepcopy(AA)
        bea._BEA__column_order = [0, 1]
        self.assertEqual(
            bea._BEA__k_is_away_from_cluster_members(0, 8, 1), False)
        self.assertEqual(
            bea._BEA__k_is_away_from_cluster_members(0, 3, 1), False)
        self.assertEqual(
            bea._BEA__k_is_away_from_cluster_members(0, 4, 1), False)
        bea._BEA__column_order.append(2)
        self.assertEqual(
            bea._BEA__k_is_away_from_cluster_members(0, 3, 1), False)
        self.assertEqual(
            bea._BEA__k_is_away_from_cluster_members(1, 3, 2), True)
        bea._BEA__column_order.append(4)
        self.assertEqual(
            bea._BEA__k_is_away_from_cluster_members(0, 5, 1), True)
        self.assertEqual(
            bea._BEA__k_is_away_from_cluster_members(1, 5, 2), True)
        self.assertEqual(
            bea._BEA__k_is_away_from_cluster_members(2, 5, 3), False)

        clusters2 = [[3, 4, 0], [2, 1], [4, 3], [3, 4, 5]]
        bea2 = BEA(AA2, column_names, clusters2)
        bea2.CA = copy.deepcopy(AA2)
        bea2._BEA__column_order = [0, 1]
        self.assertEqual(
            bea2._BEA__k_is_away_from_cluster_members(0, 4, 1), False)
        bea2._BEA__column_order.append(2)
        self.assertEqual(
            bea2._BEA__k_is_away_from_cluster_members(1, 4, 2), False)

    def test_cluster(self):
        """
        Test cluster method
        """
        bea = BEA(AA, AA_column_names, [])
        bea.cluster()
        CA = [[45, 45, 0, 0],
              [45, 53, 5, 3],
              [0, 5, 80, 75],
              [0, 3, 75, 78]]
        self.assertEqual(bea.CA, CA)
        self.assertEqual(bea.column_names, ['a', 'c', 'b', 'd'])

    def test__order_rows(self):
        """ Test order_rows method """
        AA = [[0, 0, 0, 1, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1, 0, 0],
              [1, 0, 0, 1, 0, 0, 0],
              [1, 0, 0, 1, 1, 0, 0],
              [1, 0, 0, 1, 1, 1, 0]]
        CA = [[0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 0, 0],
              [1, 0, 0, 1, 1, 1, 0],
              [1, 0, 0, 1, 1, 0, 0],
              [1, 0, 0, 1, 0, 0, 0],
              [1, 0, 0, 0, 1, 0, 0]]
        CA2 = [[0, 1, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 1],
               [0, 0, 1, 0, 1, 1, 1],
               [0, 0, 1, 0, 0, 1, 1],
               [0, 0, 1, 0, 0, 0, 1],
               [0, 0, 1, 0, 0, 1, 0]]
        column_names = ['C', 'D', 'E', 'A', 'B', 'F', 'G']
        clusters = [[3, 4, 0], [2, 1], [4, 5], [6, 4]]
        bea = BEA(AA, column_names, clusters)
        bea._BEA__column_order = [2, 1, 0, 6, 5, 4, 3]
        bea.CA = copy.deepcopy(CA)
        bea._BEA__order_rows()
        self.assertEqual(bea.CA, CA2)
