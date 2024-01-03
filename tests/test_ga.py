""" Unit tests for GA module """

import unittest
import copy
import src.ga as ga


class TestMDL(unittest.TestCase):
    """ Unit tests for GA module """

    def test_generate_chromosome(self):
        """ Test generate_chromosome function """
        chromosome = ga.generate_chromosome(7)
        self.assertEqual(len(chromosome), 7)

        no_errors = True
        for i in chromosome:
            if i not in (0, 1):
                no_errors = False
        self.assertTrue(no_errors)

    def test_generate_population(self):
        """ Test generate_population function """
        population = ga.generate_population(5, 9)
        self.assertEqual(len(population), 5)
        self.assertEqual(len(population[0]), 2)
        self.assertEqual(len(population[0][0]), 9)
        self.assertAlmostEqual(population[0][1], -1.0)

    def test_selection_pair(self):
        """ Test selection_pair function """
        population = ga.generate_population(5, 9)
        selection = ga.selection_pair(population)
        self.assertEqual(len(selection), 2)
        self.assertEqual(len(selection[0][0]), 9)
        self.assertEqual(len(selection[1][0]), 9)
        self.assertEqual(selection[0][1], -1.0)
        self.assertEqual(selection[1][1], -1.0)

    def test_uniform_crossover(self):
        """ Test uniform_crossover function """
        chromosome_pair = [[[1, 1, 1, 1, 1, 0, 0, 0, 0], -1.0],
                           [[0, 0, 1, 1, 0, 0, 0, 1, 0], -1.0]]
        pair = ga.uniform_crossover(chromosome_pair, 0.5)
        self.assertEqual(len(pair), 2)
        self.assertEqual(len(pair[0][0]), 9)
        self.assertEqual(pair[0][1], -1.0)

        no_errors = True
        for i in range(len(chromosome_pair[0][0])):
            if ((chromosome_pair[0][0][i] + chromosome_pair[1][0][i]) !=
                    (pair[0][0][i] + pair[1][0][i])):
                no_errors = False
        self.assertTrue(no_errors)

    def test_crossover(self):
        """ Test crossover function """
        population = ga.generate_population(7, 9)
        next_generation = ga.crossover(population, 6, 0.4)
        self.assertEqual(len(next_generation), 6)
        self.assertEqual(len(next_generation[0][0]), 9)
        self.assertEqual(next_generation[0][1], -1.0)

    def test_mutate(self):
        """ Test mutate function """
        population = ga.generate_population(7, 9)
        next_generation = ga.mutate(population, 0.25)
        self.assertEqual(len(next_generation), 7)
        self.assertEqual(len(next_generation[0][0]), 9)
        self.assertEqual(next_generation[0][1], -1.0)

    def test_is_next_generation_same(self):
        """ Test is_next_generation_same function """
        population = ga.generate_population(7, 9)
        self.assertTrue(ga.is_next_generation_same(population, population, 9))
        population_before = copy.deepcopy(population)
        next_generation = ga.mutate(population, 0.25)
        self.assertFalse(ga.is_next_generation_same(
            population_before, next_generation, 9))


if __name__ == '__main__':
    unittest.main()
