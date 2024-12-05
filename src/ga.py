"""
Genetic algorithm as per Yu, Yassine, and Goldberg (2007)

@author: Dilum Bandara
"""

from random import random, sample, choices
import copy
from src.datatypes import Chromosome, Population


def generate_chromosome(length: int) -> Chromosome:
    """
    Generate a chromosome

    Parameters
    ----------
    length : int
        Length of chromosome
    """

    return choices([0, 1], k=length)


def generate_population(population_size: int, chromosome_len: int) -> Population:
    """
    Create an initial population of chromosomes

    Parameters
    ----------
    population_size :int
        Size of population
    chromosome_len : int
        Length of chromosome
    """
    return [[generate_chromosome(chromosome_len), (-1, 0, 0)] for _ in range(population_size)]


def selection_pair(population: Population) -> Population:
    """
    Select 2 chromosomes from population without replacement

    Returns
    -------
    Population
        Pair of chromosomes
    """

    s = sample(population, k=2)
    return copy.deepcopy(s)


def uniform_crossover(chromosome_pair: Population, probability: float) -> Population:
    """
    Uniform crossover

    Parameters
    ----------
    chromosome_pair : Population
        Pair of chromosomes
    probability : float
        Crossover probability

    Returns
    -------
    Population
        Pair of chromosomes
    """

    length = len(chromosome_pair[0][0])
    tmp = 0

    for i in range(length):
        if random() <= probability:
            tmp = chromosome_pair[0][0][i]
            chromosome_pair[0][0][i] = chromosome_pair[1][0][i]
            chromosome_pair[1][0][i] = tmp
    return chromosome_pair


def crossover(population: Population, num_offspring: int, probability: float) -> Population:
    """
    Crossover chromosomes to produce new offspring chromosomes

    Parameters
    ----------
    population : Population
        Population
    num_offspring : int
        No of offsprings to produce
    probability : float
        Crossover probability

    Returns
    -------
    Population
        List of offspring chromosomes
    """

    offsprings = []
    for _ in range(int(num_offspring/2)):
        chromosome_pair = selection_pair(population)
        new_chromosome_pair = uniform_crossover(chromosome_pair, probability)
        offsprings += new_chromosome_pair
    return offsprings


def mutate(population: Population, probability: float) -> Population:
    """
    Mutate the genes of the offspring chromosomes. For each chromosome mutate 
    each bit randomly based on given probability

    Parameters
    ----------
    population : Population
        Population
    probability : float
        Mutation probability

    Returns
    -------
    Population
        List of mutated chromosomes
    """

    for chromosome in population:
        for i in range(len(chromosome[0])):
            if random() <= probability:
                chromosome[0][i] = abs(chromosome[0][i] - 1)
    return population


def is_next_generation_same(
        population: Population,
        next_generation: Population,
        chromosome_len: int) -> bool:
    """
    Check whether population and next generation are the same

    Parameters
    ----------
    population : Population
        Current population
    next_generation : Population
        Next generation
    chromosome_len : int
        Length of a chromosome
    """

    for i in range(len(population)):
        for k in range(chromosome_len):
            if population[i][0][k] != next_generation[i][0][k]:
                return False
    return True
