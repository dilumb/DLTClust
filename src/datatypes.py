"""Module providing datatype definitions."""
from typing import List, Tuple
from enum import Enum
from collections import namedtuple

type Matrix = List[List[int]]
type MatrixOut = List[List[str]]
type ClusterNames = List[List[str]]
type Chromosome = List[int]
type Population = List[Tuple[Chromosome, float]]
type Clusters = List[Tuple[Chromosome, int]]

Config = namedtuple(
    'Config', ['alpha', 'beta', 'population_size', 'offspring_size',
               'p_c', 'p_m', 'num_iterations_max', 'num_iterations_without_improvement',
               'cluster_can_have_read_only_elements', 'cluster_can_have_partial_bus',
               'cluster_can_have_partial_sink', 'cluster_can_have_partial_source'])


class ClusterType(Enum):
    """
    Type of cluster
    """
    SQUARE = 1
    BUS = 2
    READER = 3
    WRITER = 4
