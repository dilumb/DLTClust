"""Module providing MLD class."""
import math


class MDL:
    """
    Minimal Description Length (MDL) metric as per Yu, Yassine, and Goldberg (2007)

    @author: Dilum Bandara
    """

    def __init__(self, num_objects: int, num_clusters: int, alpha: float, beta: float):
        """
        Parameters:
        num_objects (int) : Number of objects (rows/columns) in DSM
        num_clusters (int) : Number of clusters to form
        alpha (float) : Alpha weight in Eq. 3
        beta (float) : Beta weight in Eq. 3
        """
        self.n_n = num_objects
        self.n_c = num_clusters
        self.log_n_n = math.log2(self.n_n)
        self.n_c_log_n_n = self.n_c * self.log_n_n
        self.two_log_n_n_plus_1 = 2 * self.log_n_n + 1.0
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1 - alpha - beta

    def value(self, sum_c_i: int, type_1_size: int, type_2_size: int) -> float:
        """
        Calculate MDL clustering metric as per Eq. 3
        Parameters:
        sum_c_i (int) : No of nodes in all the clusters
        type_1_size (int) : Type 1 mismatch set size
        type_2_size (int) : Type 2 mismatch set size
        """

        part1 = self.n_c_log_n_n + self.log_n_n * sum_c_i
        part2 = type_1_size * self.two_log_n_n_plus_1
        part3 = type_2_size * self.two_log_n_n_plus_1
        return self.gamma * part1 + self.alpha * part2 + self.beta * part3
