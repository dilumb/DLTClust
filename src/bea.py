"""Module providing Bond Energy Algorithm (BEA) class."""
import copy
import math
from src.datatypes import Matrix, Clusters


class BEA:
    """
    Customized version of Bond Energy Algorithm (BEA) [1] to get DSM unclustered
    items closer. While DSM is an asymmetric matrix, we assume it be symmetric
    to preserve the property of decomposability, i.e., rows’ and
    the columns’ contributions to the overall optimization function are easily
    decomposed additively, so that the search for the best column
    ordering can proceed apart from that for the rows [2]. Simplified algorithm
    details are adopted from [3]

    [1] McCormick, William T; Schweitzer, Paul J; and White, Thomas W, "Problem
    Decomposition and Data Reorganization by a Clustering Technique,"
    Operations Research, 20(5), Sep. 1972, 993-1009.
    [2] Arabie, Phipps and Lawrence J. Hubert. "The bond energy algorithm
    revisited." IEEE Trans. on Systems, Man, and Cybernetics, 20.1, 1990, 268-274.
    [3] M. Tamer O¨zsu and Patrick Valduriez, "Principles of Distributed
    Database Systems," 4th Edition.

    @author: Dilum Bandara
    """

    def __init__(self, AA: Matrix, column_names: list, clusters: Clusters):
        """
        Parameters
        ----------
        AA : Matrix
            Input DSM Matrix to rearrange
            Term AA is used from [3] to make it easier to follow algorithm
        column_names : list
            List of column names of input matrix
        clusters : Clusters
            List of clusters.
        """
        assert (len(column_names) > 0), 'Empty column names list'
        # Term AA is used from [3] to make it easier to follow algorithm
        self.n = len(AA)
        self.AA: Matrix = AA
        self.column_names = column_names
        self.clusters: Clusters = clusters
        self.CA: Matrix = []
        self.__column_order = []

    def __bond(self, i: int, j: int) -> int:
        """
        Calculate bond energy between 2 columns

        Parameters
        ----------
        i : int
            Left column index
        j : int
            Right column index

        Returns
        -------
        int
            Bond energy
        """
        total = 0
        for k in range(self.n):
            total += self.CA[i][k] * self.CA[j][k]
        return total

    def __cont(self, i: int, k: int, j: int) -> float:
        """
        Calculate contribution to the global affinity measure when column k
        is placed between columns i and j

        Parameters
        ----------
        i : int
            Left column index
        j : int
            Right column index
        k : int
            Index of column to be inserted

        Returns
        -------
        int
            Contribution to the global affinity measure
        """
        # If any other cluster member is already in, k must be adjacent to that
        if self.__k_is_away_from_cluster_members(i, k, j):
            return -1 * math.inf

        if i == -1:  # Left most column
            return 2 * self.__bond(k, j)
        if j == k+1:  # Right most column
            return 2 * self.__bond(i, k)

        # If new column is going to split an existing cluster, consider it
        # as not contributing to the global affinity measure
        if self.__k_is_splitting_cluster(i, k, j):
            return -1 * math.inf

        # If placed between 2 columns
        return (2 * self.__bond(i, k) + 2 * self.__bond(k, j)
                - 2 * self.__bond(i, j))

    def __k_is_splitting_cluster(self, i: int, k: int, j: int) -> bool:
        """
        Is the addition of new column k going to split an existing cluster?

        Parameters
        ----------
        i : int
            Left column index
        k : int
            Index of column to be inserted
        j : int
            Right column index

        Returns
        -------
        Boolean
            Split or not
        """
        i_index = self.__column_order[i]
        j_index = self.__column_order[j]

        clusters_i_j_in = []
        for c in self.clusters:  # Is i and j in same cluster?
            if (i_index in c) and (j_index in c):
                clusters_i_j_in.append(c)
        if len(clusters_i_j_in) == 0:  # No such cluster
            return False

        clusters_k_in = []
        clusters_k_not_in = []
        for c_i_j in clusters_i_j_in:  # Is i and j in same cluster but not k?
            if (i_index in c_i_j) and (j_index in c_i_j) and (k in c_i_j):
                clusters_k_in.append(c_i_j)
            if (i_index in c_i_j) and (j_index in c_i_j) and (k not in c_i_j):
                clusters_k_not_in.append(c_i_j)
        # K is in all clusters. No split
        if (len(clusters_k_in) > 0) and (len(clusters_k_not_in) == 0):
            return False
        # K is not in all clusters. Will split
        if (len(clusters_k_in) == 0) and (len(clusters_k_not_in) > 0):
            return True

        for c_i_j_k in clusters_k_not_in:  # Are other members already placed?
            for m in c_i_j_k:
                if m in (i_index, j_index):
                    continue
                if m in self.__column_order:
                    return True

        return False

    def __k_is_away_from_cluster_members(self, i: int, k: int, j: int) -> bool:
        """
        If k is in a cluster and at least 1 of the other cluster members is
        already added, is k is away from to that member(s)

        Parameters
        ----------
        i : int
            Left column index
        j : int
            Right column index
        k : int
            Index of column to be inserted

        Returns
        -------
        Boolean
            Return False if k is not in a cluster, no cluster member already
            in is adjacent to k, or k is in more than 2 clusters (then only 
            some of them can be adjacent). Else, return True.
        """
        k_cluster_members = []
        # Count when k not in overlapping part of a cluster
        num_times_k_in_cluster = 0
        for c in self.clusters:
            if k in c:   # Is k in cluster?
                num_times_k_in_cluster += 1
                # Record other members if they aren't already in the list
                for o in c:
                    if (o != k) and (o not in k_cluster_members):
                        k_cluster_members.append(o)
        if len(k_cluster_members) == 0:  # k not in a cluster
            return False

        if num_times_k_in_cluster > 2:  # k is in 3+ clusters. Can't do much
            return False

        k_cluster_members_already_in = []
        for m in k_cluster_members:
            if m in self.__column_order:
                k_cluster_members_already_in.append(m)
        if len(k_cluster_members_already_in) == 0:  # k is 1st one to be added
            return False

        # Skip left most column
        if (i != -1) and self.__column_order[i] in k_cluster_members_already_in:
            return False
        # Skip right most column
        if (j != k+1) and self.__column_order[j] in k_cluster_members_already_in:
            return False

        return True

    def __order_rows(self):
        """
        Order rows of CA based on column order
        Results is still stored in CA
        """
        for i in range(self.n):
            c = copy.deepcopy(self.CA[i])
            for j in range(self.n):
                self.CA[i][j] = c[self.__column_order[j]]

    def cluster(self):
        """
        Cluster AA using BEA
        Results is stored in CA. Column order is stored in column_order
        """
        self.one_mode_cluster()   # Rearrange columns
        # Arrange CA to be row order
        # DRow: Matrix = []
        # for i in range(self.n):
        #     row = []
        #     for j in range(self.n):
        #         row.append(self.CA[j][i])
        #     DRow.append(row)
        # self.AA: Matrix = DRow
        # self.one_mode_cluster()   # Rearrange rows
        # print(self.__column_order)

    def one_mode_cluster(self):
        """
        Cluster AA using one pass of BEA
        Results is stored in CA. Column order is stored in column_order
        """
        # Copy columns 1 and 2 as it is to clustered AA (CA).
        # To simplify calculation copy rest too.
        self.CA: Matrix = copy.deepcopy(self.AA)
        self.__column_order = [0, 1]
        index = 2

        while index < self.n:
            cont_values = []
            for i in range(index):
                cont_values.append(self.__cont(i-1, index, i))
            cont_values.append(self.__cont(index-1, index, index+1))

            # Find location of maximum contribution
            cont_max = max(cont_values)
            loc = cont_values.index(cont_max)

            # Adjust columns
            for j in range(index, loc, -1):
                self.CA[j] = copy.deepcopy(self.CA[j-1])
            self.CA[loc] = copy.deepcopy(self.AA[index])

            self.__column_order = self.__column_order[:loc] + \
                [index] + self.__column_order[loc:]
            index += 1

        # Order rows and their names according to relative ordering of columns
        self.__order_rows()
        new_column_names: list = []
        for i in self.__column_order:
            new_column_names.append(self.column_names[i])
        self.column_names = new_column_names
