"""Module providing DMM class."""
from typing import List
from operator import itemgetter
from src.mdl import MDL
import src.ga as ga
from src.datatypes import Chromosome, Matrix, Population, ClusterNames, MatrixOut

class DMM:
    """
    Cluster binary DMM based on the genetic algorithm proposed in Yu, Yassine, and
    Goldberg (2007) and several modifications added to specifically support DMM

    While there's some similarity with DSM class, DMM doesn't inherit from it as 
    there's not much of an overlap in implementation of functionality

    @author: Dilum Bandara
    """

    def __init__(self,
                 input_file_name: str,
                 output_file_name: str,
                 num_clusters: int,
                 alpha: float,
                 beta: float,
                 population_size: int,
                 offspring_size: int,
                 p_c: float,
                 p_m: float,
                 generation_limit: int,
                 generation_limit_without_improvement: int):
        """
        Parameters
        ----------
        input_file_name : str
            File with 2D DSM Matrix
        output_file_name : str
            File with clustered 2D DSM Matrix
        num_clusters :int
            No of clusters to form
        gaCfg : ConfigParser
            Configuration
        beta :float
            Beta weight in Eq. 3
        population_size :int
            Initial population size (i.e., lambda in paper)
        offspring_size : int
            No of off springs (i.e., mu in paper)
        p_c : float
            Crossover probability
        p_m : float
            Mutation probability
        numIterationsMax : int
            Maximum no of iterations
        numIterationsWithNoImprovement : int
            No of iterations with no improvement
        """
        self.__read_DMM(input_file_name)
        self.n = len(self.D)
        self.m = len(self.D[0])
        self.output_file_name = output_file_name
        self.num_clusters = num_clusters
        self.mdl = MDL(self.n * self.m, num_clusters, alpha, beta)

        self.population_size = population_size
        self.offspring_size = offspring_size
        self.p_c = p_c
        self.p_m = p_m
        self.generation_limit = generation_limit
        self.generation_limit_without_improvement = generation_limit_without_improvement

        self.clusters: ClusterNames = []
        self.D_out: MatrixOut = []
        self.D_out_header_row: List[str] = []
        self.D_out_header_column: List[str] = []
        self.D_out_start_idx_row = 0
        self.D_out_start_idx_column = 0

    def __read_DMM(self, file_name: str):
        """
        Read binary DMM Matrix and set D and D' matrices and row and column names

        Binary DMM data format should be as follows:
        ,col1, col2, col3, col4
        row1,,1,,1
        row2,,,1,1
        row3,1,,,1
        0, 1, ., o, O, x, X, or blank could be used as matrix values

        Parameters
        ----------
        file_name :str
            File name to read matrix
        """
        is_first_line = True
        self.column_names = []
        self.row_names = []
        self.points: Matrix = []    # Track marked points
        self.D: Matrix = []
        self.D_dash: Matrix = []
        n = 0
        m = 0

        with open(file_name, "r", encoding="utf-8") as fd:

            for line in fd:
                line = line.replace('\n', '')
                line_splitted = line.split(',')

                if is_first_line:
                    self.column_names = line_splitted[1:]
                    m = len(line_splitted)
                    is_first_line = False
                    continue

                if m != len(line_splitted):
                    self.D = []
                    self.D_dash = []
                    fd.close()
                    raise ValueError("No of columns not same across rows")

                self.row_names.append(line_splitted[0])
                self.D.append([])
                self.D_dash.append([])
                for i in range(1, len(line_splitted)):
                    if (line_splitted[i] == '' or line_splitted[i] == '.' or
                            line_splitted[i] == '0' or line_splitted[i].lower() == 'o'):
                        self.D[n].append(0)
                        self.D_dash[n].append(0)
                    elif line_splitted[i] == '1' or line_splitted[i].lower() == 'x':
                        self.D[n].append(1)
                        self.D_dash[n].append(0)
                        self.points.append([n, i - 1])
                    else:
                        self.D = []
                        self.D_dash = []
                        raise ValueError(
                            f"Invalid element {line_splitted[i]} in DMM")
                n += 1
            self.num_points = len(self.points)
            fd.close()

    def __all_pairs(self, rows: List[int], columns: List[int]) -> List[Chromosome]:
        """
        Given 2 lists generate list of all pairs of items

        Parameters
        ----------
        rows : List[int]
            List of rows
        columns : List[int]
            List of columns

        Returns
        -------
        Chromosome
            List of all pairs of items
        """
        pairs: List[Chromosome] = []
        for i in rows:
            for j in columns:
                pairs.append([i, j])
        return pairs

    def __build_D_dash(self, chromosome: Chromosome) -> int:
        """
        Build D' for the given chromosome

        Parameters
        ----------
        chromosome : Chromosome
            Chromosome to build the DMM

        Returns
        -------
        int
            Total no of objects in clusters
        """
        # Reset D' matrix
        for i in range(self.n):
            for j in range(self.m):
                self.D_dash[i][j] = 0

        num_objects_in_all_clusters = 0
        # Mark one cluster at a time
        for i in range(self.num_clusters):
            objects_in_cluster = []

            for j in range(self.num_points):
                idx = i * self.num_points + j
                if chromosome[idx] == 1:
                    objects_in_cluster.append(j)

            row_index_in_D = []
            column_index_in_D = []
            for p in objects_in_cluster:
                row_index_in_D.append(self.points[p][0])
                column_index_in_D.append(self.points[p][1])
            row_unique_values = list(set(row_index_in_D))
            column_unique_values = list(set(column_index_in_D))

            # A single row (or part of a it) is not a valid cluster. So penalize
            # this chromosome by expanding cluster by adding adjacent rows
            # Round up if first or last raw
            if (len(row_unique_values)) == 1:
                row_unique_values.append((row_unique_values[0] - 1) % self.n)
                row_unique_values.append((row_unique_values[0] + 1) % self.n)

            pairs = self.__all_pairs(row_unique_values, column_unique_values)
            for p in pairs:
                self.D_dash[p[0]][p[1]] = 1
                num_objects_in_all_clusters += 1

        return num_objects_in_all_clusters

    def __calculate_MDL_value(self, chromosome: Chromosome) -> float:
        """
        Calculate MDL metric as per Eq. 3.

        Parameters
        ----------
        chromosome : Chromosome
            Chromosome to build the DMM

        Returns
        -------
        float
            MDL metric value
        """
        # Build D' matrix
        sum_c_i = self.__build_D_dash(chromosome)

        # Calculate type 1 and 2 errors
        num_type_1_errors = 0
        num_type_2_errors = 0
        for i in range(self.n):
            for j in range(self.m):
                delta = self.D[i][j] - self.D_dash[i][j]
                if delta == 0:
                    continue
                if delta < 0:
                    num_type_1_errors += 1
                else:
                    num_type_2_errors += 1

        return self.mdl.value(sum_c_i, num_type_1_errors, num_type_2_errors)

    def __fitness(self, chromosomes: Population) -> Population:
        """
        Calculate fitness score based on MDL principle for given set of chromosomes

        Parameters
        ----------
        chromosomes : Population
            Set of chromosomes

        Returns
        -------
        Population
            List of (chromosome, fitness value) pairs
        """

        for i in range(len(chromosomes)):
            chromosomes[i][1] = self.__calculate_MDL_value(chromosomes[i][0])
        return chromosomes

    def cluster(self):
        """
        Cluster the DSM using GA as per the steps given in Section 5 of paper
        """
        no_change_count = 0

        # Step 1 - Generate population and then calculate fitness value
        chromosome_len = len(self.points) * self.num_clusters
        population = ga.generate_population(
            self.population_size, chromosome_len)
        population = self.__fitness(population)

        for i in range(self.generation_limit):
            # Step 2 - Generate offspring
            offspring = ga.crossover(population, self.offspring_size, self.p_c)

            # Step 3 - Mutate offspring
            offspring = ga.mutate(offspring, self.p_m)

            # Step 4 - Calculate fitness value for offspring
            offspring = self.__fitness(offspring)

            # Step 5 - lambda + mu selection
            population_plus_offspring = population + offspring
            population_plus_offspring.sort(key=itemgetter(1))
            next_generation = population_plus_offspring[:self.population_size]

            if ga.is_next_generation_same(population, next_generation, chromosome_len):
                no_change_count += 1
            else:
                no_change_count = 0
            if no_change_count == self.generation_limit_without_improvement:
                print(
                    f'No more improvements. Stopped after {i + 1} iterations')
                break

            population = next_generation

        self.__build_clustered_DMM(next_generation[0][0])
        self.__save_matrix(next_generation[0][1])

    def __build_clustered_DMM(self, chromosome: Chromosome):
        """
        Build clustered DMM

        Parameters
        ----------
        chromosome : Chromosome
            A chromosome with membership of all clusters
        """
        # Create the list of clusters. Also, count no of points in a cluster
        # and no of items a point appears in clusters
        #TODO refactor code to reduce statements, variable, and branches
        clusters = []
        num_points = len(self.points)
        overlap_count = [0] * num_points
        overlapping_clusters = []
        nonoverlapping_clusters = []
        partially_overlapping_clusters = []
        matrix_size_rows = 0
        matrix_size_columns = 0

        for i in range(self.num_clusters):
            cluster = chromosome[i * num_points: (i + 1) * num_points]
            num_cluster_members = 0
            points_in_cluster = []
            points_in_cluster_row_column = []

            for j in range(num_points):
                if cluster[j] == 1:
                    num_cluster_members += 1
                    overlap_count[j] += 1
                    points_in_cluster.append(j)
                    points_in_cluster_row_column.append(self.points[j])

            if num_cluster_members > 1:  # Non-zero cluster
                row_index_values = []
                for p in points_in_cluster:
                    row_index_values.append(self.points[p][0])
                row_unique_values = list(set(row_index_values))

                if (len(row_unique_values)) > 1:  # Cluster must includes more than 1 row
                    clusters.append(
                        (cluster, num_cluster_members, points_in_cluster_row_column))

        # Identify overlapping and nonoverlapping clusters
        for cluster in clusters:
            overlap = False
            for j in range(num_points):
                if cluster[0][j] == 1 and overlap_count[j] > 1:
                    overlap = True
                    break
            if overlap:
                overlapping_clusters.append(cluster)
            else:
                nonoverlapping_clusters.append(cluster)

        # Remove completely overlapping clusters
        # Sort from smallest to largest
        overlapping_clusters.sort(key=itemgetter(1))

        l = len(overlapping_clusters)
        for i in range(l):
            add_to_cluster_list = True
            for j in range(i + 1, l):
                # Perform an OR operation among objects in 2 clusters
                or_result = [0] * num_points
                for k in range(num_points):
                    if overlapping_clusters[i][0][k] != overlapping_clusters[j][0][k]:
                        or_result[k] = 1
                    else:
                        or_result[k] = overlapping_clusters[i][0][k]
                # Not a complete overlap

                if or_result == overlapping_clusters[j][0]:
                    add_to_cluster_list = False
                    break
            # If not a subcluster or the last cluster, add to list
            if add_to_cluster_list:
                partially_overlapping_clusters.append(
                    overlapping_clusters[i])

        # List non-overlapping clusters starting from the largest
        nonoverlapping_clusters.sort(key=itemgetter(1), reverse=True)

        # List overlapping clusters starting from the largest
        partially_overlapping_clusters.sort(
            key=itemgetter(1), reverse=True)

        # TODO Try to bring overlapping clusters closer

        # Build matrix
        for chrome in nonoverlapping_clusters:
            row_index_values = []
            column_index_values = []
            for r, c in chrome[2]:
                row_index_values.append(r)
                column_index_values.append(c)
            matrix_size_rows += len(set(row_index_values))
            matrix_size_columns += len(set(column_index_values))

        for chrome in partially_overlapping_clusters:
            row_index_values = []
            column_index_values = []
            for r, c in chrome[2]:
                row_index_values.append(r)
                column_index_values.append(c)
            matrix_size_rows += len(set(row_index_values))
            matrix_size_columns += len(set(column_index_values))

        self.D_out = [['' for i in range(matrix_size_columns)]
                      for j in range(matrix_size_rows)]

        # Populate matrix
        for chrome in nonoverlapping_clusters:
            self.__add_cluster_to_matrix(chrome[2])
        for chrome in partially_overlapping_clusters:
            self.__add_cluster_to_matrix(chrome[2])

    def __add_cluster_to_matrix(self, cluster: Matrix):
        """
        Add cluster to output matrix

        Parameters
        ----------
        cluster : Matrix
            An array with the points (row and column index) of members of one cluster
        """
        row_indexes = []
        column_indexes = []
        members = []

        for p in cluster:
            row_indexes.append(p[0])
            column_indexes.append(p[1])
        row_unique_indexes = list(set(row_indexes))
        column_unique_indexes = list(set(column_indexes))

        for r in row_unique_indexes:
            self.D_out_header_row.append(self.row_names[r])
        for c in column_unique_indexes:
            self.D_out_header_column.append(self.column_names[c])

        pairs = self.__all_pairs(row_unique_indexes, column_unique_indexes)
        for r, c in pairs:
            members.append(
                [self.row_names[r], self.column_names[c]])  # Add symbolic name
        self.clusters.append(members)

        row_end_idx = self.D_out_start_idx_row + len(row_unique_indexes)
        column_end_idx = self.D_out_start_idx_column + \
            len(column_unique_indexes)
        for r in range(self.D_out_start_idx_row, row_end_idx):
            for c in range(self.D_out_start_idx_column, column_end_idx):
                self.D_out[r][c] = 'x'
        self.D_out_start_idx_row = row_end_idx
        self.D_out_start_idx_column = column_end_idx

    def __save_matrix(self, fitness: float):
        """
        Save the cluster configuration and clustered DMM to a CSV file

        Parameters
        ----------
        fitness : float
            Fitness score of the chromosome
        """
        with open(self.output_file_name, 'w+', encoding="utf-8") as fd:
            fd.write(f'Fitness score,{fitness}\n')
            print(f'Fitness score:\t{fitness}')

            fd.write('\nClusters\n')

            for c in self.clusters:
                for x_name, y_name in c:
                    fd.write(x_name + ',' + y_name + '\n')
                fd.write('\n')

            fd.write('Clustered DMM\n')
            fd.write(',' + ','.join(self.D_out_header_column) + '\n')

            for i in range(len(self.D_out)):
                fd.write(self.D_out_header_row[i] + ',' +
                        ','.join(self.D_out[i]) + '\n')

            fd.close()
