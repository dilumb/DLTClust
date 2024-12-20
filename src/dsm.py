"""Module providing DSM class."""
from operator import itemgetter
from src.mdl import MDL
from src.bea import BEA
import src.ga as ga
from src.datatypes import Chromosome, Matrix, Population, ClusterNames, Clusters, ClusterType, MatrixOut, Fitness
from src.util import plot_overlapping_clusters


class DSM:
    """
    Cluster binary DSM based on the genetic algorithm proposed in Yu, Yassine, and
    Goldberg (2007) and several modifications we added to support data sources, 
    sinks, and valid cluster that are not fully covered in the decentralized ledger
    design context.

    @author: Dilum Bandara
    """

    def __init__(self,
                 input_file_name: str,
                 output_file_name: str,
                 num_square_clusters: int,
                 num_busses: int,
                 num_sinks: int,
                 num_sources: int,
                 alpha: float,
                 beta: float,
                 population_size: int,
                 offspring_size: int,
                 p_c: float,
                 p_m: float,
                 generation_limit: int,
                 generation_limit_without_improvement: int,
                 cluster_can_have_read_only_elements=False,
                 cluster_can_have_partial_bus=False,
                 cluster_can_have_partial_sink=False,
                 cluster_can_have_partial_source=False):
        """
        Parameters
        ----------
        input_file_name : str
            File with 2D DSM Matrix
        output_file_name : str
            File with clustered 2D DSM Matrix
        num_square_clusters :int
            No of square clusters to form
        num_busses : int
            No of busses to from in DSM
        num_sinks : int
            No of sinks, i.e., sinks to from in DSM
        num_sources : int
            No of sources, i.e., sources to from in DSM
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
        cluster_can_have_read_only_elements: bool
            Can a square cluster have real-only objects/parties? Default is False
        Cluster_can_have_partial_bus : bool
            Can a bus have a subset of rows and columns filled? Default is False
        Cluster_can_have_partial_sink : bool
            Can a reader have a subset of column filled? Default is False
        Cluster_can_have_partial_source : bool
            Can a writer have a subset of row filled? Default is False
        """

        if num_square_clusters < 0 or num_busses < 0 or num_sinks < 0 or num_sources < 0:
            raise ValueError(
                "No of clusters, busses, sinks, or sources cannot be negative")

        self.__read_DSM(input_file_name)
        self.n = len(self.D)
        self.output_file_name = output_file_name

        assert (num_square_clusters >= 0), \
            'Num of square clusters can\'t be negative'
        assert (num_busses >= 0), 'Num of busses can\'t be negative'
        assert (num_sinks >= 0), 'Num of sinks can\'t be negative'
        assert (num_sources >= 0), 'Num of sources can\'t be negative'

        self.num_square_clusters = num_square_clusters
        self.num_busses = num_busses
        self.num_sinks = num_sinks
        self.num_sources = num_sources
        self.num_clusters = num_square_clusters + num_busses + num_sinks + num_sources
        self.mdl = MDL(self.n, self.num_clusters, alpha, beta)

        self.population_size = population_size
        self.offspring_size = offspring_size
        self.p_c = p_c
        self.p_m = p_m
        self.generation_limit = generation_limit
        self.generation_limit_without_improvement = generation_limit_without_improvement
        self.cluster_can_have_read_only_elements = cluster_can_have_read_only_elements
        self.cluster_can_have_partial_bus = cluster_can_have_partial_bus
        self.cluster_can_have_partial_sink = cluster_can_have_partial_sink
        self.cluster_can_have_partial_source = cluster_can_have_partial_source

        self.clusters: ClusterNames = []
        self.D_out: MatrixOut = []
        self.D_out_header: list[str] = []
        self.D_out_start_idx = 0
        self.CA: Matrix = []
        self.CA_header: list[str] = []

    def __read_DSM(self, file_name: str):
        """
        Read binary DSM matrix and set D and D' matrices, as well as
        row and column names. Binary DSM data format should be as follows:
        ,label1, label2, label3, label4
        label1,,1,,1
        label2,,,1,1
        label3,1,,,1
        0, 1, ., o, O, x, X, or blank could be used as matrix values

        Parameters
        ----------
        file_name :str
            File name containing input matrix
        """
        is_first_line = True
        self.column_names: list[str] = []
        self.row_names: list[str] = []
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
                    else:
                        self.D = []
                        self.D_dash = []
                        fd.close()
                        raise ValueError(
                            f"Invalid object {line_splitted[i]} in DSM")
                n += 1
            fd.close()

        # For a DSM, row and column names and order must match
        if len(self.row_names) != len(self.column_names):
            self.D = []
            self.D_dash = []
            raise ValueError(
                "No of rows and columns in DSM must match.")

        if self.row_names != self.column_names:
            self.D = []
            self.D_dash = []
            raise ValueError(
                "Row and column order in DSM must match.")
        print(f"Read DSM with {n} rows and {m-1} columns")

    def __all_pairs(self, given_list: list[int]) -> list[Chromosome]:
        """
        Given a list of items, generate list of all pairs of items

        Parameters
        ----------
        given_list : list[int]
            List of items

        Returns
        -------
        Chromosome
            List of all pairs of items
        """

        return [[i, j] for idx, i in enumerate(given_list) for j in given_list[idx + 1:]]

    def __build_D_dash(self, chromosome: Chromosome) -> int:
        """
        Build D' for the given chromosome

        Parameters
        ----------
        chromosome : Chromosome
            Chromosome to build the DSM

        Returns
        -------
        int
            Total no of objects in clusters
        """
        # TODO refactor code to reduce statements, variable, and branches
        # Reset D' matrix
        for i in range(self.n):
            for j in range(self.n):
                self.D_dash[i][j] = 0

        num_objects_in_all_clusters = 0
        # Mark one cluster at a time
        for i in range(self.num_clusters):
            objects_in_cluster = []
            for j in range(self.n):
                idx = i * self.n + j
                if chromosome[idx] == 1:
                    objects_in_cluster.append(j)
                    num_objects_in_all_clusters += 1

            # Handle square clusters
            if i < self.num_square_clusters:
                pairs = self.__all_pairs(objects_in_cluster)
                for p in pairs:
                    # Check for clusters with read-only objects/parties
                    if (not self.cluster_can_have_read_only_elements) or \
                            (self.D[p[0]][p[1]] == 1 and self.D[p[1]][p[0]] == 1):
                        self.D_dash[p[0]][p[1]] = 1
                        self.D_dash[p[1]][p[0]] = 1
                    else:
                        # To be read only, all objects in a clustered row must be zero
                        is_p_0_read_only = True
                        is_p_1_read_only = True
                        for j in objects_in_cluster:
                            if self.D[p[0]][j] == 1:  # If a row member is not zero
                                is_p_0_read_only = False
                                if not is_p_1_read_only:
                                    break
                            if self.D[p[1]][j] == 1:  # If a row member is not zero
                                is_p_1_read_only = False
                                if not is_p_0_read_only:
                                    break
                        if not is_p_0_read_only:
                            self.D_dash[p[0]][p[1]] = 1
                        if not is_p_1_read_only:
                            self.D_dash[p[1]][p[0]] = 1
            # Handle busses
            elif i < self.num_square_clusters + self.num_busses:
                # If a bus doesn't need to span across an entire row and column,
                # find the common set of objects in the bus and mark them
                if self.cluster_can_have_partial_bus:
                    all_nonempty_rows = []
                    all_nonempty_columns = []

                    for o in objects_in_cluster:
                        for j in range(self.n):
                            if self.D[j][o] == 1:
                                all_nonempty_rows.append(j)
                            if self.D[o][j] == 1:
                                all_nonempty_columns.append(j)

                    # Find intersection of rows and columns
                    intersecting_nonempty_rows_columns = [
                        r for r in all_nonempty_rows if r in all_nonempty_columns]

                    for rc in intersecting_nonempty_rows_columns:
                        for o in objects_in_cluster:
                            # Iterate over unique rows & columns
                            if o == rc:  # Skip diagonal
                                continue
                            self.D_dash[o][rc] = 1
                            self.D_dash[rc][o] = 1
                else:
                    for o in objects_in_cluster:
                        for j in range(self.n):
                            if o == j:  # Skip diagonal
                                continue
                            self.D_dash[o][j] = 1
                            self.D_dash[j][o] = 1
            # Handle sinks
            elif i < self.num_square_clusters + self.num_busses + self.num_sinks:
                # If a reader doesn't need to span across an entire column,
                # mark only objects in minimum bounding box
                if self.cluster_can_have_partial_sink:
                    all_nonempty_rows = []

                    for j in range(self.n):
                        all_ones = True
                        for o in objects_in_cluster:
                            if o == j:  # Skip diagonal
                                continue
                            if self.D[j][o] == 0:
                                all_ones = False
                        if all_ones:
                            all_nonempty_rows.append(j)

                    for r in all_nonempty_rows:  # Iterate over rows
                        for o in objects_in_cluster:
                            if o == r:  # Skip diagonal
                                continue
                            self.D_dash[r][o] = 1
                else:
                    for o in objects_in_cluster:
                        for j in range(self.n):
                            if o == j:  # Skip diagonal
                                continue
                            self.D_dash[j][o] = 1
            # Handle sources
            else:
                # If a writer doesn't need to span across an entire row,
                # mark only objects in minimum bounding box
                if self.cluster_can_have_partial_source:
                    all_nonempty_columns = []

                    for j in range(self.n):
                        all_ones = True
                        for o in objects_in_cluster:
                            if o == j:  # Skip diagonal
                                continue
                            if self.D[o][j] == 0:
                                all_ones = False
                        if all_ones:
                            all_nonempty_columns.append(j)
                    # for o in objects_in_cluster:
                    #     for j in range(self.n):
                    #         if self.D[o][j] == 1:
                    #             all_nonempty_columns.append(j)

                    for c in all_nonempty_columns:  # Iterate over columns
                        for o in objects_in_cluster:
                            if o == c:  # Skip diagonal
                                continue
                            self.D_dash[o][c] = 1
                else:
                    for o in objects_in_cluster:
                        for j in range(self.n):
                            if o == j:  # Skip diagonal
                                continue
                            self.D_dash[o][j] = 1

        return num_objects_in_all_clusters

    def __calculate_MDL(self, chromosome: Chromosome) -> Fitness:
        """
        Calculate MDL metric as per Eq. 3.

        Parameters
        ----------
        chromosome : Chromosome
            Chromosome to build the DSM

        Returns
        -------
        Fitness
            MDL metric value, no of type I errors, no of type II errors
        """
        # Build D' matrix
        sum_c_i = self.__build_D_dash(chromosome)

        # Calculate type 1 and 2 errors
        num_type_1_errors = 0
        num_type_2_errors = 0
        for i in range(self.n):
            for j in range(self.n):
                delta = self.D[i][j] - self.D_dash[i][j]
                if delta == 0:
                    continue

                if delta < 0:
                    num_type_1_errors += 1
                else:
                    num_type_2_errors += 1
        value = self.mdl.value(sum_c_i, num_type_1_errors, num_type_2_errors)
        return (value, num_type_1_errors, num_type_2_errors)

    def stats(self, clusters: list[list[str]]) -> Fitness:
        """
        Calculate MDL statistics of given chromosome

        Parameters
        ----------
        clusters: list[list[str]]
            Cluster membership

        Returns
        -------

        Fitness
        MDL metric value, no of type I errors, no of type II errors
        """
        chrom = ga.build_chromosome(
            clusters, self.column_names, self.num_clusters)
        return self.__calculate_MDL(chrom)

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
        chromosomes_with_fitness = []
        for c in chromosomes:
            chromosomes_with_fitness.append((c[0], self.__calculate_MDL(c[0])))
        return chromosomes_with_fitness

    def cluster(self):
        """
        Cluster the DSM using GA as per the steps given in Section 5 of paper
        """

        no_change_count = 0

        # Step 1 - Generate population and then calculate fitness value
        chromosome_len = self.n * self.num_clusters
        population = ga.generate_population(
            self.population_size, chromosome_len)
        population = self.__fitness(population)
        next_generation: Population = []

        for i in range(self.generation_limit):
            # Step 2 - Generate offsprings
            offsprings = ga.crossover(
                population, self.offspring_size, self.p_c)

            # Step 3 - Mutate offsprings
            offsprings = ga.mutate(offsprings, self.p_m)

            # Step 4 - Calculate fitness value for offsprings
            offsprings = self.__fitness(offsprings)

            # Step 5 - lambda + mu selection
            population_plus_offspring = population + offsprings
            population_plus_offspring.sort(key=itemgetter(1))
            next_generation = population_plus_offspring[:self.population_size]

            if ga.is_next_generation_same(population, next_generation, chromosome_len):
                no_change_count += 1
                if no_change_count == self.generation_limit_without_improvement:
                    print(
                        f'No more improvements. Stopped after {i + 1} iterations')
                    break
            else:
                no_change_count = 0

            population = next_generation

        self.__build_clustered_DSM(next_generation[0][0])
        self.__save_clustered_matrix(next_generation[0][1])

    def __build_clustered_DSM(self, chromosome: Chromosome):
        """
        Build clustered DSM

        Parameters
        ----------
        chromosome : Chromosome
            A chromosome with membership of all clusters
        """
        # Create the list of clusters while removing busses, sinks, and
        # sources. Also, count no of objects in a cluster and no of items an
        # object appear in clusters
        clusters = []
        busses = []
        sinks = []
        sources = []

        for i in range(self.num_clusters):
            cluster = chromosome[i * self.n: (i + 1) * self.n]
            num_cluster_members = 0
            if i < self.num_square_clusters:  # Square clusters
                for j in range(self.n):
                    if cluster[j] == 1:
                        num_cluster_members += 1
                if num_cluster_members > 0:  # Non-empty cluster
                    clusters.append((cluster, num_cluster_members))
            else:  # All others
                for j in range(self.n):
                    if cluster[j] == 1:  # Is a non-empty cluster
                        num_cluster_members += 1
                if num_cluster_members > 0:
                    if i < self.num_square_clusters + self.num_busses:   # Busses
                        busses.append((cluster, num_cluster_members))
                    elif i < self.num_square_clusters + self.num_busses + self.num_sinks:  # Sinks
                        sinks.append((cluster, num_cluster_members))
                    else:  # Sources
                        sources.append((cluster, num_cluster_members))

        # Cleanup square clusters and identify overlapping and non-overlapping
        # clusters
        [nonoverlapping_clusters,
            overlapping_clusters] = self.__cleanup_clusters(clusters)

        # Arrange D matrix using BEA with cluster information.
        # No relationship is lost
        self.__arrange_D_using_BEA(
            nonoverlapping_clusters, overlapping_clusters)

        # Build a more clear matrix with only clustered objects. A relationship
        # not in a cluster is lost
        self.__arrange_only_clustered_objects(
            nonoverlapping_clusters,
            overlapping_clusters,
            busses,
            sinks,
            sources)

    def __cleanup_clusters(self, clusters: Clusters) -> tuple[Clusters, Clusters]:
        """
        Cleanup square clusters by removing completely overlapping clusters.
        Then identify clusters that overlap and not. Sort each set by number
        of cluster members.

        Parameters
        ----------
        clusters : Clusters
            List of clusters

        Returns
        -------
        Tuple[Clusters, Clusters]
            A tuple with lists of nonoverlapping and overlapping clusters
        """
        clusters_cleaned = []
        overlapping_clusters = []
        nonoverlapping_clusters = []

        # Sort clusters by member size
        clusters.sort(key=itemgetter(1))

        # Remove completely overlapping clusters, i.e., subclusters by checking
        # whether one of the bigger clusters contain this cluster
        l = len(clusters)
        for i in range(l):  # Pick next cluster
            completely_overlap = False
            for j in range(i + 1, l):  # Check with all other clusters
                or_result = [0] * self.n
                for k in range(self.n):
                    if clusters[i][0][k] != clusters[j][0][k]:
                        or_result[k] = 1
                    else:
                        or_result[k] = clusters[i][0][k]
                # If a complete overlap, skip this cluster
                if or_result == clusters[j][0]:
                    completely_overlap = True
                    break

            # If not a complete subcluster or the last cluster, add to list
            if not completely_overlap:
                clusters_cleaned.append(clusters[i])

        # Count number of times an object appear in clusters
        overlap_count = [0] * self.n
        m = len(clusters_cleaned)
        for i in range(m):  # Pick next cluster
            for j in range(self.n):
                if clusters_cleaned[i][0][j] == 1:
                    overlap_count[j] += 1
                    if overlap_count[j] > 2:
                        print(
                            f'Object {self.column_names[j]} in more than 2 square clusters.')

        # Separate out overlapping clusters from others
        for i in range(m):  # Pick next cluster
            overlap = False
            for k in range(self.n):
                if clusters_cleaned[i][0][k] == 1 and overlap_count[k] > 1:
                    overlap = True
                    break
            if overlap:
                overlapping_clusters.append(clusters_cleaned[i])
            else:
                nonoverlapping_clusters.append(clusters_cleaned[i])

        # Sort each cluster set starting from the largest
        nonoverlapping_clusters.sort(key=itemgetter(1), reverse=True)
        overlapping_clusters.sort(key=itemgetter(1), reverse=True)

        return nonoverlapping_clusters, overlapping_clusters

    def __arrange_only_clustered_objects(self,
                                         nonoverlapping_clusters: Clusters,
                                         overlapping_clusters: Clusters,
                                         busses: Clusters,
                                         sinks: Clusters,
                                         sources: Clusters):
        """
        Build a more clear matrix with only clustered objects. A relationship
        not in a cluster is lost

        Parameters
        ----------
        nonoverlapping_clusters : Clusters
            List of nonoverlapping clusters
        overlapping_clusters : Clusters
            List of overlapping clusters
        busses : Clusters
            List of busses
        sinks : Clusters
            List of sinks
        sources : Clusters
            List of sources
        """
        matrix_size = 0
        for nonoverlapping_cluster in nonoverlapping_clusters:
            matrix_size += nonoverlapping_cluster[1]
        for overlapping_cluster in overlapping_clusters:
            matrix_size += overlapping_cluster[1]
        for bus in busses:
            matrix_size += bus[1]
        for sink in sinks:
            matrix_size += sink[1]
        for source in sources:
            matrix_size += source[1]

        self.D_out = [['' for i in range(matrix_size)]
                      for j in range(matrix_size)]

        # Populate matrix
        for nonoverlapping_cluster in nonoverlapping_clusters:
            self.__add_cluster_to_matrix(
                nonoverlapping_cluster[0], ClusterType.SQUARE)
        for overlapping_cluster in overlapping_clusters:
            self.__add_cluster_to_matrix(
                overlapping_cluster[0], ClusterType.SQUARE)
        for bus in busses:
            self.__add_cluster_to_matrix(bus[0], ClusterType.BUS)
        for sink in sinks:
            self.__add_cluster_to_matrix(sink[0], ClusterType.READER)
        for source in sources:
            self.__add_cluster_to_matrix(source[0], ClusterType.WRITER)

    def __add_cluster_to_matrix(self, cluster: Chromosome, cluster_type: ClusterType):
        """
        Add cluster to output matrix

        Parameters
        ----------
        cluster : Chromosome
            A partial chromosome with only the members of one cluster
        cluster_type : ClusterType
            Is the given cluster a square, bus, reader, or writer cluster?
        """
        # Find names of cluster members
        members = []
        for idx in range(self.n):
            if cluster[idx] == 1:  # Object in cluster
                members.append(
                    self.row_names[idx])  # Add symbolic name

        self.D_out_header += members
        self.clusters.append((cluster_type, members))

        end_idx = self.D_out_start_idx + len(members)
        for i in range(self.D_out_start_idx, end_idx):
            if cluster_type == ClusterType.SQUARE:
                for j in range(self.D_out_start_idx, end_idx):
                    if i == j:
                        self.D_out[i][j] = '.'
                    else:
                        self.D_out[i][j] = 'x'
            elif cluster_type in (ClusterType.BUS, ClusterType.READER, ClusterType.WRITER):
                for j in range(len(self.D_out)):
                    if i == j:
                        self.D_out[i][j] = '.'
                        continue

                    if cluster_type == ClusterType.BUS:
                        self.D_out[i][j] = 'x'
                        self.D_out[j][i] = 'x'
                    elif cluster_type == ClusterType.READER:
                        self.D_out[j][i] = 'x'
                    elif cluster_type == ClusterType.WRITER:
                        self.D_out[i][j] = 'x'
            else:
                raise TypeError("Unknown cluster type")
        self.D_out_start_idx = end_idx

    def __save_clustered_matrix(self, fitness: Fitness):
        """
        Save the cluster configuration and clustered DSM to a CSV file

        Parameters
        ----------
        Fitness
            Fitness score of the chromosome, No of Type I errors, No of Type II errors
        """
        square_clusters: MatrixOut = []

        with open(self.output_file_name, 'w+', encoding="utf-8") as fd:
            fd.write(f'Fitness score,{fitness[0]}\n')
            fd.write(f'Type I errors,{fitness[1]}\n')
            fd.write(f'Type II errors,{fitness[2]}\n')
            print(f'Fitness score:\t{fitness}')

            fd.write('\nClusters\n')
            for c in self.clusters:
                if c[0] == ClusterType.SQUARE:
                    fd.write('Cluster,' + ','.join(c[1]) + '\n')
                    square_clusters.append(c[1])
                elif c[0] == ClusterType.BUS:
                    fd.write('Bus,' + ','.join(c[1]) + '\n')
                elif c[0] == ClusterType.READER:
                    fd.write('Sink,' + ','.join(c[1]) + '\n')
                elif c[0] == ClusterType.WRITER:
                    fd.write('Source,' + ','.join(c[1]) + '\n')

            fd.write('\n\nClustered DSM with all objects arranged using BEA\n')
            fd.write(',' + ','.join(self.CA_header) + '\n')
            # for i in range(len(self.CA)):
            for i, ca in enumerate(self.CA):
                fd.write(self.CA_header[i] + ',' + ','.join(ca) + '\n')

            fd.write('\nClustered DSM with only objects in clusters\n')
            fd.write(',' + ','.join(self.D_out_header) + '\n')

            # for i in range(len(self.D_out)):
            for i, out in enumerate(self.D_out):
                fd.write(self.D_out_header[i] + ',' +
                         ','.join(out) + '\n')

            fd.close()

        plot_file_name = self.output_file_name[:-3] + 'png'
        plot_overlapping_clusters(
            self.D, self.n, self.column_names, square_clusters, plot_file_name)

    def __arrange_D_using_BEA(self, nonoverlapping_clusters: Clusters,
                              overlapping_clusters: Clusters):
        """
        Arrange D matrix using Bond Energy Algorithm (BEA) with cluster details
        Then prepare it to be saved as a DSM.

        Parameters
        ----------
        nonoverlapping_clusters : Clusters
            List of nonoverlapping clusters
        overlapping_clusters : Clusters
            List of overlapping clusters
        """

        # Replace cluster chromosomes with their object indexes
        clusters = []
        # Nonoverlapping clusters
        for chrome in nonoverlapping_clusters:
            members = []
            for k in range(self.n):
                if chrome[0][k] == 1:
                    members.append(k)
            clusters.append(members)

        # Overlapping clusters
        for chrome in overlapping_clusters:
            members = []
            for k in range(self.n):
                if chrome[0][k] == 1:
                    members.append(k)
            clusters.append(members)

        # Further arrange using BEA
        AA = self.__D_2_AA()
        bea = BEA(AA, self.column_names, clusters)
        bea.cluster()

        # Make resulting cluster ready to dump to a CSV. CA is in column order
        for i in range(self.n):
            row: list = []
            for j in range(self.n):
                if i == j:  # Clean up diagonal
                    row.append('.')
                elif bea.CA[j][i] == 0:
                    row.append('')
                else:
                    row.append('x')
            self.CA.append(row)
        self.CA_header = bea.column_names  # Map column index to names

    def __D_2_AA(self) -> Matrix:
        """
        Transform D to attribute affinity (AA) matrix, which should be in column
        order (D is in row order). Also, as AA assumes objects have affinity
        with themselves, set diagonal cells to 1. This helps square clusters
        to stay together as more adjacent columns have 1s.

        Returns
        ----------
        D_column : Matrix
            D in column order
        """
        D_column: Matrix = []
        for i in range(self.n):
            column = []
            for j in range(self.n):
                if i != j:
                    column.append(self.D[j][i])
                else:
                    column.append(1)
            D_column.append(column)
        return D_column
