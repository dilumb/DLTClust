"""Module providing a set of utility functions."""
import configparser
from src.datatypes import Config, Chromosome
from src.dsm import DSM


def str_2_bool(value: str):
    """
    Convert string to boolean

    Parameters
    ----------
    value : str
        Value to convert

    Returns
    -------
    bool
        Boolean value
    """

    if value.lower() == 'true':
        return True

    if value.lower() == 'false':
        return False

    raise ValueError(f'Incorrect boolean value {0}'.format(value))


def build_config(file_name: str):
    """
    Build the configuration object based on values in configuration file

    Parameters
    ----------
    file_name : str
        Name of configuration file

    Returns
    -------
    Config
        Configuration object
    """

    cfg = configparser.ConfigParser()
    cfg.read(file_name)
    ga_cfg = cfg['GA']

    return Config(
        alpha=float(ga_cfg['alpha']),
        beta=float(ga_cfg['beta']),
        population_size=int(ga_cfg['population_size']),
        offspring_size=int(ga_cfg['offspring_size']),
        p_c=float(ga_cfg['p_c']),
        p_m=float(ga_cfg['p_m']),
        num_iterations_max=int(ga_cfg['generation_limit']),
        num_iterations_without_improvement=int(
            ga_cfg['generation_limit_without_improvement']),
        cluster_can_have_read_only_elements=str_2_bool(
            ga_cfg['cluster_can_have_read_only_elements']),
        cluster_can_have_partial_bus=str_2_bool(
            ga_cfg['cluster_can_have_partial_bus']),
        cluster_can_have_partial_sink=str_2_bool(
            ga_cfg['cluster_can_have_partial_sink']),
        cluster_can_have_partial_source=str_2_bool(
            ga_cfg['cluster_can_have_partial_source']))


def dsm_to_graph(dsm: DSM, file_name: str):
    """
    Generate Neo4j Cypher query to represent DSM as a graph
    """
    with open(file_name, 'w', encoding="utf-8") as file:
        file.write('CREATE\n')
        for n in dsm.column_names:
            file.write(f"({n}:Party {{name: '{n}'}}),\n")
        for i in range(dsm.n):
            row_sum = 1  # Self loops
            for j in range(dsm.n):
                if dsm.D[i][j] == 1:
                    row_sum += 1
            for j in range(dsm.n):
                if dsm.D[i][j] == 1:
                    file.write(
                        f"({dsm.column_names[i]})-[:WRITE {{weight: {1/row_sum}}}]->({dsm.column_names[j]}),\n")
            file.write(
                f"({dsm.column_names[i]})-[:WRITE {{weight: {1/row_sum}}}]->({dsm.column_names[i]}),\n")
        file.write(';\n')


def build_chromosome(clusters: list[list[str]], names: list[str], num_clusters: int) -> Chromosome:
    """
    Generate a chromosome to reflect the given cluster membership
    """
    n = len(names)
    chromosome = [0 for i in range(n*num_clusters)]
    row_idx = 0
    for c in clusters:
        for m in c:
            try:
                column_idx = names.index(m)
                chromosome[row_idx*n + column_idx] = 1
            except ValueError:
                print("That item does not exist")
        row_idx += 1
    return chromosome
