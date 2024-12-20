"""Module providing a set of utility functions."""
import configparser
from cdlib import viz, NodeClustering
import networkx as nx
import matplotlib.pyplot as plt
from src.datatypes import Config, Matrix, MatrixOut


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


def dsm_to_graph(dsm: Matrix, file_name: str):
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
                        # f"({dsm.column_names[i]})-[:WRITE {{weight: {1/row_sum}}}]->({dsm.column_names[j]}),\n")
                        f"('{dsm.column_names[i]}','{dsm.column_names[j]}', {{'weight': {1/row_sum}}}),\n")

            file.write(
                # f"({dsm.column_names[i]})-[:WRITE {{weight: {1/row_sum}}}]->({dsm.column_names[i]}),\n")
                f"('{dsm.column_names[i]}','{dsm.column_names[i]}', {{'weight': {1/row_sum}}}),\n")
        file.write(';\n')


def plot_overlapping_clusters(dsm: Matrix, n: int, nodes: list[str], square_clusters: MatrixOut, file_name: str):
    """
    Plot square cluster membership on a graph 
    """
    edges: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(n):
            if dsm[i][j] == 1:
                edges.append((nodes[i], nodes[j]))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    communities = NodeClustering(
        square_clusters, G, method_name="DLTClust", overlap=True)

    position = nx.kamada_kawai_layout(G, weight='weight', scale=1)
    viz.plot_network_highlighted_clusters(
        G, communities, position, node_size=275, plot_labels=True, edge_weights_intracluster=1, cmap='viridis')
    plt.savefig(file_name)
