"""
@author: Dilum Bandara
"""

import random
import argparse
from datetime import datetime
from pathlib import Path
from src.dsm import DSM
from src.dmm import DMM
from src.util import build_config, dsm_to_graph

if __name__ == '__main__':
    startTime = datetime.now()
    MOD_PATH = str(Path(__file__).parent)

    parser = argparse.ArgumentParser(description='Cluster DSM and DMM.')
    parser.add_argument('-b', '--busses', type=int,
                        help='No of busses', default=0)
    parser.add_argument('-c', '--clusters', type=int, help='No of clusters')
    parser.add_argument('-i', '--input', type=str,
                        help='Input matrix file', default=MOD_PATH + '/dsm.csv')
    parser.add_argument('-o', '--output', type=str,
                        help='Output matrix file', default=MOD_PATH + '/clusters.csv')
    parser.add_argument('-p', '--params', type=str,
                        help='Config file with parameters', default=MOD_PATH + '/config.ini')
    parser.add_argument('-s', '--sinks', type=int,
                        help='No of sinks/readers', default=0)
    parser.add_argument('-r', '--rnd', type=int,
                        help='Random seed', default=123)
    parser.add_argument('-t', '--type', type=str,
                        help='Type DSM, DSM2Graph or DMM', default='DSM')
    parser.add_argument('-u', '--sources', type=int,
                        help='No of sources/writers', default=0)
    args = parser.parse_args()

    assert args.clusters, 'At least number of square clusters must be specified.'

    # Set seed
    random.seed(args.rnd)

    # Load configuration parameters
    config = build_config(args.params)

    # DSM clustering
    if args.type == 'DSM':
        dsm = DSM(args.input,
                  args.output,
                  args.clusters,
                  args.busses,
                  args.sinks,
                  args.sources,
                  config.alpha,
                  config.beta,
                  config.population_size,
                  config.offspring_size,
                  config.p_c,
                  config.p_m,
                  config.num_iterations_max,
                  config.num_iterations_without_improvement,
                  config.cluster_can_have_read_only_elements,
                  config.cluster_can_have_partial_bus,
                  config.cluster_can_have_partial_sink,
                  config.cluster_can_have_partial_source)
        dsm.cluster()

    # DSM to graph
    elif args.type == 'DSM2Graph':
        dsm = DSM(args.input,
                  args.output,
                  args.clusters,
                  args.busses,
                  args.sinks,
                  args.sources,
                  config.alpha,
                  config.beta,
                  config.population_size,
                  config.offspring_size,
                  config.p_c,
                  config.p_m,
                  config.num_iterations_max,
                  config.num_iterations_without_improvement,
                  config.cluster_can_have_read_only_elements,
                  config.cluster_can_have_partial_bus,
                  config.cluster_can_have_partial_sink,
                  config.cluster_can_have_partial_source)
        dsm_to_graph(dsm, './dsm_to_graph.txt')

    # DMM clustering
    elif args.type == 'DMM':
        dmm = DMM(args.input,
                  args.output,
                  args.clusters,
                  config.alpha,
                  config.beta,
                  config.population_size,
                  config.offspring_size,
                  config.p_c,
                  config.p_m,
                  config.num_iterations_max,
                  config.num_iterations_without_improvement)
        dmm.cluster()
    else:
        raise ValueError(
            'Unsupported matrix type {args.type}. Only DSM or DMM is supported')

    endTime = datetime.now()
    print("Execution time: ", (endTime-startTime))
