from re import M
from community.community_louvain import best_partition
from networkx.algorithms.centrality.degree_alg import degree_centrality
from networkx.algorithms.centrality import eigenvector_centrality, eigenvector_centrality_numpy
from networkx.algorithms.centrality.katz import katz_centrality
from networkx.algorithms.link_analysis.hits_alg import hits
from numpy.core.fromnumeric import size
import scipy as sp
from scipy.sparse import data
from scipy.sparse.linalg import eigsh, eigs
import numpy as np
from scipy import sparse
import networkx as nx
import collections
from networkx.readwrite.edgelist import parse_edgelist 
from networkx.readwrite.gml import read_gml, parse_gml
from igraph import Graph
import igraph as ig
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from networkx.algorithms.community.label_propagation import asyn_lpa_communities, label_propagation_communities
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import community
from networkx.classes.function import is_directed
import gcn
import json
from networkx.generators.community import LFR_benchmark_graph

from networkx.classes.function import is_directed
from networkx.algorithms.components import is_connected



def q1_load_enron_dataset():

    edgelist_og = np.loadtxt("./data/email-Enron/email-Enron.txt", dtype=np.int64)

    # Remove the timestamp column
    edgelist = np.delete(edgelist_og, 2, axis=1)

    # Convert to a list for usage with Counter
    edgelist = [tuple(line) for line in edgelist]

    # Count the number of occurences of each edge
    counts = collections.Counter(edgelist)

    # Convert the count of occurences as the weight of each edge
    edgelist_weighted = []
    for k,v in counts.items():
        line = f"{k[0]} {k[1]} {v}"
        edgelist_weighted.append(line)

    return parse_edgelist(edgelist_weighted, nodetype=int, create_using=nx.DiGraph, data=(("weight", int),))


def q1_centrality_measures(G):
    print(f"\nTop 5 nodes with highest centrality according to different measures:\n")
    deg_centralities = degree_centrality(G)
    degrees = dict(G.degree())

    print("Degree Centralities:")
    print(sorted(deg_centralities.items(), key=lambda item: item[1], reverse=True)[0:5])
    print("Degrees:") 
    print(sorted(degrees.items(), key=lambda item: item[1], reverse=True)[0:5])


    eigen_centrality = eigenvector_centrality(G)
    eigen_centrality_np = eigenvector_centrality_numpy(G)

    print("\nEigenvector centrality:") 
    print(sorted(eigen_centrality.items(), key=lambda item: item[1], reverse=True)[0:5])
    print("Eigenvector centrality (numpy):")
    print(sorted(eigen_centrality_np.items(), key=lambda item: item[1], reverse=True)[0:5])

    M = nx.to_scipy_sparse_matrix(G, nodelist=list(G), weight="weight", dtype=float)
    eigenvalue, _ = eigs(M.T, k=1, which="LR", maxiter=1000)

    print(f"\nLargest eigenvalue: {eigenvalue.real}")

    katz_c = katz_centrality(G, alpha=float(1/eigenvalue.real))
    print("Katz Centrality:")
    print(sorted(katz_c.items(), key=lambda item: item[1], reverse=True)[0:5])

    hubs, authorities = hits(G)
    print("\nHubs:")
    print(sorted(hubs.items(), key=lambda item: item[1], reverse=True)[0:5])
    print("Authorities:")
    print(sorted(authorities.items(), key=lambda item: item[1], reverse=True)[0:5])


"""
    Convert the clustering returned by networkx functions to an array 
    with the cluster id of node i at index i
"""
def get_labels_pred(G, c):

    labels_pred = [0] * len(G)

    for cluster_id in range(0, len(c)):
        nodes = c[cluster_id]
        for n in nodes:
            labels_pred[n] = cluster_id + 1

    return labels_pred


def q2_get_mean_scores(scores):
    mean_scores = {}

    # Calculate the means for each algo
    for algo, measure_score_dict in scores.items():
        mean_scores[algo] = {}

        for measure, all_scores in measure_score_dict.items():
            mean_scores[algo][measure] = np.mean(all_scores)

    return mean_scores



def format_LFR_graph_communities(LFR):
    # Extract the unique communities
    communities = set({})
    for node, comm in LFR.nodes.data('community'):
        communities.add(frozenset(comm))

    communities = list(communities)

    n_communities = len(communities)
    # print(f"{communities}")

    # Set labels according to which community a node is in
    for i in range(len(LFR)):
        node = LFR.nodes[i]
        node["value"] = communities.index(node['community'])
        # Remove the community attribute that is no longer needed
        del LFR.nodes[i]['community']

    return LFR, n_communities


def score_clustering(G, c, algo_used, labels_true, scores):

    if not algo_used in scores:
        scores[algo_used] = {"Q": [], "NMI": [], "ARI": []}

    labels_pred = get_labels_pred(G, c)

    Q = modularity(G, c)
    NMI = normalized_mutual_info_score(labels_true, labels_pred)
    ARI = adjusted_rand_score(labels_true, labels_pred)

    scores[algo_used]["Q"].append(Q)
    scores[algo_used]["NMI"].append(NMI)
    scores[algo_used]["ARI"].append(ARI)

    print(f"\t\tQ = {Q}")
    print(f"\t\tNMI = {NMI}")
    print(f"\t\tARI = {ARI}")

    return scores


def q2_test_girvan_newman(G, scores, labels_true, k=500):
    t1 = time.time()

    comp = girvan_newman_k_samples(G, k)

    # Iterate over the genator given and stop when it returns once Q starts dropping
    i = 1
    q_max = 0
    t1 = time.time()

    for comm in comp:   
        print(f"GN iteration {i}")

        c = tuple(sorted(x) for x in comm)

        Q = modularity(G, c)

        # Stop iterating once the modularity starts dropping
        if Q > q_max:
            print(f"better Q reached: {Q} vs {q_max} previously")
            q_max = Q
        else:
            print(f"Modularity is starting to drop, keep the last communities")
            break
        
        t2 = time.time()
        print(f"Iteration {i} took {t2-t1}s")
        t1 = time.time()
        i += 1

    print(f"\tGirvan-Newman modularity with {len(c)} communities.")

    scores = score_clustering(G, c, "girvan", labels_true, scores)

    t2 = time.time()
    print(f"Girvan Newman took {t2-t1}s")

    return scores


def q2_test_greedy_modularity(G, scores, labels_true):
    t1 = time.time()

    # Clauset et al. 2004 "Fast Modularity"
    c = greedy_modularity_communities(G)
    print(f"\tGreedy modularity with {len(c)} communities.")

    scores = score_clustering(G, c, "greedy", labels_true, scores)

    t2 = time.time()
    print(f"Greedy Modularity took {t2-t1}s")

    return scores


def q2_test_lpa(G, scores, labels_true):
    t1 = time.time()

    c = list(asyn_lpa_communities(G, seed=3))
    print(f"\tLPA modularity with {len(c)} communities")

    scores = score_clustering(G, c, "lpa", labels_true, scores)

    t2 = time.time()
    print(f"LPA took {t2-t1}s")

    return scores


def q2_test_louvain(G, scores, labels_true):
    t1 = time.time()

    # Convert the graph to an undirected graph to be able to use the Louvain algo from 
    # https://python-louvain.readthedocs.io/en/latest/
    if is_directed(G):
        G = G.to_undirected()

    try:
        best_p = community.best_partition(G)
    except Exception as e:
        print(e)
        best_p = {}

    c_dict = {}

    # Convert the dictionary returned by best_partition to a format modularity will accept:
    # It is given in a dictionary with {nodeid: clusterid}
    for node_id, clust_id in best_p.items():
        if clust_id in c_dict:
            c_dict[clust_id].add(node_id)
        else:
            c_dict[clust_id] = set({node_id})
    
    c = [s for s in c_dict.values()]

    print(f"\tLouvain modularity with {len(c)} communities")

    scores = score_clustering(G, c, "louvain", labels_true, scores)

    t2 = time.time()
    print(f"Louvain took {t2-t1}s")

    return scores



def q2_test_fluidc(G, scores, labels_true, n_communities):
    ## Algorithm 4
    ## Only applicable to connected graphs
    ## and is_connected() function is only available for undirected graphs
    t1 = time.time()

    if is_directed(G):
        G = G.to_undirected()

    if is_connected(G):

        if not "fluidc" in scores:
            scores["fluidc"] = {"Q": [], "NMI": [], "ARI": []}

        c = list(asyn_fluidc(G, k=n_communities))

        print(f"\tAsync Fluid Communities with {len(c)} communities")

        scores = score_clustering(G, c, "fluidc", labels_true, scores)

    t2 = time.time()
    print(f"Fluidc took {t2-t1}s")

    return scores


def q2_test_all_algos(G, scores, labels_true, n_communities, girvan=False, only_girvan=False):

    # Girvan Newman is optional because it is the slowest
    if girvan:
        scores = q2_test_girvan_newman(G, scores, labels_true)

    if not only_girvan:
        scores = q2_test_greedy_modularity(G, scores, labels_true)
        scores = q2_test_lpa(G, scores, labels_true)
        scores = q2_test_louvain(G, scores, labels_true)
        scores = q2_test_fluidc(G, scores, labels_true, n_communities)

    return scores


def q2_calculate_clustering_scores(datasets, dataset_type, girvan=False, only_girvan=False):

    scores = {}

    for dataset in datasets:
        t1 = time.time()

        print(f"\nDataset {dataset}")

        if dataset_type == 'real-classic':
            G = Graph.Read_GML(f"./data/real-classic/{dataset}.gml")
            G = G.to_networkx()

            # Polbooks dataset has string labels which throws a ValueError when we 
            # try the first line. We convert it to int using ord().
            try:
                labels_true = [ int(G.nodes[node]['value']) for node in G.nodes ]
            except ValueError:
                labels_true = [ ord(G.nodes[node]['value']) for node in G.nodes ]

            n_communities = len(np.unique(labels_true))

        elif dataset_type == 'real-node-label':
            G, labels_true, n_communities = gcn.load_data(dataset)

        print(f"Real number of communities: {n_communities}")
        
        q2_test_all_algos(G, scores, labels_true, n_communities, girvan, only_girvan)

        t2 = time.time()
        print(f"{dataset} took {t2-t1}s")

    return scores


def q2_3_LFR_graph_communities(n, µ, tau1, tau2, avg_d, min_c, n_iter=10, girvan=False, only_girvan=False):
    t1_total = time.time()

    scores = {} 

    print(f"Generating LFR models with µ={µ}")

    for _ in range(n_iter):
        t1 = time.time()
        try:
            G = LFR_benchmark_graph(n, tau1, tau2, µ, average_degree = avg_d, min_community = min_c)
            G, n_communities = format_LFR_graph_communities(G)
        except Exception:
            continue

        print(f"For LFR graph with mu={µ}, iteration {_}")
        print(f"N={len(G.nodes)}, E={len(G.edges)}")

        labels_true = [ int(G.nodes[node]['value']) for node in G.nodes ]

        print(f"\tTrue number of communities: {n_communities}")

        scores = q2_test_all_algos(G, scores, labels_true, n_communities, girvan, only_girvan)

        t2 = time.time()
        print(f"LFR with µ={µ}, iteration {_} took {t2-t1}s")

    t2_total= time.time()
    print(f"LFR with µ={µ} all iterations took {t2_total-t1_total}s")

    return scores



def q2_run_real_classic(girvan=False, only_girvan=False):
    datasets = ["karate", "football", "polblogs", "polbooks", "strike"]

    scores_classic = q2_calculate_clustering_scores(datasets, 'real-classic', girvan, only_girvan)
    means_classic = q2_get_mean_scores(scores_classic)

    with open(f"./results/real_classic_scores_{int(time.time())}", "w") as out:
        out.write(json.dumps(scores_classic))
        out.write("\n")
        out.write(json.dumps(means_classic))

    print(means_classic)


def q2_run_real_node_labels(girvan=False, only_girvan=False):

    datasets = ["citeseer", "cora", "pubmed"]

    scores_label = q2_calculate_clustering_scores(datasets, 'real-node-label', girvan, only_girvan)
    means_label = q2_get_mean_scores(scores_label)

    with open(f"./results/real_label_scores_{int(time.time())}", "w") as out:
        out.write(json.dumps(scores_label))
        out.write("\n")
        out.write(json.dumps(means_label))


def q2_run_LFR(n_iter=10, girvan=False, only_girvan=False):
    n = 1000
    µ = .5
    tau1 = 3
    tau2 = 1.5
    avg_d = 5
    min_c = 20

    scores_LFR = q2_3_LFR_graph_communities(n, µ, tau1, tau2, avg_d, min_c, n_iter=n_iter, girvan=girvan, only_girvan=only_girvan)
    means_LFR = q2_get_mean_scores(scores_LFR)

    with open(f"./results/LFR_scores_{int(time.time())}", "w") as out:
        out.write(json.dumps(scores_LFR))
        out.write("\n")
        out.write(json.dumps(means_LFR))


# Adapted from networkx source
def _without_most_central_edges(G, most_valuable_edge):
    """Returns the connected components of the graph that results from
    repeatedly removing the most "valuable" edge in the graph.

    `G` must be a non-empty graph. This function modifies the graph `G`
    in-place; that is, it removes edges on the graph `G`.

    `most_valuable_edge` is a function that takes the graph `G` as input
    (or a subgraph with one or more edges of `G` removed) and returns an
    edge. That edge will be removed and this process will be repeated
    until the number of connected components in the graph increases.

    """
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)
    return new_components


# Adapted from networkx source
def girvan_newman_k_samples(G, k, most_valuable_edge=None):
    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return
    # If no function is provided for computing the most valuable edge,
    # use the edge betweenness centrality.
    if most_valuable_edge is None:

        def most_valuable_edge(G):
            """Returns the edge with the highest betweenness centrality
            in the graph `G`.

            """
            # We have guaranteed that the graph is non-empty, so this
            # dictionary will never be empty.
            betweenness = nx.edge_betweenness_centrality(G, k)
            return max(betweenness, key=betweenness.get)

    # The copy of G here must include the edge weight data.
    g = G.copy().to_undirected()
    # Self-loops must be removed because their removal has no effect on
    # the connected components of the graph.
    g.remove_edges_from(nx.selfloop_edges(g))
    while g.number_of_edges() > 0:
        yield _without_most_central_edges(g, most_valuable_edge)


def q2_clustering_algorimths(girvan, girvan_only, LFR_iter):
    q2_run_real_classic(girvan, girvan_only)
    q2_run_LFR(LFR_iter, girvan, girvan_only)
    q2_run_real_node_labels(girvan, girvan_only)
    

if __name__=='__main__':
    t1 = time.time()

    G = q1_load_enron_dataset()
    q1_centrality_measures(G)

    n_iter = 10
    girvan = False
    only_girvan = False

    q2_clustering_algorimths(girvan, only_girvan, n_iter)


    t2 = time.time()
    print(f"Total time: {t2-t1}s")








