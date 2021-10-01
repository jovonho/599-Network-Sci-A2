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
from networkx.algorithms.community import greedy_modularity_communities, naive_greedy_modularity_communities
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



def q1_load_enron_dataset(save_graph=False):

    edgelist_og = np.loadtxt("./data/email-Enron/email-Enron.txt", dtype=np.int64)

    # Remove the timestamp column
    edgelist = np.delete(edgelist_og, 2, axis=1)

    # edgelist = [(int(line[0]), int(line[1])) for line in edgelist]
    edgelist = [tuple(line) for line in edgelist]

    counts = collections.Counter(edgelist)

    # edgelist_weighted = np.ndarray(shape=(0,3), dtype=np.int64)
    edgelist_weighted = []

    if save_graph:
        with open("./data/enron-weighted.edges.txt", "w") as out:
            for k,v in counts.items():
                line = f"{k[0]} {k[1]} {v}"
                out.write(f"{k[0]}\t{k[1]}\t{v}\n")
                edgelist_weighted.append(line)


    return parse_edgelist(edgelist_weighted, nodetype=int, create_using=nx.DiGraph, data=(("weight", int),))


def q1(G):
    deg_centralities = degree_centrality(G)
    degrees = dict(G.degree())

    print("Degree Centralities:")
    print(sorted(deg_centralities.items(), key=lambda item: item[1], reverse=True)[0:10])
    print("Degrees:") 
    print(sorted(degrees.items(), key=lambda item: item[1], reverse=True)[0:10])


    eigen_centrality = eigenvector_centrality(G)
    eigen_centrality_np = eigenvector_centrality_numpy(G)

    print("Eigenvector centrality:") 
    print(sorted(eigen_centrality.items(), key=lambda item: item[1], reverse=True)[0:10])
    print("Eigenvector centrality (numpy):")
    print(sorted(eigen_centrality_np.items(), key=lambda item: item[1], reverse=True)[0:10])


    M = nx.to_scipy_sparse_matrix(G, nodelist=list(G), weight="weight", dtype=float)
    eigenvalue, _ = eigs(M.T, k=1, which="LR", maxiter=1000)

    print(eigenvalue)

    katz_c = katz_centrality(G, alpha=float(1/eigenvalue.real))
    print("Katz Centrality:")
    print(sorted(katz_c.items(), key=lambda item: item[1], reverse=True)[0:10])

    hubs, authorities = hits(G)
    print("Hubs:")
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


def q2_test_girvan_newman(G, scores, labels_true):
    t1 = time.time()

    if not "girvan" in scores:
        scores["girvan"] = {"Q": [], "NMI": [], "ARI": []}

    comp = girvan_newman(G)

    # Iterate over the genator given and stop when it returns once Q starts dropping
    i = 1
    q_max = 0
    for comm in comp:   
        print(f"GN iteration {i}")

        c = tuple(sorted(x) for x in comm)

        labels_pred = get_labels_pred(G, c)
        Q = modularity(G, c)

        # Stop iterating once the modularity starts dropping
        if Q > q_max:
            print(f"better Q reached: {Q} vs {q_max} previsouly")
            q_max = Q
        else:
            print(f"Modularity is starting to drop, keep the last communities")
            break
        
        i += 1

    print(f"\tGirvan-Newman modularity with {len(c)} communities.")

    labels_pred = get_labels_pred(G, c)

    Q = modularity(G, c)
    NMI = normalized_mutual_info_score(labels_true, labels_pred)
    ARI = adjusted_rand_score(labels_true, labels_pred)

    scores["girvan"]["Q"].append(Q)
    scores["girvan"]["NMI"].append(NMI)
    scores["girvan"]["ARI"].append(ARI)

    print(f"\t\tQ = {Q}")
    print(f"\t\tNMI = {NMI}")
    print(f"\t\tARI = {ARI}")

    t2 = time.time()
    print(f"Girvan Newman took {t2-t1}s")

    return scores


def q2_test_greedy_modularity(G, scores, labels_true):
    t1 = time.time()

    if not "greedy" in scores:
        scores["greedy"] = {"Q": [], "NMI": [], "ARI": []}

    # Clauset et al. 2004 "Fast Modularity"
    c = greedy_modularity_communities(G)
    print(f"\tGreedy modularity with {len(c)} communities.")

    labels_pred = get_labels_pred(G, c)

    # print(labels_true)

    Q = modularity(G, c)
    NMI = normalized_mutual_info_score(labels_true, labels_pred)
    ARI = adjusted_rand_score(labels_true, labels_pred)

    scores["greedy"]["Q"].append(Q)
    scores["greedy"]["NMI"].append(NMI)
    scores["greedy"]["ARI"].append(ARI)

    print(f"\t\tQ = {Q}")
    print(f"\t\tNMI = {NMI}")
    print(f"\t\tARI = {ARI}")

    t2 = time.time()
    print(f"Greedy Modularity took {t2-t1}s")

    return scores

def q2_test_lpa(G, scores, labels_true):

    t1 = time.time()

    if not "lpa" in scores:
        scores["lpa"] = {"Q": [], "NMI": [], "ARI": []}

    c = list(asyn_lpa_communities(G, seed=3))
    print(f"\tLPA modularity with {len(c)} communities")

    labels_pred = get_labels_pred(G, c)

    Q = modularity(G, c)
    NMI = normalized_mutual_info_score(labels_true, labels_pred)
    ARI = adjusted_rand_score(labels_true, labels_pred)
    
    scores["lpa"]["Q"].append(Q)
    scores["lpa"]["NMI"].append(NMI)
    scores["lpa"]["ARI"].append(ARI)
    
    print(f"\t\tQ = {Q}")
    print(f"\t\tNMI = {NMI}")
    print(f"\t\tARI = {ARI}")

    t2 = time.time()
    print(f"LPA took {t2-t1}s")

    return scores


def q2_test_louvain(G, scores, labels_true):
    t1 = time.time()

    if not "louvain" in scores:
        scores["louvain"] = {"Q": [], "NMI": [], "ARI": []}

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

    labels_pred = get_labels_pred(G, c)

    Q = modularity(G, c)
    NMI = normalized_mutual_info_score(labels_true, labels_pred)
    ARI = adjusted_rand_score(labels_true, labels_pred)
    
    scores["louvain"]["Q"].append(Q)
    scores["louvain"]["NMI"].append(NMI)
    scores["louvain"]["ARI"].append(ARI)
    
    print(f"\t\tQ = {Q}")
    print(f"\t\tNMI = {NMI}")
    print(f"\t\tARI = {ARI}")

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

        labels_pred = get_labels_pred(G, c)

        Q = modularity(G, c)
        NMI = normalized_mutual_info_score(labels_true, labels_pred)
        ARI = adjusted_rand_score(labels_true, labels_pred)
        
        scores["fluidc"]["Q"].append(Q)
        scores["fluidc"]["NMI"].append(NMI)
        scores["fluidc"]["ARI"].append(ARI)
        
        print(f"\t\tQ = {Q}")
        print(f"\t\tNMI = {NMI}")
        print(f"\t\tARI = {ARI}")

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
        
        q2_test_all_algos(G, scores, labels_true, n_communities, girvan)

        t2 = time.time()
        print(f"{dataset} took {t2-t1}s")

    return scores



"""
    The common practice is to sample for varying values of µ which controls how well separated
    are the communities, i.e. generating synthetic graphs with µ = .1 to µ = .9, reporting average
    performance for 10 realizations at each difficulty level, see https://arxiv.org/abs/0805.4770,
    Fig 5 for example. N = 1000, or 5000 are common settings. For this experiments, you can use
    µ = .5, n=1000, tau1 = 3, tau2 = 1.5, average degree=5, min community=20.
"""
def q2_3_LFR_graph_communities_iterate(n, tau1, tau2, avg_d, min_c, girvan=False, only_grivan=False):

    # algos = ["girvan", "greedy", "lpa", "louvain", "fluidc"]
    # scores = {algo: {"Q": [], "NMI": [], "ARI": []} for algo in algos}
    scores = {}

    for mu in np.linspace(0.1, 1, 10):
        print(f"Generating LFR models with mu={mu}")

        for _ in range(10):
            try:
                G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree = avg_d, min_community = min_c)
                G, n_communities = format_LFR_graph_communities(G)
            except Exception:
                continue

            print(f"For LFR graph with mu={mu}, iteration {_}")
            print(f"N={len(G.nodes)}, E={len(G.edges)}")

            labels_true = [ int(G.nodes[node]['value']) for node in G.nodes ]

            print(f"\tTrue number of communities: {n_communities}")

            scores = q2_test_all_algos(G, scores, labels_true, n_communities, girvan, only_grivan)
           
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

    scores_LFR = q2_3_LFR_graph_communities(n, µ, tau1, tau2, avg_d, min_c, n_iter=10, girvan=girvan, only_girvan=only_girvan)
    means_LFR = q2_get_mean_scores(scores_LFR)


    with open(f"./results/LFR_scores_{int(time.time())}", "w") as out:
        out.write(json.dumps(scores_LFR))
        out.write("\n")
        out.write(json.dumps(means_LFR))


if __name__=='__main__':
    t1 = time.time()

    # G = q1_load_enron_dataset()
    # Print out the different centrality measures
    # q1(G)

    n_iter = 10
    girvan = False
    only_girvan = False

    q2_run_real_classic(girvan, only_girvan)
    q2_run_real_node_labels(girvan, only_girvan)
    q2_run_LFR(n_iter, girvan, only_girvan)

    t2 = time.time()
    print(f"Total time: {t2-t1}s")








