import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

"""
    Adapted from the original
    https://github.com/tkipf/gcn/blob/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81/gcn/utils.py
"""
def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/gcn/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/gcn/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    G = nx.from_dict_of_lists(graph)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    n_cat = labels.shape[1]

    # There are 15 nodes with no category in citeseer
    # We will arbitrarily assign them to category 0 
    # since its easier than modifying the graph object to remove them
    if dataset_str == 'citeseer':
        labels_all = []

        for line in labels:
            category = np.where(line == 1)
            if len(category[0]) > 0:
                category = int(category[0][0])
            else:
                category = 0
            labels_all.append(category)
        
        labels = labels_all
    else:
        labels = list(labels.nonzero()[1])

    print(f"Graph has N={len(G.nodes)}, E={len(G.edges)} and has {n_cat} categories")

    return G, labels, n_cat


if __name__=='__main__':

    G, labels, n_cat = load_data("citeseer")

    print(len(labels))

    