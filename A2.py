from numpy.core.fromnumeric import size
import scipy as sp
import numpy as np
from scipy import sparse
import networkx as nx
import collections
from networkx.readwrite.edgelist import parse_edgelist 


if __name__=='__main__':

    edgelist_og = np.loadtxt("./data/email-Enron/email-Enron.txt", dtype=np.int64)

    # Remove the timestamp column
    edgelist = np.delete(edgelist_og, 2, axis=1)

    # edgelist = [(int(line[0]), int(line[1])) for line in edgelist]
    edgelist = [tuple(line) for line in edgelist]

    counts = collections.Counter(edgelist)

    edgelist_weighted = np.ndarray(shape=(0,3), dtype=np.int64)
    edgelist_weighted = []

    with open("./data/enron-weighted.edges.txt", "w") as out:
        for k,v in counts.items():
            line = f"{k[0]} {k[1]} {v}"
            out.write(f"{k[0]}\t{k[1]}\t{v}\n")
            edgelist_weighted.append(line)


    G = parse_edgelist(edgelist_weighted, nodetype=int, create_using=nx.DiGraph, data=(("weight", int),))

    G2 = nx.read_edgelist("./data/enron-weighted.edges.txt", create_using=nx.MultiDiGraph, nodetype=int, data=(("weight", int),))

    print(len(G))
    print(G[1][107])
    print(G[101][112])
    
    print(len(G2))
    print(G2[1][107])
    print(G2[101][112])