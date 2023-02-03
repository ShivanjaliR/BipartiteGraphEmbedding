import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
import pandas as pd
from sklearn import datasets
import itertools
import networkx as nx
from resources.constant import final_edges_index_csv, final_distances_csv, final_datapoints_csv, \
    adjacency_matrix_csv_file, iris_dataset_graph_img, iris_dataset
from utils import calculateEuclideanDistance, plotGraph, writeCsvFile


def datasetloading():
    iris = datasets.load_iris()
    writeCsvFile(iris['data'], iris_dataset)
    X = iris.data
    y = iris.target
    iris = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )
    return X


def generateEdges(dataset):
    # Calculating euclidean distance between points
    final_distances = []
    final_datapoints = []
    final_edges_index = []
    sub_distances = []
    sub_datapoints = []
    sub_edges_index = []
    index = 0
    previous_index = 0
    for (index1, data1), (index2, data2) in itertools.combinations(enumerate(dataset), 2):
        if previous_index != index1:
            final_edges_index.extend(sub_edges_index)
            final_distances.extend(sub_distances)
            final_datapoints.extend(sub_datapoints)
            sub_distances = []
            sub_datapoints = []
            sub_edges_index = []
        a1 = data1[0]
        a2 = data1[1]
        a3 = data1[2]
        a4 = data1[3]
        b1 = data2[0]
        b2 = data2[1]
        b3 = data2[2]
        b4 = data2[3]
        distance = calculateEuclideanDistance(a1, b1, a2, b2, a3, b3, a4, b4)
        if len(sub_distances) <= 3:
            sub_distances.append(distance)
            sub_datapoints.append((data1, data2))
            sub_edges_index.append((index1, index2))
            final_datapoints.append((data1, data2))
        else:
            max_distance_index = sub_distances.index(max(sub_distances))
            if distance < sub_distances[max_distance_index]:
                sub_distances[max_distance_index] = distance
                sub_datapoints[max_distance_index] = (data1, data2)
                sub_edges_index[max_distance_index] = (index1, index2)
                final_datapoints[max_distance_index] = (data1, data2)
        previous_index = index1

    pd.DataFrame(final_edges_index).to_csv(final_edges_index_csv, encoding='utf-8', float_format='%.5f')
    pd.DataFrame(final_distances).to_csv(final_distances_csv, encoding='utf-8', float_format='%.5f')
    pd.DataFrame(final_datapoints).to_csv(final_datapoints_csv, encoding='utf-8', float_format='%.5f')
    return final_edges_index


def generateGraph(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    A = nx.adjacency_matrix(G)
    adj_matrix = pd.DataFrame(A.toarray(), columns=np.array(G.nodes), index=np.array(G.nodes))
    print(adj_matrix)
    adj_matrix.to_csv(adjacency_matrix_csv_file, encoding='utf-8')
    return G


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = datasetloading()
    dataset_len = len(dataset)

    edges = generateEdges(dataset)
    nodes = range(1, len(dataset))
    G = generateGraph(nodes, edges)

    plotGraph(G, iris_dataset_graph_img)
