'''
Similarity Matrix: https://dataaspirant.com/five-most-popular-similarity-measures-implementation-in-python/
'''
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
import pandas as pd
from sklearn import datasets
import itertools
import networkx as nx
from resources.constant import final_edges_index_csv, final_distances_csv, final_datapoints_csv, \
    adjacency_matrix_csv_file, iris_dataset_graph_img, iris_dataset, node_embedding_scatter_plot, \
    edge_embedding_scatter_plot, node_embedding_title, edge_embedding_title
from utils import calculateEuclideanDistance, plotGraph, writeCsvFile, calculateEuclideanDistanceEdge, \
    calculateManhattanDistance, calculateMinkowskiDistance, calculateCosinDistance
from node2vec import Node2Vec
from sklearn.manifold import TSNE
# Embed edges using Hadamard method
from node2vec.edges import HadamardEmbedder


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
        distance = calculateEuclideanDistanceEdge(a1, b1, a2, b2, a3, b3, a4, b4)
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


def featureExtraction(G, title):
    model = Node2Vec(G, dimensions=10, walk_length=4, num_walks=100,
                     p=0.25, q=4, workers=1)
    model = model.fit(window=10, min_count=1, batch_words=4)
    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = model.wv.vectors
    tsne = TSNE(n_components=2, random_state=7, perplexity=15)
    embeddings_2d = tsne.fit_transform(node_embeddings)

    figure = plt.figure(figsize=(11, 9))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    for i, word in enumerate(node_ids):
        plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))
    plt.title(title)
    plt.savefig(node_embedding_scatter_plot)
    plt.show()
    print(node_ids)
    print(node_embeddings)

    return model, node_embeddings


def getEuclideanSimilarity(low_dimensional_node_embedding):
    occurrences = np.zeros((len(low_dimensional_node_embedding), len(low_dimensional_node_embedding)), dtype=np.int32)
    for (index1, data1), (index2, data2) in itertools.combinations(enumerate(low_dimensional_node_embedding), 2):
        distance = calculateEuclideanDistance(data1, data2)
        occurrences[index1][index2] = distance
        occurrences[index2][index1] = distance
    return occurrences

def getManhattanSimilarity(low_dimensional_node_embedding):
    occurrences = np.zeros((len(low_dimensional_node_embedding), len(low_dimensional_node_embedding)), dtype=np.int32)
    for (index1, data1), (index2, data2) in itertools.combinations(enumerate(low_dimensional_node_embedding), 2):
        distance = calculateManhattanDistance(data1, data2)
        occurrences[index1][index2] = distance
        occurrences[index2][index1] = distance
    return occurrences

def getMinkowskiSimilarity(low_dimensional_node_embedding):
    occurrences = np.zeros((len(low_dimensional_node_embedding), len(low_dimensional_node_embedding)), dtype=np.int32)
    for (index1, data1), (index2, data2) in itertools.combinations(enumerate(low_dimensional_node_embedding), 2):
        distance = calculateMinkowskiDistance(data1, data2, 3)
        occurrences[index1][index2] = distance
        occurrences[index2][index1] = distance
    return occurrences

def getCosineSimilarity(low_dimensional_node_embedding):
    occurrences = np.zeros((len(low_dimensional_node_embedding), len(low_dimensional_node_embedding)), dtype=np.int32)
    for (index1, data1), (index2, data2) in itertools.combinations(enumerate(low_dimensional_node_embedding), 2):
        distance = calculateCosinDistance(data1, data2, 3)
        occurrences[index1][index2] = distance
        occurrences[index2][index1] = distance
    return occurrences

def edgeEmbedding(model, title):
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
    # Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
    edges_kv = edges_embs.as_keyed_vectors()
    node_ids = model.wv.index_to_key  # list of node IDs
    tsne = TSNE(n_components=2, random_state=7, perplexity=15)
    embeddings_2d = tsne.fit_transform(edges_kv.vectors)

    figure = plt.figure(figsize=(11, 9))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    '''for i, word in enumerate(node_ids):
        plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))'''
    plt.title(title)
    plt.savefig(edge_embedding_scatter_plot)
    plt.show()
    print(edges_kv.vectors)
    print(len(edges_kv.vectors))
    return edges_kv.vectors


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = datasetloading()
    dataset_len = len(dataset)

    edges = generateEdges(dataset)
    nodes = range(1, len(dataset))
    G = generateGraph(nodes, edges)

    plotGraph(G, iris_dataset_graph_img)

    model, node_embedding = featureExtraction(G, node_embedding_title)

    edgeEmbedding_matrix = edgeEmbedding(model, edge_embedding_title)

    similarity_matrix = getEuclideanSimilarity(node_embedding)
