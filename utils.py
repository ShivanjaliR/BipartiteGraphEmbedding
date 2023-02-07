import csv
from decimal import Decimal
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx


def calculateEuclideanDistanceEdge(a1, b1, a2, b2, a3, b3, a4, b4):
    return sqrt(np.power((a1 - b1), 2) + np.power((a2 - b2), 2) + np.power((a3 - b3), 2) + np.power((a4 - b4), 2))


def calculateEuclideanDistance(node1, node2):
    sum = 0
    for i in range(len(node1)):
        sum += np.power((node1[i] - node2[i]), 2)
    return sqrt(sum)

def calculateManhattanDistance(node1, node2):
    sum = 0
    for i in range(len(node1)):
        sum += np.abs((node1[i] - node2[i]))
    return sum

def calculateMinkowskiDistance(node1, node2, p_value):
    sum = 0
    for i in range(len(node1)):
        sum += np.power(np.abs((node1[i] - node2[i])), p_value)
    root_value = 1 / float(sum)
    result = round(Decimal(sum) ** Decimal(root_value), p_value)
    return result

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)
def calculateCosinDistance(node1, node2):
    numerator = 0
    for i in range(len(node1)):
        numerator += node1[i] * node2[i]

    denominator = square_rooted(node1) * square_rooted(node2)
    return round(numerator / float(denominator), 3)

def readCsvFile(fileName):
    content = []
    # opening the CSV file
    with open(fileName, mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)

        # displaying the contents of the CSV file
        for lines in csvFile:
            content.append(lines)
    return content


def writeCsvFile(dataset, fileName):
    rows = []
    with open(fileName, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dataset)


def plotGraph(G, graphName):
    # set figure size
    plt.figure(figsize=(10, 10))

    # define position of nodes in figure
    # pos = nx.nx_agraph.graphviz_layout(G)
    pos = nx.spring_layout(G)

    node_colors = [node[1]['color'] for node in G.nodes(data=True)]
    # draw nodes and edges
    nx.draw(G, pos=pos, with_labels=True, node_color=node_colors)

    # plot the title (if any)
    plt.title('Iris Dataset Graph')
    plt.savefig(graphName)
    plt.show()
