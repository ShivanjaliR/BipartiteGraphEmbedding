import csv
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

def calculateEuclideanDistance(a1, b1, a2, b2, a3, b3, a4, b4):
    return sqrt(np.power((a1 - b1), 2) + np.power((a2 - b2), 2) + np.power((a3 - b3), 2) + np.power((a4 - b4), 2))

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

    # draw nodes and edges
    nx.draw(G, pos=pos, with_labels=True)

    # plot the title (if any)
    plt.title('Iris Dataset Graph')
    plt.savefig(graphName)
    plt.show()

