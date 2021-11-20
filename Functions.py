import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



def fileToNetwork (nameIn):
    network = nx.Graph()
    
    with open(nameIn) as file: 
        nv, ne = [int(x) for x in next(file).split()]
        for line in file:
            e = line.split()
            edge = [int(e[0]), int(e[1])]
            network.add_edge(*edge)

    return network




def networkToFile (network, nameOut):
    with open(nameOut, 'w') as file: 
        file.write(str(len(list(network.nodes))) + ' ' + str(len(list(network.edges))))
        file.write('\n')
        for e in list(network.edges):
            file.write(str(e[0]) + ' ' + str(e[1]) + '\n')

