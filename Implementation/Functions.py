import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict



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



def physical_hJ (target_graph, minor, logical_h, logical_J):
    used_qubits = []
    for logical_qubit in minor:
        for physical_qubit in minor[logical_qubit]:
            used_qubits.append(physical_qubit)
    used_qubits.sort()
    
    number_of_logical_qubits = len(minor)
    used_graph = target_graph.subgraph(used_qubits)

    # Construct physical_h
    physical_h = defaultdict(int)
    for i in range(number_of_logical_qubits):
        chain_length = float(len(minor[i]))
        for physical_qubit in minor[i]:
            physical_h[(physical_qubit)] = logical_h[(i)]/chain_length

    # Construct physical_J
    physical_J = defaultdict(int)
    for i in range(number_of_logical_qubits):
            for j in range(i):
                number_of_connections = 0
                for node_i in minor[i]:
                    for node_j in used_graph.neighbors(node_i):
                        if node_j in minor[j]:
                            number_of_connections += 1
                for node_i in minor[i]:
                    for node_j in used_graph.neighbors(node_i):
                        if node_j in minor[j]:
                            physical_J[(max(node_i, node_j), min(node_i, node_j))] = logical_J[(i,j)]/number_of_connections


    return physical_h, physical_J



def evaluate_M (h, J, h_range, J_range):
    h_max = max(h.values())
    h_min = min(h.values())
    J_max = max(J.values())
    J_min = min(J.values())

    return max(J_max, -J_min, h_max/h_range, -h_min/h_range)

