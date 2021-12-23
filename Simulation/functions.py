import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import networkx as nx

def fraction (t):
    return 100/(t+1)

def exponential (t):
    return 100/np.exp(t)

def exponentialsqrt (t):
    return 100/np.exp(np.sqrt(t))

def linearA (s):
    return (1 - s)

def linearB (s):
    return s

def schedule (nameIn):
    vecA = np.empty(1000)
    vecB = np.empty(1000)
    t = np.empty(1000)
    i = 0
    
    with open(nameIn) as file:
        for line in file:
            content = line.split()
            t[i] = float(content[0])
            vecA[i] = np.pi*float(content[1])
            vecB[i] = np.pi*float(content[2])
            i += 1
            
    A = interp1d(t, vecA, kind = 'quadratic')
    B = interp1d(t, vecB, kind = 'quadratic')
    
    return A, B


def f(t, y, H0, H1, A, B, t_f):
    return (0. - 1.j)*np.matmul(A(t/t_f)*H0 + B(t/t_f)*H1, y)
    


# G = nx.Graph()
# G.add_edge(1, 2)
# G.add_edge(1, 3)
# G.add_edge(2, 3)
# N0 = [1]
# N1 = [2]
# N2 = [3]
# E0 = []
# E1 = [(1, 2), (1, 3), (2, 3)]
# node_labels = {1: -1, 2: -1, 3: -2}
# edge_labels = {(1, 2): 1, (1, 3): 1, (2, 3): 1}


# plt.figure()
# pos = nx.planar_layout(G)
# nx.draw_networkx_nodes(G, pos, nodelist = N0, node_color='green')
# nx.draw_networkx_nodes(G, pos, nodelist = N1, node_color='red')
# nx.draw_networkx_nodes(G, pos, nodelist = N2, node_color='blue')
# nx.draw_networkx_edges(G, pos, edgelist = E0, style='dashdot', alpha=0.5, width=3)
# nx.draw_networkx_edges(G, pos, edgelist = E1, style='solid', width=3)
# nx.draw_networkx_labels(G, pos, labels=node_labels)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# plt.savefig("triangle.png", bbox_inches='tight', dpi=300)


# G1 = nx.Graph()
# G1.add_edge(1, 2)
# G1.add_edge(1, 3)
# G1.add_edge(2, 4)
# G1.add_edge(3, 4)
# N0 = [1]
# N1 = [2]
# N2 = [3, 4]
# E0 = [(3, 4)]
# E1 = [(1, 2), (1, 3), (2, 4)]
# node_labels = {1: -1, 2: -1, 3: -1, 4: -1}
# edge_labels = {(1, 2): 1, (1, 3): 1, (2, 4): 1}


# plt.figure()
# pos = nx.spring_layout(G1)
# nx.draw_networkx_nodes(G1, pos, nodelist = N0, node_color='green')
# nx.draw_networkx_nodes(G1, pos, nodelist = N1, node_color='red')
# nx.draw_networkx_nodes(G1, pos, nodelist = N2, node_color='blue')
# nx.draw_networkx_edges(G1, pos, edgelist = E0, style='dashed', alpha=0.5, width=3)
# nx.draw_networkx_edges(G1, pos, edgelist = E1, style='solid', width=3)
# nx.draw_networkx_labels(G1, pos, labels=node_labels)
# nx.draw_networkx_edge_labels(G1, pos, edge_labels=edge_labels)
# plt.savefig("triangle_embedded.png", bbox_inches='tight', dpi=300)




# A, B = schedule("Advantage_system4.1.txt")
   

# plt.figure()
# xAxis = np.linspace(0, 1, 200)
# plt.xlim([0,1])
# plt.xlabel("Scaled annealing time s=t/T")
# plt.ylabel("Energy/h (GHz)")
# A_values = A(xAxis)
# B_values = B(xAxis)
# plt.plot(xAxis, A_values, label='A(s)')
# plt.plot(xAxis, B_values, label='B(s)')
# plt.legend(loc='best')
# plt.savefig("Advantage schedule", bbox_inches='tight', dpi=300)
# plt.show()
    
    
    
    
    
    
    