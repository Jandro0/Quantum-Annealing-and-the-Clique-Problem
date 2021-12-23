import numpy as np
import scipy.special
import networkx as nx
from collections import defaultdict


class graph:
    def __init__(self,  nv, ne, adjacency):
        self.nv = nv
        self.ne = ne
        self.adjacency = adjacency
        
    # Transform a graph to a graph in the networkx module        
    def toNetwork (self):
        network = nx.Graph()
        network.add_nodes_from(range(self.nv))
        for v in range(self.nv):
            for u in self.adjacency[v]:
                if (v < u):
                    e = (v, u)
                    network.add_edge(*e)
        
        
        return network
        
        
        
    # Create the Hamiltonian with chains for graph2.txt
    def makeHamiltonianWithChains (self, target_graph, minor, K, A, B, c, RCS):
        used_nodes = []
        for vertex in minor:
            for node in minor[vertex]:
                used_nodes.append(node)
        used_nodes.sort()
        
        number_of_physical_qubits = len(used_nodes)
        used_graph = target_graph.subgraph(used_nodes)

        
            
        dim = pow(2, number_of_physical_qubits)
        Hamiltonian = np.zeros(dim) 
        state = [-1 for i in range(number_of_physical_qubits)]
        
        h = defaultdict(int)
        logical_h = defaultdict(int)
        for node in used_graph.nodes:
            h[(node)] = 0
        for i in range(self.nv):
            chain_length = float(len(minor[i]))
            logical_h[(i)]= -A*(2*K-self.nv)/2. - B*len(self.adjacency[i])/4.
            for node in minor[i]:
                h[(node)] = logical_h[(i)]/chain_length
                
        logical_J = defaultdict(int)
        for i in range(self.nv):
            for j in range(i):
                logical_J[(i,j)] = A/2.
            for j in self.adjacency[i]:
                if (i > j):
                    logical_J[(i,j)] -= B/4.
        
        # logical_J_values = logical_J.values()
        # logical_J_max = max(logical_J_values)
        # logical_J_min = min(logical_J_values)
        # logical_h_max = max(logical_h.values())
        # logical_h_min = min(logical_h.values())
        # chain_strength = max(logical_J_max, -logical_J_min, logical_h_max, -logical_h_min) * RCS
        
        J = defaultdict(int)
        for i in range(number_of_physical_qubits):
            for j in range(i):
                J[(used_nodes[i], used_nodes[j])] = 0
        
        
        for i in range(self.nv):
            for j in range(i):
                number_of_connections = 0
                for node_i in minor[i]:
                    for node_j in used_graph.neighbors(node_i):
                        if node_j in minor[j]:
                            number_of_connections += 1
                for node_i in minor[i]:
                    for node_j in used_graph.neighbors(node_i):
                        if node_j in minor[j]:
                            J[(max(node_i, node_j), min(node_i, node_j))] = logical_J[(i,j)]/number_of_connections
            
        
        c[0] = A*K*K + (B*K*(K-1) - A*(2*K-1)*self.nv)/2. + (A*self.nv*(self.nv-1) - B*self.ne)/4.
        
        
        h_range = 2.0
        J_range = 1.0
        J_values = J.values()
        J_max = max(J_values)
        J_min = min(J_values)
        h_max = max(h.values())
        h_min = min(h.values())
        chain_strength = max(J_max, -J_min, h_max/h_range, -h_min/h_range) * RCS
        
        for i in range(self.nv):
            chain = used_graph.subgraph(minor[i])
            chain_edges = list(chain.edges)
            for edge in chain_edges:
                J[max(edge), min(edge)] = -chain_strength
        
        
        h_values = h.values() 
        
        h_max = max(h_values)
        h_min = min(h_values)
        J_values = J.values()
        J_max = max(J_values)
        J_min = min(J_values)
    
    
        scale = max([max([h_max/h_range, 0]), max([-h_min/h_range, 0]), max([J_max/J_range, 0]), max([-J_min/J_range, 0])])
        
        for i in range(number_of_physical_qubits):
            h[(used_nodes[i])] /= scale
        for i in range(number_of_physical_qubits):
            for j in range(i):
                J[(used_nodes[i], used_nodes[j])] /= scale
        
        
        for k in range(dim):     
            for i in range(number_of_physical_qubits):
                Hamiltonian[k] += h[used_nodes[i]]*state[i]
                for j in range(i):
                    Hamiltonian[k] += J[(used_nodes[i],used_nodes[j])]*state[i]*state[j]
            
            nextIsing(state)
        
        return Hamiltonian
        
        
        
    # Create the Ising Hamiltonian as a vector in the NP-complete case
    def makeIsingHamiltonian(self, K, A, B, c):
        dim = pow(2, self.nv)
        Hamiltonian = np.zeros(dim)
        state = [-1 for i in range(self.nv)]
        
        h = defaultdict(int)
        for i in range(self.nv):
            h[(i)] = -A*(2*K-self.nv)/2. - B*len(self.adjacency[i])/4.
        
        J = defaultdict(int)
        for i in range(self.nv):
            for j in range(i):
                J[(i,j)] = A/2.
            for j in self.adjacency[i]:
                if (i > j):
                    J[(i,j)] -= B/4.
        
        c[0] = A*K*K + (B*K*(K-1) - A*(2*K-1)*self.nv)/2. + (A*self.nv*(self.nv-1) - B*self.ne)/4.
        
        h_values = h.values() 
        h_range = 2.0
        h_max = max(h_values)
        h_min = min(h_values)
        J_values = J.values()
        J_range = 1.0
        J_max = max(J_values)
        J_min = min(J_values)
        
        scale = max([max([h_max/h_range, 0]), max([-h_min/h_range, 0]), max([J_max/J_range, 0]), max([-J_min/J_range, 0])])
        
        
        for i in range(self.nv):
            h[(i)] /= scale
        for i in range(self.nv):
            for j in range(i):
                J[(i,j)] /= scale
        
        
        for k in range(dim):     
            for i in range(self.nv):
                Hamiltonian[k] += h[i]*state[i]
                for j in range(i):
                    Hamiltonian[k] += J[(i,j)]*state[i]*state[j]
            
            nextIsing(state)
        
        return Hamiltonian
        
    
        
    # Create the Hamiltionian as a vector in the NP-complete case
    def makeHamiltonian(self, K, A, B):
        dim = pow(2,self.nv)
        Hamiltonian = np.empty(dim)
        state = []
        
        for i in range(self.nv):
            state.append(0)
        
        for i in range(dim):
            sum1 = 0 
            sum2 = 0
            for v in range(self.nv):
                sum1 += state[v]
                if (state[v] == 1):
                    for k in self.adjacency[v]:
                        sum2 += state[k]
                            
            Hamiltonian[i] = (A * pow(K - sum1, 2) + B * (K*(K-1) - sum2)/2) 
            plus1(state)
        
        
        return Hamiltonian
    
    
    # Create the Hamiltionian as a vector in the NP-hard case
    def makeHamiltonian2(self, A, B, C):
        M = int(np.floor(np.log2(self.nv)))
        N = int(self.nv + M + 1)
        dim = pow(2, N)
        Hamiltonian = np.empty(dim)
        state = []
        
        for i in range(N):
            state.append(0)
        
        for i in range(dim):
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for v in range(self.nv):
                sum1 += state[v]
                if (state[v] == 1):
                    for k in self.adjacency[v]:
                        sum2 += state[k]
            for k in range(M + 1):
                sum3 += pow(2, M - k) * state[self.nv + k]            
                            
            Hamiltonian[i] = A * pow(sum3 - sum1, 2) + B * (sum3*(sum3-1) - sum2)/2 - C*sum1
            plus1(state)
        
        
        return Hamiltonian
    
    
    
    # Create the Hamiltionian as a vector in the NP-complete (alternative) case
    def makeHamiltonian3 (self, K, B):
        dim = int(scipy.special.binom(self.nv, K))
        Hamiltonian = np.empty(dim)
        state = []
        
        for i in range(self.nv):
            state.append(0)
    
        i = 0
        while (i < dim):
            number = 0
            for k in range(self.nv):
                number += state[k]
            if (number == K):
                sum1 = 0 
                for v in range(self.nv):
                    if (state[v] == 1):
                        for k in self.adjacency[v]:
                            sum1 += state[k]
                                
                Hamiltonian[i] = B * (K*(K-1) - sum1)/2 
                i += 1
            
            plus1(state)
    
        return Hamiltonian
    
    
    # Create the Hamiltionian as a vector in the NP-hard (alternative) case
    def makeHamiltonian4 (self, B, C):
        dim = int(pow(2,self.nv))
        R = int(self.nv + np.floor(np.log2(self.nv)) + 1)
        Hamiltonian = np.empty(dim)
        state = []
        
        for i in range(R):
            state.append(0)
    
        i = 0
        while (i < dim):
            number = 0
            size = 0
            for k in range(self.nv):
                number += state[k]
            for k in range(self.nv, R):
                size += int(pow(2, R-k-1))*state[k]
                
            if (number == size):
                sum1 = 0 
                for v in range(self.nv):
                    if (state[v] == 1):
                        for k in self.adjacency[v]:
                            sum1 += state[k]
                                
                Hamiltonian[i] = B * (size*(size-1) - sum1)/2 - C * number
                i += 1
            
            plus1(state)
    
        return Hamiltonian
    
    
    
    
# Read a text file and create a graph
def createGraph (nameIn):
    nv = 0
    ne = 0
    vertex = [[]]
    
    with open(nameIn) as file:
        nv, ne = [int(x) for x in next(file).split()]
        for i in range(nv - 1):
            vertex.append([])
        for line in file:
            e = line.split()
            vertex[int(e[0])].append(int(e[1]))
            vertex[int(e[1])].append(int(e[0]))
        
    return graph(nv, ne, vertex)





# Read a text file and create a network
def fileToNetwork (nameIn):
    network = nx.Graph()
    
    with open(nameIn) as file: 
        nv, ne = [int(x) for x in next(file).split()]
        for line in file:
            e = line.split()
            network.add_edge(int(e[0]), int(e[1]))
    
    return network





def networkToGraph (networkG):
    nv = 0
    ne = 0
    vertex = [[]]
    
    nv = len(list(networkG.nodes))
    ne = len(list(networkG.edges))
    
    for i in range(nv - 1):
        vertex.append([])
    for e in list(networkG.edges):
        vertex[e[0]].append(e[1])
        vertex[e[1]].append(e[0])
    
    return graph(nv, ne, vertex)



def networkToFile (network, nameOut):
    with open(nameOut, 'w') as file: 
        file.write(str(len(list(network.nodes))) + ' ' + str(len(list(network.edges))))
        file.write('\n')
        for e in list(network.edges):
            file.write(str(e[0]) + ' ' + str(e[1]) + '\n')



# Sum +1 in binary
def plus1 (number):
    n = len(number) - 1
    done = False

    while not (done):
        if (number[n] == 0):
            done = True
            break
        n -= 1
        if (n < 0): 
            break
    
    number[n] = 1
    n += 1
    while (n < len(number)):
        number[n] = 0
        n += 1
        
        
# Next Ising state
def nextIsing (state):
    n = len(state) - 1
    done = False

    while not (done):
        if (state[n] == -1):
            done = True
            break
        n -= 1
        if (n < 0): 
            break
    
    state[n] = 1
    n += 1
    while (n < len(state)):
        state[n] = -1
        n += 1
 

def decimalToBinary (n, d):
    b = [0 for i in range(n)]
    
    for i in range(n):
        mod = d % 2
        b[n - i - 1] = int(mod)
        d = (d - mod)/2
    
    return b

def binaryToDecimal (b):
    n = len(b)
    d = 0
    
    for i in range(n):
        d += b[n - i - 1]*pow(2, i)
    
    return d




def generalHamiltonian (target_graph, minor, logical_h, logical_J, RCS):
    number_of_logical_qubits = len(logical_h)
    number_of_physical_qubits = 0
    for logical_qubit in minor:
        number_of_physical_qubits += len(minor[logical_qubit])
    
    dim = pow(2, number_of_physical_qubits)
    Hamiltonian = np.zeros(dim)
    state = [-1 for i in range(number_of_physical_qubits)]
    physical_h = defaultdict(int)
    physical_J = defaultdict(int)
    
    for i in range(number_of_physical_qubits):
        physical_h[(i)] = 0.0
        for j in range(i):
            physical_J[(i, j)] = 0.0
    
    for logical_qubit in range(number_of_logical_qubits):
        chain_length = float(len(minor[logical_qubit]))
        for physical_qubit in minor[logical_qubit]:
            physical_h[(physical_qubit)] = logical_h[(logical_qubit)]/chain_length 
    

    # logical_J_max = max(logical_J.values())
    # logical_J_min = min(logical_J.values())
    # logical_h_max = max(logical_h.values())
    # logical_h_min = min(logical_h.values())
    # chain_strength = max(logical_J_max, -logical_J_min, logical_h_max, -logical_h_min) * RCS
    
    
    for i in range(number_of_logical_qubits):
        for j in range(i):
            number_of_connections = 0
            for physical_qubit_i in minor[i]:
                for physical_qubit_j in target_graph.neighbors(physical_qubit_i):
                    if physical_qubit_j in minor[j]:
                        number_of_connections += 1
            for physical_qubit_i in minor[i]:
                for physical_qubit_j in target_graph.neighbors(physical_qubit_i):
                    if physical_qubit_j in minor[j]:
                        physical_J[(max(physical_qubit_i, physical_qubit_j), min(physical_qubit_i, physical_qubit_j))] = logical_J[(i,j)]/number_of_connections
    
    h_range = 2.0
    J_range = 1.0
    physical_J_max = max(physical_J.values())
    physical_J_min = min(physical_J.values())
    physical_h_max = max(physical_h.values())
    physical_h_min = min(physical_h.values())
    chain_strength = max(physical_J_max, -physical_J_min, physical_h_max/h_range, -physical_h_min/h_range) * RCS
    #chain_strength = max(physical_J_max, -physical_J_min) * RCS
    
    for i in range(number_of_logical_qubits):
        chain = target_graph.subgraph(minor[i])
        chain_edges = list(chain.edges)
        for edge in chain_edges:
            physical_J[(max(edge), min(edge))] = -chain_strength
            
    
    h_values = physical_h.values() 
    h_max = max(h_values)
    h_min = min(h_values)
    J_values = physical_J.values()
    J_max = max(J_values)
    J_min = min(J_values)
    
    scale = max([max([h_max/h_range, 0]), max([-h_min/h_range, 0]), max([J_max/J_range, 0]), max([-J_min/J_range, 0])])
    
    for i in range(number_of_physical_qubits):
        physical_h[(i)] /= scale
    for i in range(number_of_physical_qubits):
        for j in range(i):
            physical_J[(i, j)] /= scale
    
    
    for k in range(dim):     
        for i in range(number_of_physical_qubits):
            Hamiltonian[k] += physical_h[i]*state[i]
            for j in range(i):
                Hamiltonian[k] += physical_J[(i, j)]*state[i]*state[j]
        
        nextIsing(state)
    
    return Hamiltonian
    

def evaluateHamiltonian (n, h, J):
    dim = pow(2, n)
    Hamiltonian = np.zeros(dim)
    state = [-1 for i in range(n)]
    
    for k in range(dim):     
        for i in range(n):
            Hamiltonian[k] += h[i]*state[i]
            for j in range(i):
                Hamiltonian[k] += J[(i, j)]*state[i]*state[j]
        
        nextIsing(state)
    
    return Hamiltonian



# networkG = nx.fast_gnp_random_graph(120, 0.6)
# networkToFile(networkG, "graph4.txt")
# print(nx.graph_clique_number(networkG))


