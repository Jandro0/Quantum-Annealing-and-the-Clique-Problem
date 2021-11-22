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
        
        
        
    # Create the Ising Hamiltonian as a vector in the NP-complete case
    def makeIsingHamiltonian(self, K, A, B, c):
        dim = pow(2, self.nv)
        Hamiltonian = np.zeros(dim)
        state = [1 for i in range(self.nv)]
        
        h = defaultdict(int)
        for i in range(self.nv):
            h[(i)] = A*(2*K-self.nv)/2. + B*len(self.adjacency[i])/4.
        
        J = defaultdict(int)
        for i in range(self.nv):
            for j in range(i):
                J[(i,j)] = A/2.
            for j in self.adjacency[i]:
                if (i > j):
                    J[(i,j)] -= B/4.
        
        c[0] = A*K*K + (B*K*(K-1) - A*(2*K-1)*self.nv)/2. + (A*self.nv*(self.nv-1) - B*self.ne)/4.
                
                
        for k in range(dim):     
            for i in range(self.nv):
                Hamiltonian[k] += h[i]*state[i]
                for j in range(i):
                    Hamiltonian[k] += J[(i,j)]*state[i]*state[j]
            
            
            nextIsing(state)
        
        
        # bstate = [0 for i in range(self.nv)]
        
        # h = defaultdict(int)
        # for i in range(self.nv):
        #     h[(i)] = A*(1-2*K)
        
        # J = defaultdict(int)
        # for i in range(self.nv):
        #     for j in range(i):
        #         J[(i,j)] = 4*A
        #     for j in self.adjacency[i]:
        #         if (i > j):
        #             J[(i,j)] -= B
        
        # c[0] = A*K*K + (B*K*(K-1) - A*(2*K-1)*self.nv + A*self.nv*(self.nv-1) - B*self.ne)/2.
            
        # for k in range(dim):     
        #     for i in range(self.nv):
        #         Hamiltonian[k] += h[i]*bstate[i]
        #         for j in range(i):
        #             Hamiltonian[k] += J[(i,j)]*bstate[i]*bstate[j]
            
        #     print(bstate)
        #     plus1(bstate)
            
        
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
            network.add_edge(*e)
    
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
        if (state[n] == 1):
            done = True
            break
        n -= 1
        if (n < 0): 
            break
    
    state[n] = -1
    n += 1
    while (n < len(state)):
        state[n] = 1
        n += 1
 

def decimalToBinary (nv, d):
    b = [0 for i in range(nv)]
    
    for i in range(nv):
        mod = d % 2
        b[nv - i - 1] = int(mod)
        d = (d - mod)/2
    
    return b
    

    

# networkG = nx.fast_gnp_random_graph(120, 0.6)
# networkToFile(networkG, "graph4.txt")
# print(nx.graph_clique_number(networkG))


