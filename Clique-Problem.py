# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Things to do:
 - Please name this file <demo_name>.py
 - Fill in [yyyy] and [name of copyright owner] in the copyright (top line)
 - Add demo code below
 - Format code so that it conforms with PEP 8
"""

import numpy as np
import networkx as nx
import dwave_networkx as dnx
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from Functions import fileToNetwork, networkToFile
from dwave.system.samplers import DWaveSampler, LeapHybridSampler, LeapHybridBQMSampler
from dwave.system.composites import EmbeddingComposite


# Graph parameters
K = 4 #Size of the clique we are searching
G = fileToNetwork("graph1.txt")
nv = nx.number_of_nodes(G)


# Quantum parameters 
num_reads = 10
B = 1
gamma = (K + 1)*B
chain_strength = 150



# Initialize matrix and fill in appropiate values
Q = defaultdict(int)

for j in range(nv):
    Q[(j,j)] += gamma*(1-2*K)
    for i in range(j):
        Q[(i,j)] += 2*gamma

for i,j in G.edges:
    Q[(int(i),int(j))] -= B


constant = gamma*K*K + B*K*(K-1)/2



# Run the QUBO with the desired sampler
select = 0
if (select == 0):
    sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))
    sampleset = sampler.sample_qubo(Q,
                                    chain_strength=chain_strength,
                                    num_reads=num_reads,
                                    label='Test - Clique Problem')
elif (select == 1):
    sampler = LeapHybridSampler()
    sampleset = sampler.sample_qubo(Q)
elif (select == 2):
    sampler = LeapHybridBQMSampler()
    sampleset = sampler.sample_qubo(Q)


# Print results
print(sampleset.to_pandas_dataframe())
print(' ')
print('Energy: ' + str(constant + sampleset.first.energy))
#print(sampleset.data)
#print(sampleset.info)
#print(sampleset.first)



# Check if the best solution found is actually a K-clique
state = sampleset.record[0][0]
if (constant == -sampleset.first.energy): 
    print(str(K) + '-clique found:', state)
else: 
    print('No '+ str(K) + '-clique found.')


# Plot and save
N0 = [i for i in G.nodes if not state[int(i)]]
N1 = [i for i in G.nodes if state[int(i)]]
E0 = [(i,j) for i,j in G.edges if state[int(i)] == 0 or state[int(j)] == 0]
E1 = [(i,j) for i,j in G.edges if state[int(i)] == 1 and state[int(j)] == 1]

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, nodelist = N0, node_color='red')
nx.draw_networkx_nodes(G, pos, nodelist = N1, node_color='blue')
nx.draw_networkx_edges(G, pos, edgelist = E0, style='dashdot', alpha=0.5, width=3)
nx.draw_networkx_edges(G, pos, edgelist = E1, style='solid', width=3)
nx.draw_networkx_labels(G, pos)

filename = "K-clique.png"
plt.savefig(filename, bbox_inches='tight')