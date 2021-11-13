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
from collections import defaultdict
from Functions import fileToNetwork, networkToFile
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite


# Graph parameters
K = 4 #Size of the clique we are searching
G = fileToNetwork("graph1.txt")
nv = nx.number_of_nodes(G)

<<<<<<< HEAD
# Quantum parameters 
num_reads = 10
gamma = K + 1
chain_strength = 10
=======
# Parameters 
num_reads = 1
gamma = 80
>>>>>>> 97ee482e6008691fd47a82c37ab3654a9f9761d0

# Initialize matrix and fill in appropiate values
Q = defaultdict(int)

for j in range(nv):
    Q[(j,j)] += gamma*(1-2*K)
    for i in range(j):
        Q[(i,j)] += 2*gamma

for i,j in G.edges:
    Q[(int(i),int(j))] -= 1

<<<<<<< HEAD
constant = gamma*K*K + K*(K-1)/2
=======
#sampler = EmbedddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))
#sampleset = sampler.sample_qubo(Q)
>>>>>>> 97ee482e6008691fd47a82c37ab3654a9f9761d0



# Run the QUBO
sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))
sampleset = sampler.sample_qubo(Q,
                                chain_strength=chain_strength,
                                num_reads=num_reads,
                                label='Test - Clique Problem')

print(sampleset.to_pandas_dataframe())
print(' ')
print('Energy: ' + str(constant + sampleset.first.energy))
#print(sampleset.data)
#print(sampleset.info)
#print(sampleset.first)



# Check if the best solution found is actually a K-clique
if (constant == -sampleset.first.energy): 
    print(str(K) + '-clique found:', sampleset.record[0][0])
else: 
    print('No '+ str(K) + '-clique found.')



