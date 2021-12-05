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
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
import dwave.inspector
from dwave.embedding.chimera import find_clique_embedding
import minorminer as mm


# Graph parameters
G = fileToNetwork("graph2.txt")
K = nx.graph_clique_number(G) #Size of the clique we are searching
print("Clique number: " + str(K))
nv = nx.number_of_nodes(G)
ne = nx.number_of_edges(G)


# Quantum parameters 
B = 1.0
A = (K + 1)*B
chain_strength = 10.0


# Initialize matrix and fill in appropiate values
h = defaultdict(int)
for i in range(nv):
    h[(i)] = -A*(2*K-nv)/2. - B*len(G[i])/4.

J = defaultdict(int)
for i in range(nv):
    for j in range(i):
        J[(i,j)] = A/2.
    for j in G.neighbors(i):
        if (i > j):
            J[(i,j)] -= B/4.


constant = A*K*K + (B*K*(K-1) - A*(2*K-1)*nv)/2. + (A*nv*(nv-1) - B*ne)/4.
print("Ground state energy: " + str(constant))


# Compute the maximum strength 
h_max = np.absolute(max(h.values()))
h_min = np.absolute(min(h.values()))
J_max = np.absolute(max(J.values()))
J_min = np.absolute(min(J.values()))
max_strength = max(J_max, J_min)


# Do the embedding
select = 0
if select == 0:
    c1 = dnx.chimera_graph(1)
    networkToFile(c1, "chimera.txt")
    plt.figure()
    dnx.draw_chimera(c1)
    filename = "Chimera_graph.png"
    plt.savefig(filename, bbox_inches='tight')
    K4 = nx.complete_graph(4)
    embedding = mm.find_embedding(K4, c1, random_seed=1)

    qpu = DWaveSampler(solver={'topology__type': 'chimera'})
    sampler = FixedEmbeddingComposite(qpu, embedding)
    sampleset = sampler.sample_ising(h, J,
                                    chain_strength=chain_strength,
                                    num_reads=1,
                                    auto_scale=True,
                                    annealing_time=20.0,
                                    label='Test - Clique Problem')
    dwave.inspector.show(sampleset)
elif select == 1:
    p1 = dnx.pegasus_graph(16, fabric_only=False)
    networkToFile(p1, "pegasus.txt")
    plt.figure()
    dnx.draw_pegasus(p1)
    filename = "Pegasus_graph.png"
    plt.savefig(filename, bbox_inches='tight')
    K4 = nx.complete_graph(4)
    embedding = mm.find_embedding(K4, p1, random_seed=1)

    qpu = DWaveSampler(solver={'topology__type': 'pegasus'})
    sampler = FixedEmbeddingComposite(qpu, embedding)
    sampleset = sampler.sample_ising(h, J,
                                    chain_strength=chain_strength,
                                    num_reads=1,
                                    auto_scale=True,
                                    annealing_time=20.0,
                                    label='Test - Clique Problem')
    dwave.inspector.show(sampleset)

