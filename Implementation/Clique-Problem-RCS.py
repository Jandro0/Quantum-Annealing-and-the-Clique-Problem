# Copyright [2022] [Alejandro Garcia Rivas]
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

# This piece of code scans the selected problem over RCS with fixed annealing time 


import numpy as np
import networkx as nx
import dwave_networkx as dnx
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from Functions import fileToNetwork, networkToFile, physical_hJ
from dwave.system.samplers import DWaveSampler, LeapHybridSampler, LeapHybridBQMSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
import dwave.inspector
import minorminer as mm


# Graph parameters
number_of_vertices = 30
p = 0.8
#G = fileToNetwork("graph5.txt")
G = fileToNetwork("graph" + str(number_of_vertices) + "-" + str(p) + ".txt")
K = nx.graph_clique_number(G) #Size of the clique we are searching
print("Clique number: " + str(K))
nv = nx.number_of_nodes(G)
ne = nx.number_of_edges(G)


# Quantum parameters 
min_RCS = 0.0
max_RCS = 1.0
num_RCS = 21
RCS = np.linspace(min_RCS, max_RCS, num_RCS)
success_rate = np.empty(num_RCS)
deviation = np.empty(num_RCS)
num_reads = 500
beta = 2.0
alpha = 1.0




# Initialize coefficients h_i, J_{ij} and fill in appropiate values
h = defaultdict(int)
for i in range(nv):
    h[(i)] = -alpha/2.0 + beta*(nv - 1 - len(G[i]))/4.0

J = defaultdict(int)
for i in range(nv):
    for j in range(i):
        if (j not in G.neighbors(i)):
            J[(i,j)] = beta/4.0

constant = -alpha*nv/2.0 + beta*(nv*(nv-1)/2 - ne)/4.0

# Create the complement of G (graph to be embedded in the QPU)
G_complement = nx.complement(G)


# Compute minor-embeddings 
chimera = dnx.chimera_graph(16)
chimera_embedding = mm.find_embedding(G_complement, chimera, random_seed=1)
qpu_chimera = DWaveSampler(solver={'topology__type': 'chimera'})
sampler_chimera = FixedEmbeddingComposite(qpu_chimera, chimera_embedding)

pegasus = dnx.pegasus_graph(16, fabric_only=False)
pegasus_embedding = mm.find_embedding(G_complement, pegasus, random_seed=1)
qpu_pegasus = DWaveSampler(solver={'topology__type': 'pegasus'})
sampler_pegasus = FixedEmbeddingComposite(qpu_pegasus, pegasus_embedding)

# Compute max_strength
h_max = max(h.values())
h_min = min(h.values())
J_max = max(J.values())
J_min = min(J.values())
max_strength = max(h_max, -h_min, J_max, -J_min)


# Run the annealing with the desired sampler
for i in range(num_RCS):
    for select in [0]: #This controls which QPU we want to use (0=DW2000Q and 1=Advantage)
        if (select == 0):
            sampleset = sampler_chimera.sample_ising(h, J,
                                            chain_strength=RCS[i]*max_strength,
                                            num_reads=num_reads,
                                            auto_scale=True,
                                            annealing_time=20.0,
                                            label='Maximum Clique Problem with DW2000Q')
            #dwave.inspector.show(sampleset)
        elif (select == 1):
            sampleset = sampler_pegasus.sample_ising(h, J,
                                            chain_strength=RCS[i]*max_strength,
                                            num_reads=num_reads,
                                            auto_scale=True,
                                            annealing_time=20.0,
                                            label='Maximum Clique Problem with Advantage')
            #dwave.inspector.show(sampleset)
        elif (select == 2):
            sampler = LeapHybridSampler()
            sampleset = sampler.sample_ising(h, J)
        elif (select == 3):
            sampler = LeapHybridBQMSampler()
            sampleset = sampler.sample_ising(h, J)


        # Print results
        if (select == 0):
            print("-------------------------DW_2000Q_6-------------------------")
        elif (select == 1):
            print("--------------------Adavantage_system4.1--------------------")
        print(sampleset.to_pandas_dataframe())
        print(' ')
        print('Energy: ' + str(constant + sampleset.first.energy))
        #print(sampleset.data)
        #print(sampleset.info)
        #print(sampleset.first)



        # Check if the best solution found is actually a maximum clique and print results
        dict_state = sampleset.first.sample.values()
        state = list(dict_state)
        if (constant + sampleset.first.energy == -float(K)): 
            print(str(K) + '-clique found with RCS = ' + str(RCS[i]) + ':', state, '\n\n')

            groundStateSet = sampleset.lowest(atol=0.1)
            success_rate[i] = float(np.sum(groundStateSet.record.num_occurrences))/float(num_reads)
            deviation[i] = np.sqrt(success_rate[i]*(1-success_rate[i])/num_reads)

        else: 
            print('No '+ str(K) + '-clique found with RCS = ' + str(RCS[i]) + '\n\n')
            success_rate[i] = 0.0
            deviation[i] = 0.0


        # # Plot and save
        # N0 = [i for i in G.nodes if state[i] == -1]
        # N1 = [i for i in G.nodes if state[i] == 1]
        # E0 = [(i,j) for i,j in G.edges if (state[i] == -1 or state[j] == -1)]
        # E1 = [(i,j) for i,j in G.edges if (state[i] == 1 and state[j] == 1)]

        # plt.figure()
        # pos = nx.spring_layout(G)
        # nx.draw_networkx_nodes(G, pos, nodelist = N0, node_color='red')
        # nx.draw_networkx_nodes(G, pos, nodelist = N1, node_color='blue')
        # nx.draw_networkx_edges(G, pos, edgelist = E0, style='dashdot', alpha=0.5, width=3)
        # nx.draw_networkx_edges(G, pos, edgelist = E1, style='solid', width=3)
        # nx.draw_networkx_labels(G, pos)

        # filename = "K-clique " + "(QPU: " + str(select) + ").png"
        # plt.savefig(filename, bbox_inches='tight')



# Save results
filename = "Success rate DW (" + str(nv) + "-" + str(p) + ").txt"
with open(filename, 'w') as file:
    for i in range(num_RCS):
        file.write(str(RCS[i]) + ' ' + str(success_rate[i]) + ' ' + str(deviation[i]) + '\n')


