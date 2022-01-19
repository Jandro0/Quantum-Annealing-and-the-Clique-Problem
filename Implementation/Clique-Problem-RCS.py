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
from Functions import fileToNetwork, networkToFile, physical_hJ, evaluate_M
from dwave.system.samplers import DWaveSampler, LeapHybridSampler, LeapHybridBQMSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
import dwave.inspector
import minorminer as mm


# Graph parameters
G = fileToNetwork("graph2.txt")
# G = nx.fast_gnp_random_graph(7, 0.5)
K = nx.graph_clique_number(G) #Size of the clique we are searching
print("Clique number: " + str(K))
nv = nx.number_of_nodes(G)
ne = nx.number_of_edges(G)


# Quantum parameters 
min_annealing_time = 10.
max_annealing_time = 10.
num_annealing_time = 1
annealing_steps = np.linspace(np.log10(min_annealing_time), np.log10(max_annealing_time), num_annealing_time, dtype=float)
annealing_times = np.power(10, annealing_steps)
min_RCS = 0.6
max_RCS = 0.6
num_RCS = 1
RCS = np.linspace(min_RCS, max_RCS, num_RCS)
probability_of_success = np.empty((2, num_RCS))
num_reads = 200
B = 1.0
A = (K + 1)*B
token = "DEV-fd3f1d6b05742414a33e65d30d4ac65edc88415b"



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

# Compute minor-embeddings
chimera = dnx.chimera_graph(3)
Kn = nx.complete_graph(nv)
chimera_embedding = mm.find_embedding(Kn, chimera, random_seed=1)

pegasus = dnx.pegasus_graph(16, fabric_only=False)
Kn = nx.complete_graph(nv)
pegasus_embedding = mm.find_embedding(Kn, pegasus, random_seed=1)

# Compute values of M
physical_chimera_h, physical_chimera_J = physical_hJ(chimera, chimera_embedding, h, J)
physical_pegasus_h, physical_pegasus_J = physical_hJ(pegasus, pegasus_embedding, h, J)
M_chimera = evaluate_M(physical_chimera_h, physical_chimera_J, 2.0, 1.0)
M_pegasus = evaluate_M(physical_pegasus_h, physical_chimera_J, 4.0, 1.0)



# Run the annealing with the desired sampler
for i in range(num_RCS):
    for select in [0]:
        if (select == 0):
            qpu = DWaveSampler(solver={'topology__type': 'chimera'}, token=token)
            sampler = FixedEmbeddingComposite(qpu, chimera_embedding)
            sampleset = sampler.sample_ising(h, J,
                                            chain_strength=RCS[i]*M_chimera,
                                            num_reads=num_reads,
                                            auto_scale=True,
                                            annealing_time=20.0,
                                            label='Test - Clique Problem')
            dwave.inspector.show(sampleset)
        elif (select == 1):
            qpu = DWaveSampler(solver={'topology__type': 'pegasus'}, token=token)
            sampler = FixedEmbeddingComposite(qpu, pegasus_embedding)
            sampleset = sampler.sample_ising(h, J,
                                            chain_strength=RCS[i]*M_pegasus,
                                            num_reads=num_reads,
                                            auto_scale=True,
                                            annealing_time=20.0,
                                            label='Test - Clique Problem')
            #dwave.inspector.show(sampleset)
        elif (select == 2):
            sampler = LeapHybridSampler()
            sampleset = sampler.sample_ising(h, J, token=token)
        elif (select == 3):
            sampler = LeapHybridBQMSampler()
            sampleset = sampler.sample_ising(h, J, token=token)


        # Print results
        if (select == 0):
            print("-------------------------DW_2000Q_6-------------------------")
        elif (select == 1):
            print("--------------------Adavantage_system4.1--------------------")
        print(sampleset.to_pandas_dataframe())
        print(' ')
        print('Energy: ' + str(constant + sampleset.first.energy))
        print("h_range:", qpu.properties["h_range"], "J_range:", qpu.properties["j_range"])
        #print(sampleset.data)
        #print(sampleset.info)
        #print(sampleset.first)



        # Check if the best solution found is actually a K-clique and print results
        dict_state = sampleset.first.sample.values()
        state = list(dict_state)
        if (constant == -sampleset.first.energy): 
            print(str(K) + '-clique found with RCS = ' + str(RCS[i]) + ':', state, '\n\n')

            groundStateSet = sampleset.lowest(atol=0.1)
            probability_of_success[select][i] = float(np.sum(groundStateSet.record.num_occurrences))/float(num_reads)

        else: 
            print('No '+ str(K) + '-clique found with RCS = ' + str(RCS[i]) + '\n\n')
            probability_of_success[select][i] = 0.0


        # Plot and save
        N0 = [i for i in G.nodes if state[i] == -1]
        N1 = [i for i in G.nodes if state[i] == 1]
        E0 = [(i,j) for i,j in G.edges if (state[i] == -1 or state[j] == -1)]
        E1 = [(i,j) for i,j in G.edges if (state[i] == 1 and state[j] == 1)]

        plt.figure()
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, nodelist = N0, node_color='red')
        nx.draw_networkx_nodes(G, pos, nodelist = N1, node_color='blue')
        nx.draw_networkx_edges(G, pos, edgelist = E0, style='dashdot', alpha=0.5, width=3)
        nx.draw_networkx_edges(G, pos, edgelist = E1, style='solid', width=3)
        nx.draw_networkx_labels(G, pos)

        filename = "K-clique " + "(QPU: " + str(select) + ").png"
        plt.savefig(filename, bbox_inches='tight')


plt.figure()
xAxis = RCS
plt.xlabel("Relative chain strength")
plt.ylabel("Probability of success")
plt.ylim([0,1])
plt.plot(xAxis, probability_of_success[0], color='blue', label='DW_6000Q_6')
# plt.plot(xAxis, probability_of_success[1], color='red', label='Advantage_system4.1')
# plt.legend(loc='best')
filename = "Probabilty of success for different chain strengths.png"
plt.savefig(filename, bbox_inches='tight')


filename = "Success rate.txt"
with open(filename, 'w') as file:
    for i in range(num_RCS):
        file.write(str(RCS[i]) + ' ' + str(probability_of_success[0][i]) + ' ' + str(probability_of_success[1][i]))


