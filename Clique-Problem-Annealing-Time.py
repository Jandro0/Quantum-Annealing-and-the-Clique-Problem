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
import dwave.inspector


# Graph parameters
# G = fileToNetwork("graph1.txt")
G = nx.fast_gnp_random_graph(7, 0.5)
K = nx.graph_clique_number(G) #Size of the clique we are searching
print("Clique number: " + str(K))
nv = nx.number_of_nodes(G)
ne = nx.number_of_edges(G)


# Quantum parameters 
min_annealing_time = 20.
max_annealing_time = 20.
num_annealing_time = 1
annealing_steps = np.linspace(np.log10(min_annealing_time), np.log10(max_annealing_time), num_annealing_time, dtype=float)
annealing_times = np.power(10, annealing_steps)
min_RCS = 0.1
max_RCS = 1.0
num_RCS = 20
RCS = np.linspace(min_RCS, max_RCS, num_RCS)
probability_of_success = np.empty((2, num_annealing_time))
num_reads = 100
B = 1.0
A = (K + 1)*B



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

# Compute the maximum strength 
max_strength = max(J.values())



# Run the annealing with the desired sampler
for i in range(num_annealing_time):
    for select in [0, 1]:
        if (select == 0):
            qpu = DWaveSampler(solver={'topology__type': 'chimera'}, auto_scale=True)
            sampler = EmbeddingComposite(qpu)
            sampleset = sampler.sample_ising(h, J,
                                            num_reads=num_reads,
                                            annealing_time=annealing_times[i],
                                            label='Test - Clique Problem')
            #dwave.inspector.show(sampleset)
        elif (select == 1):
            qpu = DWaveSampler(solver={'topology__type': 'pegasus'}, auto_scale=True)
            sampler = EmbeddingComposite(qpu)
            sampleset = sampler.sample_ising(h, J,                
                                            num_reads=num_reads,
                                            annealing_time=annealing_times[i],
                                            label='Test - Clique Problem')
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
        print(qpu.properties["h_range"], qpu.properties["j_range"])
        #print(sampleset.data)
        #print(sampleset.info)
        #print(sampleset.first)



        # Check if the best solution found is actually a K-clique and print results
        state = sampleset.record[0][0]
        if (constant == -sampleset.first.energy): 
            print(str(K) + '-clique found with annealing_time = ' + str(annealing_times[i]) + ':', state, '\n\n')

            groundStateSet = sampleset.lowest(atol=2.0)
            probability_of_success[select][i] = float(np.sum(groundStateSet.record.num_occurrences))/float(num_reads)

        else: 
            print('No '+ str(K) + '-clique found with annealing_time = ' + str(annealing_times[i]) + '\n\n')
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
xAxis = annealing_times
plt.xlabel("Annealing time (microseconds)")
plt.ylabel("Probability of success")
plt.ylim([0,1])
plt.plot(xAxis, probability_of_success[0], color='blue', label='DW_6000Q_6')
plt.plot(xAxis, probability_of_success[1], color='red', label='Advantage_system4.1')
plt.legend(loc='best')
filename = "Probabilty of success for different annealing times.png"
plt.savefig(filename, bbox_inches='tight')


