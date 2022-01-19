# This file includes functions that implement the desired schedules and 
# code to represent figures




import numpy as np
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'serif',
        'serif' : ['Computer Modern Roman'],
        'size'   : 12}
matplotlib.rc('font', **font)
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
from scipy.interpolate import interp1d
import networkx as nx
from collections import defaultdict
from matplotlib.collections import LineCollection
from graph_things import fileToNetwork


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
    









# nv = [30, 40]
# number_of_nv = len(nv)
# annealing_steps = np.linspace(np.log10(1.0), np.log10(1999.0), 21, dtype=float)
# annealing_times = np.power(10, annealing_steps)
# success_rate_DW = np.empty((number_of_nv, 21))
# deviation_DW = np.empty((number_of_nv, 21))
# success_rate_Advantage = np.empty((number_of_nv, 21))
# deviation_Advantage = np.empty((number_of_nv, 21))
# linewidth_factor = 300

# for j in range(number_of_nv):
#     i = 0
#     with open("Success rate annealing times DW (" + str(nv[j]) + "-0.8).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate_DW[j][i] = content[1]
#             deviation_DW[j][i] = content[2]
#             i += 1
    
#     i = 0
#     with open("Success rate annealing times Advantage (" + str(nv[j]) + "-0.8).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate_Advantage[j][i] = content[1]
#             deviation_Advantage[j][i] = content[2]
#             i += 1

# fig, axs = plt.subplots(1, number_of_nv)
# for j in range(number_of_nv):
#     points = np.array([annealing_times, success_rate_DW[j]]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, linewidths=linewidth_factor*deviation_DW[j], color='blue', alpha=0.1)
#     axs[j].add_collection(lc)
#     points = np.array([annealing_times, success_rate_Advantage[j]]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, linewidths=linewidth_factor*deviation_Advantage[j], color='red', alpha=0.1)
#     axs[j].add_collection(lc)
#     axs[j].set_xlim(1.0, 2000.0)
#     axs[j].set_ylim(0.0, 1.0)
#     axs[j].set_xscale('log')
#     axs[j].set_title(str(nv[j]) + " vertices", y=1.0, x=0.80, pad=-14, fontsize=13)
#     axs[j].errorbar(annealing_times, success_rate_DW[j], yerr=deviation_DW[j], label='DW_2000Q_6', fmt='o', markersize=2, capsize=1.5, alpha = 1.0, color='blue')
#     axs[j].errorbar(annealing_times, success_rate_Advantage[j], yerr=deviation_Advantage[j], label='Advantage_system4.1', fmt='o', markersize=2, capsize=1.5, alpha = 1.0, color='red')

# for ax in axs.flat:
#     ax.set(xlabel='Annealing time (µs)', ylabel='Success rate')
# for ax in axs.flat:
#     ax.label_outer()
# fig.set_size_inches(7, 2.25, forward=True)
# fig.subplots_adjust(wspace=0.05, hspace=0.12)
# plt.savefig("Success rate vs annealing_time (nv = " + str(nv) + ").png", bbox_inches='tight', dpi = 1000)
# plt.show()



# nv_vector = [15, 20, 25, 30, 35, 40]
# max_success_rate_DW = np.empty((3, 6))
# extended_max_success_rate_DW = np.empty(10)
# deviation_DW = np.empty((3, 6))
# max_success_rate_Advantage = np.empty((3, 6))
# extended_max_success_rate_Advantage = np.empty(10)
# deviation_Advantage = np.empty((3, 6))
# for j in range(6):
#     nv = nv_vector[j]
#     RCS = np.linspace(0.0, 1.0, 21)
#     success_rate = np.empty((3, 21))
#     deviation = np.empty((3, 21))
#     i = 0
#     with open("Success rate DW (" + str(nv) + "-0.2).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate[0][i] = content[1]
#             deviation[0][i] = content[2]
#             i += 1
    
#     i = 0
#     with open("Success rate DW (" + str(nv) + "-0.5).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate[1][i] = content[1]
#             deviation[1][i] = content[2]
#             i += 1
    
#     i = 0
#     with open("Success rate DW (" + str(nv) + "-0.8).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate[2][i] = content[1]
#             deviation[2][i] = content[2]
#             i += 1
    
#     for k in range(3):
#         max_success_rate_DW[k][j] = max(success_rate[k])
#         deviation_DW[k][j] = deviation[k][np.argmax(success_rate[k])]
    
#     i = 0
#     with open("Success rate Advantage (" + str(nv) + "-0.2).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate[0][i] = content[1]
#             deviation[0][i] = content[2]
#             i += 1
    
#     i = 0
#     with open("Success rate Advantage (" + str(nv) + "-0.5).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate[1][i] = content[1]
#             deviation[1][i] = content[2]
#             i += 1
    
#     i = 0
#     with open("Success rate Advantage (" + str(nv) + "-0.8).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate[2][i] = content[1]
#             deviation[2][i] = content[2]
#             i += 1
#     for k in range(3):
#         max_success_rate_Advantage[k][j] = max(success_rate[k])
#         deviation_Advantage[k][j] = deviation[k][np.argmax(success_rate[k])]


# for k in range(6):
#     extended_max_success_rate_DW[k] = max_success_rate_DW[2][k]
#     extended_max_success_rate_Advantage[k] = max_success_rate_Advantage[2][k]

# extended_max_success_rate_DW[6] = 0.002
# extended_max_success_rate_DW[7] = 0.0
# extended_max_success_rate_DW[8] = 0.0
# extended_max_success_rate_DW[9] = 0.0
# extended_max_success_rate_Advantage[6] = 0.008
# extended_max_success_rate_Advantage[7] = 0.024
# extended_max_success_rate_Advantage[8] = 0.0
# extended_max_success_rate_Advantage[9] = 0.02


# nv_vector2 = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# fig, axs = plt.subplots()
# axs.set_xlim(12, 63)
# axs.set_ylim(0.0, 1.0)
# axs.errorbar(nv_vector, max_success_rate_DW[0], fmt='o', markersize=3, capsize=2, color='blue', label='DW_2000Q_6, p = 0.2', alpha=0.7)
# axs.errorbar(nv_vector, max_success_rate_DW[1], fmt='o', markersize=3, capsize=2, color='orange', label='DW_2000Q_6, p = 0.5', alpha=0.7)
# axs.errorbar(nv_vector2, extended_max_success_rate_DW , fmt='o', markersize=3, capsize=2, color='green', label='DW_2000Q_6, p = 0.8', alpha=0.7)
# axs.errorbar(nv_vector, max_success_rate_Advantage[0], fmt='x', markersize=3, capsize=2, color='blue', label='Advantage, p = 0.2', alpha=0.7)
# axs.errorbar(nv_vector, max_success_rate_Advantage[1], fmt='x', markersize=3, capsize=2, color='orange', label='Advantage, p = 0.5', alpha=0.7)
# axs.errorbar(nv_vector2, extended_max_success_rate_Advantage, fmt='x', markersize=3, capsize=2, color='green', label='Advantage, p = 0.8', alpha=0.7)
# axs.set(xlabel='Number of vertices', ylabel='Maximum success rate')

# fig.set_size_inches(5.5, 3.2, forward=True)
# plt.legend(loc='best')
# plt.savefig("Success rate vs nvertices.png", bbox_inches='tight', dpi=1000)
# plt.show()



# fig, axs = plt.subplots(1, number_of_nv)
# for j in range(number_of_nv):
#     points = np.array([annealing_times, success_rate_DW[j]]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, linewidths=linewidth_factor*deviation_DW[j], color='blue', alpha=0.1)
#     axs[j].add_collection(lc)
#     points = np.array([annealing_times, success_rate_Advantage[j]]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, linewidths=linewidth_factor*deviation_Advantage[j], color='red', alpha=0.1)
#     axs[j].add_collection(lc)
#     axs[j].set_xlim(1.0, 2000.0)
#     axs[j].set_ylim(0.0, 1.0)
#     axs[j].set_xscale('log')
#     axs[j].set_title(str(nv[j]) + " vertices", y=1.0, x=0.84, pad=-14, fontsize=10)
#     axs[j].errorbar(annealing_times, success_rate_DW[j], yerr=deviation_DW[j], label='DW_2000Q_6', fmt='o', markersize=2, capsize=1.5, alpha = 1.0, color='blue')
#     axs[j].errorbar(annealing_times, success_rate_Advantage[j], yerr=deviation_Advantage[j], label='Advantage_system4.1', fmt='o', markersize=2, capsize=1.5, alpha = 1.0, color='red')

# for ax in axs.flat:
#     ax.set(xlabel='Annealing time (µs)', ylabel='Success rate')
# for ax in axs.flat:
#     ax.label_outer()
# fig.set_size_inches(7, 2.25, forward=True)
# fig.subplots_adjust(wspace=0.05, hspace=0.12)
# plt.savefig("Success rate vs annealing_time (nv = " + str(nv) + ").png", bbox_inches='tight', dpi = 1000)
# plt.show()




# nv = [15, 20, 30, 40]
# number_of_nv = len(nv)
# RCS = np.linspace(0.0, 1.0, 21)
# success_rate_DW = np.empty((number_of_nv, 3, 21))
# success_rate_Advantage = np.empty((number_of_nv, 3, 21))
# deviation_DW = np.empty((number_of_nv, 3, 21))
# deviation_Advantage = np.empty((number_of_nv, 3, 21))
# linewidth_factor = 250

# for j in range(number_of_nv):
#     i = 0
#     with open("Success rate DW (" + str(nv[j]) + "-0.2).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate_DW[j][0][i] = content[1]
#             deviation_DW[j][0][i] = content[2]
#             i += 1
    
#     i = 0
#     with open("Success rate DW (" + str(nv[j]) + "-0.5).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate_DW[j][1][i] = content[1]
#             deviation_DW[j][1][i] = content[2]
#             i += 1
    
#     i = 0
#     with open("Success rate DW (" + str(nv[j]) + "-0.8).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate_DW[j][2][i] = content[1]
#             deviation_DW[j][2][i] = content[2]
#             i += 1
            
    
#     i = 0
#     with open("Success rate Advantage (" + str(nv[j]) + "-0.2).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate_Advantage[j][0][i] = content[1]
#             deviation_Advantage[j][0][i] = content[2]
#             i += 1
    
#     i = 0
#     with open("Success rate Advantage (" + str(nv[j]) + "-0.5).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate_Advantage[j][1][i] = content[1]
#             deviation_Advantage[j][1][i] = content[2]
#             i += 1
    
#     i = 0
#     with open("Success rate Advantage (" + str(nv[j]) + "-0.8).txt") as file:
#         for line in file:
#             content = line.split()
#             success_rate_Advantage[j][2][i] = content[1]
#             deviation_Advantage[j][2][i] = content[2]
#             i += 1

# fig, axs = plt.subplots(number_of_nv, 2)
# for j in range(number_of_nv):
#     points = np.array([RCS, success_rate_DW[j][0]]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, linewidths=linewidth_factor*deviation_DW[j][0], color='blue', alpha=0.1)
#     axs[j, 0].add_collection(lc)
#     points = np.array([RCS, success_rate_DW[j][1]]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, linewidths=linewidth_factor*deviation_DW[j][1], color='orange', alpha=0.1)
#     axs[j, 0].add_collection(lc)
#     points = np.array([RCS, success_rate_DW[j][2]]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, linewidths=linewidth_factor*deviation_DW[j][2], color='green', alpha=0.1)
#     axs[j, 0].add_collection(lc)
#     axs[j, 0].set_xlim(0.0, 1.0)
#     axs[j, 0].set_ylim(0.0, 1.0)
#     axs[j, 0].set_title(str(nv[j]) + " vertices", y=1.0, x=0.80, pad=-14, fontsize=10)
#     axs[j, 0].errorbar(RCS, success_rate_DW[j][0], yerr=deviation_DW[j][0], label='p = 0.2', fmt='o', markersize=1, capsize=1.2, alpha = 1.0, color='blue')
#     axs[j, 0].errorbar(RCS, success_rate_DW[j][1], yerr=deviation_DW[j][1], label='p = 0.5', fmt='o', markersize=1, capsize=1.2, alpha = 1.0, color='orange')
#     axs[j, 0].errorbar(RCS, success_rate_DW[j][2], yerr=deviation_DW[j][2], label='p = 0.8', fmt='o', markersize=1, capsize=1.2, alpha = 1.0, color='green')
    
#     points = np.array([RCS, success_rate_Advantage[j][0]]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, linewidths=linewidth_factor*deviation_Advantage[j][0], color='blue', alpha=0.1)
#     axs[j, 1].add_collection(lc)
#     points = np.array([RCS, success_rate_Advantage[j][1]]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, linewidths=linewidth_factor*deviation_Advantage[j][1], color='orange', alpha=0.1)
#     axs[j, 1].add_collection(lc)
#     points = np.array([RCS, success_rate_Advantage[j][2]]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, linewidths=linewidth_factor*deviation_Advantage[j][2], color='green', alpha=0.1)
#     axs[j, 1].add_collection(lc)
#     axs[j, 1].set_xlim(0.0, 1.0)
#     axs[j, 1].set_ylim(0.0, 1.0)
#     axs[j, 1].set_title(str(nv[j]) + " vertices", y=1.0, x=0.80, pad=-14, fontsize=10)
#     axs[j, 1].errorbar(RCS, success_rate_Advantage[j][0], yerr=deviation_Advantage[j][0], label='p = 0.2', fmt='o', markersize=1, capsize=1.2, alpha = 1.0, color='blue')
#     axs[j, 1].errorbar(RCS, success_rate_Advantage[j][1], yerr=deviation_Advantage[j][1], label='p = 0.5', fmt='o', markersize=1, capsize=1.2, alpha = 1.0, color='orange')
#     axs[j, 1].errorbar(RCS, success_rate_Advantage[j][2], yerr=deviation_Advantage[j][2], label='p = 0.8', fmt='o', markersize=1, capsize=1.2, alpha = 1.0, color='green')

# axs[0, 0].text(0.0, 1.05, "DW_2000Q_6")
# axs[0, 1].text(0.0, 1.05, "Advantage_system4.1")
# for ax in axs.flat:
#     ax.set(xlabel='RCS', ylabel='Success rate')
# for ax in axs.flat:
#     ax.label_outer()
# fig.subplots_adjust(wspace=0.12, hspace=0.16)
# fig.set_size_inches(6, 6, forward=True)
# plt.savefig("Success rate vs RCS (nv = " + str(nv) + ").png", bbox_inches='tight', dpi = 1000)
# plt.show()






# RCS = np.empty(41)
# success_rate = np.empty(41)
# deviation = np.empty(41)
# i = 0
# with open("Success rate DW (500).txt") as file: 
#     for line in file:
#         content = line.split()
#         RCS[i] = float(content[0])
#         success_rate[i] = float(content[1])
#         deviation[i] = float(content[2])
#         i += 1

# success_rate_advantage = np.linspace(0.912, 0.912, 41)
# deviation_advantage = np.linspace(0.01267, 0.01267, 41)

# RCS_sim = np.empty(100)
# success_rate_sim = np.empty(100)
# i = 0
# with open("Success rate DW (simulation, 5.5 ns).txt") as file:
#     for line in file:
#         content = line.split()
#         RCS_sim[i] = float(content[0])
#         success_rate_sim[i] = float(content[1])
#         i += 1



# plt.figure()
# plt.xlabel("Relative chain strength")
# plt.ylabel("Success rate")
# plt.ylim([0,1])
# plt.xlim([0,2.0])
# plt.errorbar(RCS, success_rate, yerr=deviation, label='DW_2000Q_6 with t_f=20 µs', fmt='o', markersize=3, capsize=2, alpha = 1.0)
# plt.plot(RCS_sim, success_rate_sim, label='simulation with t_f=5.5 ns', linestyle = 'dashed')
# #plt.errorbar(RCS, success_rate_advantage, yerr=deviation_advantage, label='Advantage_system4.1', markersize=3, capsize=2, alpha = 0.7)
# plt.legend(loc='best')

# plt.savefig("Success rate simple graph (5.5 ns).png", bbox_inches='tight', dpi = 1000)







# G = fileToNetwork("graph5.txt")
# N0 = [2, 3]
# N1 = [0, 1, 4]
# E0 = [(1, 2), (3, 4), (2, 4)]
# E1 = [(0, 1), (0, 4), (1, 4)]
# node_labels = {0:0, 1:1, 2:2, 3:3, 4:4}

# plt.figure()
# pos = nx.spring_layout(G)
# nx.draw_networkx_nodes(G, pos, nodelist = N0, node_color='red')
# nx.draw_networkx_nodes(G, pos, nodelist = N1, node_color='black')
# nx.draw_networkx_edges(G, pos, edgelist = E0, style='dashdot', alpha=0.5, width=3)
# nx.draw_networkx_edges(G, pos, edgelist = E1, style='solid', width=3)
# # nx.draw_networkx_labels(G, pos, labels=node_labels)
# plt.savefig("triangle.png", bbox_inches='tight', dpi=1000)


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
    
    
    
    
    
    
    