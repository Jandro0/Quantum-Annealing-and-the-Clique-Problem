import math
import time
import sys
from random import uniform
import numpy as np
import scipy.integrate
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
font = {'family' : 'serif',
        'serif' : ['Computer Modern Roman'],
        'size'   : 10}
matplotlib.rc('font', **font)
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
from collections import defaultdict
from graph_things import graph, createGraph, networkToGraph, fileToNetwork, binaryToDecimal
from graph_things import networkToFile, decimalToBinary, generalHamiltonian, evaluateHamiltonian
from evolution import evolutionCN2, evolutionCN3, evolutionCN4, evolutionCN5, evolutionCN6
from evolution import evolutionABCN2, evolutionABCN3, evolutionABCN4, evolutionABCN5, evolutionABCN6
from evolution import adiabaticEvolution, adiabaticEvolutionAB, spectra, epsilon
from evolution import evolutionRK2, evolutionRK3, evolutionRK4, evolutionRK5, evolutionRK6
from evolution import evolutionABRK2, evolutionABRK3, evolutionABRK4, evolutionABRK5, evolutionABRK6
from evolution import makeInitialHamiltonian, makeInitialHamiltonian2, makeFinalHamiltonian, groundState
import functions as fun
import networkx as nx
from evolution import evolutionABRK62, evolutionABRK22






#----------------Simulation of the logical system for random instances---------------#
# nv_min = 3  #Minimum number of vertices
# nv_max = 5 #Maximum number of vertices
# p = 0.7  #Probability that there is an edge joining two vertices
# t_f = 5.0  #Time of the evolution
# delta_t = 0.01  #Step in the Crank-Nicolson/Runge-Kutta algorithm
# number_of_experiments = 1  #Number of simulations for each size
# simulation_times = np.empty((5, number_of_experiments))  #Probabilities of success at a given time
# gaps = np.empty((nv_max - nv_min + 1, number_of_experiments)) #Gaps for different graphs
# inversesqgaps = np.empty((nv_max - nv_min + 1, number_of_experiments)) #Inverse squares of the gaps
# meanGap = np.empty(nv_max - nv_min + 1) #Mean minimum gap
# deviationGap = np.empty(nv_max - nv_min + 1) #Standard deviation of the minimum gap
# meanTime = np.empty((5, nv_max - nv_min + 1))  #Mean probabilities for each size
# deviationTime = np.empty((5, nv_max - nv_min + 1))  #Standard deviation of the probabilities for each size
# successProbabilities = np.zeros((nv_max - nv_min + 1, number_of_experiments)) #List of success probabilities

# for nv in range(nv_min, nv_max + 1):
#     for experiment in range(number_of_experiments):
#         #Create graph
#         networkG = nx.fast_gnp_random_graph(nv, p)
#         G = networkToGraph(networkG)
        
        
#         #Create Hamiltonian matrices and initial state
#         c = [0]
#         K = nx.graph_clique_number(networkG)
#         beta = 1.0
#         alpha = (K + 1)*beta
#         H = graph.makeIsingHamiltonian(G, K, alpha, beta, c)
#         dim = len(H)
#         t_f = 5.0
#         delta_t = 0.01
#         number_of_eigenstates = 8
#         number_of_overlaps = 50
#         iterations = int(t_f/delta_t)
#         gs = groundState(dim, H)
#         H1 = makeFinalHamiltonian(dim, H)
#         H0 = makeInitialHamiltonian(len(G.adjacency))
#         successProbabilityRK = [np.empty(0)]
#         successProbabilityRK3 = [np.empty(0)]
#         successProbabilityRK6 = [np.empty(0)]
#         successProbabilityCN = [np.empty(0)]
#         successProbabilityODE = [np.empty(0)]
#         adiabatic = [np.empty(0)]
#         energy = [np.empty([number_of_eigenstates, number_of_overlaps])]
#         overlap = [np.empty([number_of_eigenstates, number_of_overlaps])]
#         overlap1 = [np.empty(0)]
#         A = fun.linearA
#         B = fun.linearB
        
        
        # #Make evolution
        # adiabaticEvolutionAB(dim, H0, H1, A, B, adiabatic, gs, t_f)
        
        # psi = np.ones(dim, dtype = complex)
        # initialT = time.time()
        # psi = evolutionABRK2(dim, H0, H1, psi, A, B, successProbabilityRK, gs, t_f, delta_t)
        # finalT = time.time()
        # simulation_times[0][experiment] = finalT - initialT
        
        # psi = np.ones(dim, dtype = complex)
        # initialT = time.time()
        # psi = evolutionABCN2(dim, H0, H1, psi, A, B, successProbabilityCN, gs, t_f, delta_t)
        # finalT = time.time()
        # simulation_times[1][experiment] = finalT - initialT
        
        # psi = np.ones(dim, dtype = complex)
        # initialT = time.time()
        # psi = evolutionABRK22(dim, H0, H1, psi, A, B, successProbabilityRK3, gs, t_f, delta_t)
        # finalT = time.time()
        # simulation_times[2][experiment] = finalT - initialT
        
        # psi = np.ones(dim, dtype = complex)
        # initialT = time.time()
        # psi = evolutionABRK62(dim, H0, H1, psi, A, B, successProbabilityRK6, gs, t_f, delta_t)
        # finalT = time.time()
        # simulation_times[3][experiment] = finalT - initialT
        
        # psi = np.ones(dim, dtype = complex)
        # t = np.linspace(0, t_f, int(t_f/delta_t))
        # iteration = 0
        # initialT = time.time()
        # r = scipy.integrate.ode(fun.f).set_integrator('zvode', method='bdf', with_jacobian=False)
        # r.set_initial_value(psi, 0).set_f_params(H0, H1, A, B, t_f)
        # while r.successful() and r.t < t_f - 2*delta_t:
        #     psi_t = r.integrate(r.t + delta_t)
            
        #     if (iteration%30 == 0):
        #         successProbabilityODE[0] = np.append(successProbabilityODE[0], 0.)
        #         for i in gs:
        #             successProbabilityODE[0][-1] += np.real(psi_t[i])*np.real(psi_t[i]) + np.imag(psi_t[i])*np.imag(psi_t[i])
               
        #         successProbabilityODE[0][-1] = successProbabilityODE[0][-1]/float(dim)
        # finalT = time.time()
        # simulation_times[4][experiment] = finalT - initialT
        
        # psi = np.ones(dim, dtype = complex)
        # psi = evolutionABRK6(dim, H0, H1, psi, A, B, energy, overlap, t_f, delta_t, number_of_overlaps, number_of_eigenstates)
        
        # psi = np.ones(dim, dtype = complex)
        # psi = evolutionABRK5(dim, H0, H1, psi, A, B, overlap1, gs, t_f, delta_t)
        
        
        # # Probabilities
        # probability = np.empty(dim)
        # for k in range(dim):
        #     probability[k] = np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k])
        # probability = probability / float(dim)
        
        # for i in gs:
        #     successProbabilities[nv - nv_min][experiment] += probability[i]
        
        
        # probability =  np.amax(H)*probability
        # xAxis1 = np.linspace(0, dim - 1,  dim)
        # xAxis2a = np.linspace(0, t_f, len(successProbabilityRK[0]))
        # xAxis2b = np.linspace(0, t_f, len(adiabatic[0]))
        # xAxis2c = np.linspace(0, t_f, len(successProbabilityCN[0]))
        # xAxis2d = np.linspace(0, t_f, len(successProbabilityRK3[0]))
        # xAxis2e = np.linspace(0, t_f, len(successProbabilityRK6[0]))
        # xAxis2f = np.linspace(0, t_f, len(successProbabilityODE[0]))
        # xAxis3 = np.linspace(0, t_f, number_of_overlaps)
        
        
        
        
        # Draw stuff
        # plot1 = plt.figure(1)
        # plt.plot(xAxis1, H)
        # plt.plot(xAxis1, probability)
        
        # plot2 = plt.figure(2)
        # plt.xlim([0,t_f])
        # plt.ylim([0,1])
        # plt.plot(xAxis2a, successProbabilityRK[0], 0.1, color = 'blue')
        # plt.plot(xAxis2c, successProbabilityCN[0], 0.1, color = 'red')
        # plt.plot(xAxis2d, successProbabilityRK3[0], 0.1, color = 'orange')
        # plt.plot(xAxis2e, successProbabilityRK6[0], 0.1, color = 'cyan')
        # plt.plot(xAxis2f, successProbabilityODE[0], 0.1, color = 'black')
        # plt.plot(xAxis2b, adiabatic[0], 0.1, color = 'green')
        
        
        
        
        
        # plot3 = plt.figure(3)
        # state = decimalToBinary(G.nv, gs[0])
        # N0 = [i for i in networkG.nodes if state[i] == 0]
        # N1 = [i for i in networkG.nodes if state[i] == 1]
        # E0 = [(i,j) for i,j in networkG.edges if (state[i] == 0 or state[j] == 0)]
        # E1 = [(i,j) for i,j in networkG.edges if (state[i] == 1 and state[j] == 1)]
        
        # pos = nx.spring_layout(networkG)
        # nx.draw_networkx_nodes(networkG, pos, nodelist = N0, node_color='red')
        # nx.draw_networkx_nodes(networkG, pos, nodelist = N1, node_color='blue')
        # nx.draw_networkx_edges(networkG, pos, edgelist = E0, style='dashdot', alpha=0.5, width=3)
        # nx.draw_networkx_edges(networkG, pos, edgelist = E1, style='solid', width=3)
        # nx.draw_networkx_labels(networkG, pos)
        
        
        
        # # Plot spectra over time 
        # plot4 = plt.figure(4)
        # plt.xlim(0, t_f)
        # # for i in range(number_of_eigenstates):
        # #     plt.plot(xAxis3, energy[0][i])
        # for i in range(number_of_eigenstates):
        #     plt.scatter(xAxis3, energy[0][i], s = 100*overlap[0][i])
        
        # divisions = 100
        # sVector = np.linspace(0, 1, divisions)
        # minimum_gap, energies = spectra(dim, H0, H1, A, B, gs, divisions, number_of_eigenstates)
        # gaps[nv - nv_min][experiment] = minimum_gap
        # inversesqgaps[nv - nv_min][experiment] = 1./(minimum_gap*minimum_gap)
        
        # plot5 = plt.figure(5)
        # plt.xlim(0, 1)
        # for i in range(number_of_eigenstates):
        #     plt.plot(sVector, energies[i], 1)
        
        
        
        # plot6 = plt.figure(6)
        # xAxis6 = np.linspace(0, t_f, len(overlap1[0]))
        # plot6 = plt.figure(6)
        # plt.xlim([0,t_f])
        # plt.ylim([0,1])
        # plt.plot(xAxis6, overlap1[0], 0.1)
        
        
        # plot7 = plt.figure(7)
        # xAxis7 = np.linspace(0, t_f, 1000)
        # plotA = A(xAxis7/t_f)
        # plotB = B(xAxis7/t_f)
        # plt.xlim([0,t_f])
        # plt.plot(xAxis7, plotA)
        # plt.plot(xAxis7, plotB)
        
        # plt.show()

    # for j in range(5):
    #     meanTime[j][nv - nv_min] = np.mean(simulation_times[j])
    #     deviationTime[j][nv - nv_min] = np.std(simulation_times[j])
    # meanGap[nv - nv_min] = np.mean(gaps[nv - nv_min])
    # deviationGap[nv - nv_min] = np.std(gaps[nv - nv_min])


# plot8 = plt.figure(8)
# xAxis = np.linspace(nv_min, nv_max, nv_max - nv_min + 1)
# plt.xlim([nv_min, nv_max])
# plt.xlabel("Size of the graph")
# plt.ylabel("Mean simulation time (s)")
# plt.xticks(np.arange(nv_min, nv_max + 1, step=1, dtype=int))
# plt.plot(xAxis, meanTime[0], color = 'blue', label='RK4')
# plt.plot(xAxis, meanTime[1], color = 'red', label='CN')
# plt.plot(xAxis, meanTime[2], color = 'orange', label='RK3')
# plt.plot(xAxis, meanTime[3], color = 'cyan', label='RK6')
# plt.plot(xAxis, meanTime[4], color = 'black', label='ODE')
# plt.legend(loc='best')

# plot9 = plt.figure(9)
# xAxis = np.linspace(nv_min, nv_max, nv_max - nv_min + 1)
# plt.xlim([nv_min, nv_max])
# plt.xlabel("Size of the graph")
# plt.ylabel("Mean minimum gap")
# plt.plot(xAxis, meanGap)

# plot10 = plt.figure(10)
# plt.xlabel("Size of the graph")
# plt.ylabel("Mean minimum gap")
# for nv in range(nv_min, nv_max + 1):
#     xAxis = np.linspace(nv, nv, number_of_experiments)
#     plt.scatter(xAxis, gaps[nv - nv_min], s = 20)

# plot11 = plt.figure(11)
# plt.xlabel("Minimum gap")
# plt.ylabel("Success probability")
# plt.ylim([0,1])
# for nv in range(nv_min, nv_max + 1):
#     plt.scatter(gaps[nv - nv_min], successProbabilities[nv - nv_min], s = 20)
    
# plot12 = plt.figure(12)
# plt.xlabel("1/(gap)^2")
# plt.ylabel("Success probability")
# plt.ylim([0,1])
# for nv in range(nv_min, nv_max + 1):
#     plt.scatter(inversesqgaps[nv - nv_min], successProbabilities[nv - nv_min], s = 20)
    
    
# plt.show()




#---------------Simulation of the logical system for a fixed graph------------------#
# #Create graph
# G = createGraph("graph5.txt")
# networkG = graph.toNetwork(G)


# #Create Hamiltonian matrices and initial state
# c = [0]
# K = nx.graph_clique_number(networkG)
# alpha = 1.0
# beta = 2*alpha
# H = graph.makeIsingHamiltonian(G, alpha, beta, c)
# dim = len(H)
# t_f = 2.0
# delta_t = 0.01
# number_of_eigenstates = 6
# number_of_overlaps = 50
# iterations = int(t_f/delta_t)
# gs = groundState(dim, H)
# H1 = makeFinalHamiltonian(dim, H)
# H0 = makeInitialHamiltonian(len(G.adjacency))
# successProbabilityRK = [np.empty(0)]
# successProbabilityRK3 = [np.empty(0)]
# successProbabilityRK6 = [np.empty(0)]
# successProbabilityCN = [np.empty(0)]
# successProbabilityODE = [np.empty(0)]
# adiabatic = [np.empty(0)]
# energy = [np.empty([number_of_eigenstates, number_of_overlaps])]
# overlap = [np.empty([number_of_eigenstates, number_of_overlaps])]
# overlap1 = [np.empty(0)]
# A = fun.linearA
# B = fun.linearB


    
# #Make evolution
# adiabaticEvolutionAB(dim, H0, H1, A, B, adiabatic, gs, t_f)

# # psi = np.ones(dim, dtype = complex)
# # initialT = time.time()
# # psi = evolutionABRK2(dim, H0, H1, psi, A, B, successProbabilityRK, gs, t_f, delta_t)
# # finalT = time.time()
# # print("RK4: " + str(finalT - initialT))

# psi = np.ones(dim, dtype = complex)
# initialT = time.time()
# psi = evolutionABCN2(dim, H0, H1, psi, A, B, successProbabilityCN, gs, t_f, delta_t)
# finalT = time.time()
# # print("CN: " + str(finalT - initialT))

# # psi = np.ones(dim, dtype = complex)
# # initialT = time.time()
# # psi = evolutionABRK22(dim, H0, H1, psi, A, B, successProbabilityRK3, gs, t_f, delta_t)
# # finalT = time.time()
# # print("RK3: " + str(finalT - initialT))

# # psi = np.ones(dim, dtype = complex)
# # initialT = time.time()
# # psi = evolutionABRK62(dim, H0, H1, psi, A, B, successProbabilityRK6, gs, t_f, delta_t)
# # finalT = time.time()
# # print("RK6: " + str(finalT - initialT))

# # psi = np.ones(dim, dtype = complex)
# # t = np.linspace(0, t_f, int(t_f/delta_t))
# # iteration = 0
# # initialT = time.time()
# # r = scipy.integrate.ode(fun.f).set_integrator('zvode', method='Adams', with_jacobian=False)
# # r.set_initial_value(psi, 0).set_f_params(H0, H1, A, B, t_f)
# # while r.successful() and r.t < t_f - 2*delta_t:
# #     psi_t = r.integrate(r.t + delta_t)
    
# #     if (iteration%30 == 0):
# #         successProbabilityODE[0] = np.append(successProbabilityODE[0], 0.)
# #         for i in gs:
# #             successProbabilityODE[0][-1] += np.real(psi_t[i])*np.real(psi_t[i]) + np.imag(psi_t[i])*np.imag(psi_t[i])
       
# #         successProbabilityODE[0][-1] = successProbabilityODE[0][-1]/float(dim)
# # finalT = time.time()
# # print("ODE: " + str(finalT - initialT))

# psi = np.ones(dim, dtype = complex)
# psi = evolutionABCN6(dim, H0, H1, psi, A, B, energy, overlap, t_f, delta_t, number_of_overlaps, number_of_eigenstates)

# psi = np.ones(dim, dtype = complex)
# psi = evolutionABCN5(dim, H0, H1, psi, A, B, overlap1, gs, t_f, delta_t)






# # Plot final Hamiltonian spectra and final state
# probability = np.empty(dim)
# for k in range(dim):
#     probability[k] = np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k])
# probability = probability / float(dim)
# probability =  np.amax(H)*probability

# plot1 = plt.figure(1)
# xAxis1 = np.linspace(0, dim - 1,  dim)
# plt.plot(xAxis1, H)
# plt.plot(xAxis1, probability)



# plot2 = plt.figure(2)
# state = decimalToBinary(G.nv, gs[0])
# N0 = [i for i in networkG.nodes if state[i] == 0]
# N1 = [i for i in networkG.nodes if state[i] == 1]
# E0 = [(i,j) for i,j in networkG.edges if (state[i] == 0 or state[j] == 0)]
# E1 = [(i,j) for i,j in networkG.edges if (state[i] == 1 and state[j] == 1)]

# pos = nx.spring_layout(networkG)
# nx.draw_networkx_nodes(networkG, pos, nodelist = N0, node_color='red')
# nx.draw_networkx_nodes(networkG, pos, nodelist = N1, node_color='blue')
# nx.draw_networkx_edges(networkG, pos, edgelist = E0, style='dashdot', alpha=0.5, width=3)
# nx.draw_networkx_edges(networkG, pos, edgelist = E1, style='solid', width=3)
# nx.draw_networkx_labels(networkG, pos)



# # Plot spectra over time 
# plot3 = plt.figure(3)
# xAxis3 = np.linspace(0, 1, number_of_overlaps)
# plt.xlim(0, 1)
# # for i in range(number_of_eigenstates):
# #     plt.plot(xAxis3, energy[0][i])
# for i in range(number_of_eigenstates):
#     plt.scatter(xAxis3, energy[0][i], s = 100*overlap[0][i])

# divisions = 1000
# sVector = np.linspace(0, 1, divisions)
# minimum_gap, energies = spectra(dim, H0, H1, A, B, gs, divisions, number_of_eigenstates)
# max_epsilon = epsilon(dim, H0, H1, A, B, gs, divisions, 0.001)

# plot4 = plt.figure(4)
# plt.xlim(0, 1)
# for i in range(number_of_eigenstates):
#     plt.plot(sVector, energies[i], 1)



# # Plot overlap with target states
# plot5 = plt.figure(5)
# xAxis2a = np.linspace(0, t_f, len(successProbabilityRK[0]))
# xAxis2b = np.linspace(0, t_f, len(successProbabilityCN[0]))
# xAxis2c = np.linspace(0, t_f, len(successProbabilityRK3[0]))
# xAxis2d = np.linspace(0, t_f, len(successProbabilityRK6[0]))
# xAxis2e = np.linspace(0, t_f, len(successProbabilityODE[0]))
# xAxis2f = np.linspace(0, t_f, len(adiabatic[0]))
# plt.xlim([0,t_f])
# plt.ylim([0,1])
# # plt.plot(xAxis2a, successProbabilityRK[0], 0.1, color = 'blue')
# plt.plot(xAxis2b, successProbabilityCN[0], 0.1, color = 'red')
# # plt.plot(xAxis2c, successProbabilityRK3[0], 0.1, color = 'orange')
# # plt.plot(xAxis2d, successProbabilityRK6[0], 0.1, color = 'cyan')
# # plt.plot(xAxis2e, successProbabilityODE[0], 0.1, color = 'black')
# plt.plot(xAxis2f, adiabatic[0], 0.1, color = 'green')


# # Plot overlap with the instantaneous ground state
# plot6 = plt.figure(6)
# xAxis6 = np.linspace(0, 1, len(overlap1[0]))
# plot6 = plt.figure(6)
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.xlabel("Annealing scaled time s=t/T")
# plt.ylabel("Overlap with the instantaneous ground state")
# plt.plot(xAxis6, overlap1[0], 0.1)





# # # Plot schedule functions A(s), B(s)
# # plot7 = plt.figure(7)
# # xAxis7 = np.linspace(0, t_f, 1000)
# # plotA = A(xAxis7/t_f)
# # plotB = B(xAxis7/t_f)
# # plt.xlim([0,t_f])
# # plt.plot(xAxis7, plotA)
# # plt.plot(xAxis7, plotB)

# # plt.show()





#-----------------Spectra of H1 and H(s) for varying chain strength-----------------#
# #Create graph 
# G = createGraph("graph5.txt")
# networkG = graph.toNetwork(G)


# #Parameters
# c = [0]
# K = nx.graph_clique_number(networkG)
# alpha = 1.0
# beta = 2*alpha
# min_RCS = 0.0
# max_RCS = 2.0
# num_RCS = 250
# RCS = np.linspace(min_RCS, max_RCS, num_RCS)
# t_f = 1.0
# delta_t = 0.005
# number_of_eigenstates = 8
# s_divisions = 2
# sVector = np.linspace(0, 1, s_divisions)
# spectra_of_H1 = np.empty((number_of_eigenstates, num_RCS))
# spectra_of_H = np.empty((number_of_eigenstates, s_divisions))
# spectra_of_Hr = np.empty((number_of_eigenstates, num_RCS))
# minimum_gap = np.empty(num_RCS)
# target_graph = fileToNetwork("chimera.txt")
# minor = {0: [2028], 2: [2027, 2029], 3: [2024], 1: [2031], 4: [1033]}
# number_of_physical_qubits = 0
# for i in range(len(minor)):
#     number_of_physical_qubits += len(minor[i])
# H0 = makeInitialHamiltonian(number_of_physical_qubits)
# A, B = fun.schedule("DW_2000Q_6.txt")

# for i in range(num_RCS):
#     H = graph.makeHamiltonianWithChains(G, target_graph, minor, alpha, beta, c, RCS[i])
#     dim = len(H)
#     gs = groundState(dim, H)
#     print(str(RCS[i]) + ': ' + str(gs))
#     H1 = makeFinalHamiltonian(dim, H)
#     energy = np.linalg.eigvalsh(H1)
    
#     for j in range(number_of_eigenstates):
#         spectra_of_H1[j][i] = energy[j]
    
#     minimum_gap[i], w = spectra(dim, H0, H1, A, B, gs, s_divisions, number_of_eigenstates)
    
        
        
# # #Plot animation of H(s) vs RCS
# # plt.figure()
# # for i in range(num_RCS):
# #     H = graph.makeHamiltonianWithChains(G, target_graph, minor, K, alpha, beta, c, RCS[i])
# #     dim = len(H)
# #     gs = groundState(dim, H)
# #     H1 = makeFinalHamiltonian(dim, H)

# #     for j in range(s_divisions):
# #         energy = np.linalg.eigvalsh(A(sVector[j])*H0 + sVector[j]*H1)
# #         for k in range(number_of_eigenstates):
# #             spectra_of_H[k][j] = energy[k]
        
# #     plt.xlim([0.6, 1])
# #     plt.ylim([-10, 0])
# #     plt.title("RCS = " + str(RCS[i]))
# #     plt.xlabel("Scaled annealing time t/T")
# #     plt.ylabel("Energy (GHz)")
# #     for j in range(number_of_eigenstates):
# #         plt.plot(sVector, spectra_of_H[j])
# #     plt.savefig("RCS" + str(i) + ".png", bbox_inches='tight', dpi=300)
# #     plt.pause(0.0000001)
# #     plt.clf()

# # #Plot animation of H(RCS) vs s
# # plt.figure()
# # for i in range(s_divisions):
# #     for j in range(num_RCS):
# #         H = graph.makeHamiltonianWithChains(G, target_graph, minor, K, alpha, beta, c, RCS[j])
# #         dim = len(H)
# #         gs = groundState(dim, H)
# #         H1 = makeFinalHamiltonian(dim, H)
# #         energy = np.linalg.eigvalsh(A(sVector[i])*H0 + sVector[i]*H1)
# #         for k in range(number_of_eigenstates):
# #             spectra_of_Hr[k][j] = energy[k]
        
# #     plt.xlim([0, max_RCS])
# #     plt.title("s = " + str(sVector[i]))
# #     plt.xlabel("Relative chain strength")
# #     plt.ylabel("Energy (GHz)")
# #     for j in range(number_of_eigenstates):
# #         plt.plot(RCS, spectra_of_Hr[j])
# #     plt.savefig("s" + str(i) + ".png", bbox_inches='tight', dpi=300)
# #     plt.pause(0.0000001)
# #     plt.clf()



# #Plot spectra of H1 vs RCS
# plot1 = plt.figure()
# plt.xlim([0, max_RCS])
# plt.xlabel("Relative chain strength")
# plt.ylabel("Energy (GHz)")
# for i in range(number_of_eigenstates):
#     plt.plot(RCS, spectra_of_H1[i])
# filename = "Spectra of H1 vs RCS.png"
# plt.savefig(filename, bbox_inches='tight', dpi=300)


# #Plot minimum gap vs RCS
# plot2 = plt.figure()
# plt.xlim([0, max_RCS])
# plt.xlabel("Relative chain strength")
# plt.ylabel("Minumum gap (GHz)")
# plt.plot(RCS, minimum_gap)
# filename = "Minimum gap vs RCS.png"
# plt.savefig(filename, bbox_inches='tight', dpi=300)




#---------------Simulation of the physical system for a fixed graph-----------------#
#Create graph
G = createGraph("graph5.txt")
networkG = graph.toNetwork(G)



#Parameters
c = [0]
K = nx.graph_clique_number(networkG)
alpha = 1.0
beta = 2*alpha
min_RCS = 0.0
max_RCS = 2.0
num_RCS = 100
RCS = np.linspace(min_RCS, max_RCS, num_RCS)
min_tf = 0.5
max_tf = 1.7
num_tf = 5
tf = np.linspace(min_tf, max_tf, num_tf)
delta_t = 0.01
number_of_eigenstates = 6
number_of_overlaps = 50
physical_successProbability = []
target_graph = fileToNetwork("chimera.txt")
# minor = {0: [3679, 873], 1: [858], 2: [3514, 888], 3: [3619], 4: [3604], 5: [843]}
# minor = {0: [461, 457], 3: [458], 4: [462], 5: [453], 1: [822], 2: [1352]} #graph6
minor = {0: [2028], 2: [2027, 2029], 3: [2024], 1: [2031], 4: [1033]} #graph5
number_of_physical_qubits = 0
for i in range(len(minor)):
    number_of_physical_qubits += len(minor[i])
A, B = fun.schedule("DW_2000Q_6.txt")
logical_H = graph.makeIsingHamiltonian(G, alpha, beta, c)
logical_gs = groundState(pow(2, G.nv), logical_H)
physical_gs = [0 for i in range(len(logical_gs))]
for i in range(len(logical_gs)):
    logical_binary_gs = decimalToBinary(G.nv, logical_gs[i])
    physical_binary_gs_dict = defaultdict(int)
    for logical_qubit in minor:
        for physical_qubit in minor[logical_qubit]:
            physical_binary_gs_dict[(physical_qubit)] = logical_binary_gs[logical_qubit]
    physical_binary_gs_dict = dict(sorted(physical_binary_gs_dict.items()))
    physical_binary_gs = []
    for physical_qubit in physical_binary_gs_dict:
        physical_binary_gs.append(physical_binary_gs_dict[(physical_qubit)])
    physical_gs[i] = binaryToDecimal(physical_binary_gs)
        

for i_time in range(num_tf):
    t_f = tf[i_time]
    successProbability_for_a_given_tf = []
    print(i_time)
    for i in range(num_RCS):
        H = graph.makeHamiltonianWithChains(G, target_graph, minor, alpha, beta, c, RCS[i])
        dim = len(H)
        gs = groundState(dim, H)
        print('  ' + str(RCS[i]) + ': ' + str(gs))
        H1 = makeFinalHamiltonian(dim, H)
        H0 = makeInitialHamiltonian(number_of_physical_qubits)
        successProbabilityRK = [np.empty(0)]
        successProbabilityRK3 = [np.empty(0)]
        successProbabilityRK6 = [np.empty(0)]
        successProbabilityCN = [np.empty(0)]
        successProbabilityODE = [np.empty(0)]
        adiabatic = [np.empty(0)]
        energy = [np.empty([number_of_eigenstates, number_of_overlaps])]
        overlap = [np.empty([number_of_eigenstates, number_of_overlaps])]
        overlap1 = [np.empty(0)]
        
        
        #Make evolution
        adiabaticEvolutionAB(dim, H0, H1, A, B, adiabatic, physical_gs, t_f)
        
        # psi = np.ones(dim, dtype = complex)
        # initialT = time.time()
        # psi = evolutionABRK2(dim, H0, H1, psi, A, B, successProbabilityRK, gs, t_f, delta_t)
        # finalT = time.time()
        # print("RK4: " + str(finalT - initialT))
        
        psi = np.ones(dim, dtype = complex)
        initialT = time.time()
        psi = evolutionABCN2(dim, H0, H1, psi, A, B, successProbabilityCN, physical_gs, t_f, delta_t)
        successProbability_for_a_given_tf.append(successProbabilityCN[0][-1])
        finalT = time.time()
        # print("CN: " + str(finalT - initialT))
        
        # psi = np.ones(dim, dtype = complex)
        # initialT = time.time()
        # psi = evolutionABRK22(dim, H0, H1, psi, A, B, successProbabilityRK3, gs, t_f, delta_t)
        # finalT = time.time()
        # print("RK3: " + str(finalT - initialT))
        
        # psi = np.ones(dim, dtype = complex)
        # initialT = time.time()
        # psi = evolutionABRK62(dim, H0, H1, psi, A, B, successProbabilityRK6, gs, t_f, delta_t)
        # finalT = time.time()
        # print("RK6: " + str(finalT - initialT))
        
        psi = np.ones(dim, dtype = complex)
        t = np.linspace(0, t_f, int(t_f/delta_t))
        iteration = 0
        initialT = time.time()
        r = scipy.integrate.ode(fun.f).set_integrator('zvode', method='Adams', with_jacobian=False)
        r.set_initial_value(psi, 0).set_f_params(H0, H1, A, B, t_f)
        while r.successful() and r.t < t_f - 2*delta_t:
            psi_t = r.integrate(r.t + delta_t)
            
            if (iteration%30 == 0):
                successProbabilityODE[0] = np.append(successProbabilityODE[0], 0.)
                for i in physical_gs:
                    successProbabilityODE[0][-1] += np.real(psi_t[i])*np.real(psi_t[i]) + np.imag(psi_t[i])*np.imag(psi_t[i])
               
                successProbabilityODE[0][-1] = successProbabilityODE[0][-1]/float(dim)
        finalT = time.time()
        # print("ODE: " + str(finalT - initialT))
        
        # psi = np.ones(dim, dtype = complex)
        # psi = evolutionABCN6(dim, H0, H1, psi, A, B, energy, overlap, t_f, delta_t, number_of_overlaps, number_of_eigenstates)
        
        # psi = np.ones(dim, dtype = complex)
        # psi = evolutionABCN5(dim, H0, H1, psi, A, B, overlap1, gs, t_f, delta_t)
        
        
        
        
        
        
        # # Plot final Hamiltonian spectra and final state
        # probability = np.empty(dim)
        # for k in range(dim):
        #     probability[k] = np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k])
        # probability = probability / float(dim)
        # probability =  np.amax(H)*probability
        
        # plot1 = plt.figure(1)
        # xAxis1 = np.linspace(0, dim - 1,  dim)
        # plt.xlabel("State")
        # plt.ylabel("Energy")
        # plt.plot(xAxis1, H)
        # plt.plot(xAxis1, probability)
        
        
        
        
        # # Plot spectra over time 
        # plot3 = plt.figure(3)
        # xAxis3 = np.linspace(0, 1, number_of_overlaps)
        # plt.xlim(0, 1)
        # plt.xlabel("Annealing scaled time s=t/T")
        # plt.ylabel("Energy")
        # # for i in range(number_of_eigenstates):
        # #     plt.plot(xAxis3, energy[0][i])
        # for i in range(number_of_eigenstates):
        #     plt.scatter(xAxis3, energy[0][i], s = 100*overlap[0][i])
        
        # divisions = 100
        # sVector = np.linspace(0, 1, divisions)
        # minimum_gap, energies = spectra(dim, H0, H1, A, B, gs, divisions, number_of_eigenstates)
        
        # plot4 = plt.figure(4)
        # plt.xlim(0, 1)
        # plt.xlabel("Annealing scaled time s=t/T")
        # plt.ylabel("Energy")
        # for i in range(number_of_eigenstates):
        #     plt.plot(sVector, energies[i], 1)
        
        
        
        # # Plot overlap with target states
        # plt.figure()
        # # xAxis2a = np.linspace(0, t_f, len(successProbabilityRK[0]))
        # xAxis2b = np.linspace(0, 1, len(successProbabilityCN[0]))
        # # xAxis2c = np.linspace(0, t_f, len(successProbabilityRK3[0]))
        # # xAxis2d = np.linspace(0, t_f, len(successProbabilityRK6[0]))
        # xAxis2e = np.linspace(0, 1, len(successProbabilityODE[0]))
        # xAxis2f = np.linspace(0, 1, len(adiabatic[0]))
        # plt.xlim([0,1])
        # plt.ylim([0,1])
        # plt.xlabel("Annealing scaled time s=t/T")
        # plt.ylabel("Overlap with target states")
        # # plt.plot(xAxis2a, successProbabilityRK[0], 0.1, color = 'blue')
        # plt.plot(xAxis2b, successProbabilityCN[0], 0.1, color = 'red')
        # # plt.plot(xAxis2c, successProbabilityRK3[0], 0.1, color = 'orange')
        # # plt.plot(xAxis2d, successProbabilityRK6[0], 0.1, color = 'cyan')
        # plt.plot(xAxis2e, successProbabilityODE[0], 0.1, color = 'black')
        # plt.plot(xAxis2f, adiabatic[0], 0.1, color = 'green')
        
        # plt.show()
    
    # # Plot overlap with the instantaneous ground state
    # plot6 = plt.figure(6)
    # xAxis6 = np.linspace(0, 1, len(overlap1[0]))
    # plot6 = plt.figure(6)
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.xlabel("Annealing scaled time s=t/T")
    # plt.ylabel("Overlap with instantaneous ground state")
    # plt.plot(xAxis6, overlap1[0], 0.1)
    
    
    
    
    
    # # Plot schedule functions A(s), B(s)
    # plot7 = plt.figure(7)
    # xAxis7 = np.linspace(0, t_f, 1000)
    # plotA = A(xAxis7/t_f)
    # plotB = B(xAxis7/t_f)
    # plt.xlim([0,t_f])
    # plt.plot(xAxis7, plotA)
    # plt.plot(xAxis7, plotB)
    
    physical_successProbability.append(successProbability_for_a_given_tf)


plot8 = plt.figure(8)
normalize = mcolors.Normalize(vmin=min_tf, vmax=max_tf)
colormap = cm.jet
plt.xlim([0, max_RCS])
plt.ylim([0, 1])
plt.xlabel("Relative chain strength")
plt.ylabel("Success probability")
for i_time in range(num_tf):
    plt.plot(RCS, physical_successProbability[i_time], color = colormap(normalize(tf[i_time])), alpha = 0.7)
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(tf)
plt.colorbar(scalarmappaple)
filename = "Success probability vs chain strength.png"
plt.savefig(filename, bbox_inches='tight', dpi=1000)





#-----------Simulation of the logical system for different annealing times------------#
#Create graph
G = createGraph("graph5.txt")
networkG = graph.toNetwork(G)


#Create Hamiltonian matrices and initial state
c = [0]
K = nx.graph_clique_number(networkG)
alpha = 1.0
beta = 2*alpha
H = graph.makeIsingHamiltonian(G, alpha, beta, c)
dim = len(H)
min_tf = np.power(0.5, 1.0)
max_tf = np.power(1.7, 1.0)
num_tf = 5
tf = np.linspace(min_tf, max_tf, num_tf)
tf = np.power(tf, 1.0)
delta_t = 0.001
number_of_eigenstates = 6
number_of_overlaps = 50
gs = groundState(dim, H)
H1 = makeFinalHamiltonian(dim, H)
H0 = makeInitialHamiltonian(len(G.adjacency))
overlap_with_instantaneous_gs = []
overlap_with_target = []
# A, B = fun.schedule("Advantage_system4.1.txt")
A, B = fun.schedule("DW_2000Q_6.txt")
# A = fun.linearA
# B = fun.linearB

#Compute minimum gap
minimum_gap, energies = spectra(dim, H0, H1, A, B, gs, 250, 3)

for i_time in range(num_tf):
    print(i_time)
    t_f = tf[i_time]
    successProbabilityRK = [np.empty(0)]
    successProbabilityRK3 = [np.empty(0)]
    successProbabilityRK6 = [np.empty(0)]
    successProbabilityCN = [np.empty(0)]
    successProbabilityODE = [np.empty(0)]
    adiabatic = [np.empty(0)]
    energy = [np.empty([number_of_eigenstates, number_of_overlaps])]
    overlap = [np.empty([number_of_eigenstates, number_of_overlaps])]
    overlap1 = [np.empty(0)]
    iterations = int(t_f/delta_t)
    
    #Make evolution   
    psi = np.ones(dim, dtype = complex)
    initialT = time.time()
    psi = evolutionABCN2(dim, H0, H1, psi, A, B, successProbabilityRK, gs, t_f, delta_t)
    finalT = time.time()
    overlap_with_target.append(successProbabilityRK[0])
    # print("RK4: " + str(finalT - initialT))
    
    # psi = np.ones(dim, dtype = complex)
    # initialT = time.time()
    # psi = evolutionABCN2(dim, H0, H1, psi, A, B, successProbabilityCN, gs, t_f, delta_t)
    # finalT = time.time()
    # print("CN: " + str(finalT - initialT))
    
    # psi = np.ones(dim, dtype = complex)
    # initialT = time.time()
    # psi = evolutionABRK22(dim, H0, H1, psi, A, B, successProbabilityRK3, gs, t_f, delta_t)
    # finalT = time.time()
    # print("RK3: " + str(finalT - initialT))
    
    # psi = np.ones(dim, dtype = complex)
    # initialT = time.time()
    # psi = evolutionABRK62(dim, H0, H1, psi, A, B, successProbabilityRK6, gs, t_f, delta_t)
    # finalT = time.time()
    # print("RK6: " + str(finalT - initialT))
    
    # psi = np.ones(dim, dtype = complex)
    # t = np.linspace(0, t_f, int(t_f/delta_t))
    # iteration = 0
    # initialT = time.time()
    # r = scipy.integrate.ode(fun.f).set_integrator('zvode', method='Adams', with_jacobian=False)
    # r.set_initial_value(psi, 0).set_f_params(H0, H1, A, B, t_f)
    # while r.successful() and r.t < t_f - 2*delta_t:
    #     psi_t = r.integrate(r.t + delta_t)
        
    #     if (iteration%30 == 0):
    #         successProbabilityODE[0] = np.append(successProbabilityODE[0], 0.)
    #         for i in gs:
    #             successProbabilityODE[0][-1] += np.real(psi_t[i])*np.real(psi_t[i]) + np.imag(psi_t[i])*np.imag(psi_t[i])
           
    #         successProbabilityODE[0][-1] = successProbabilityODE[0][-1]/float(dim)
    # finalT = time.time()
    # print("ODE: " + str(finalT - initialT))
    
    # psi = np.ones(dim, dtype = complex)
    # psi = evolutionABRK6(dim, H0, H1, psi, A, B, energy, overlap, t_f, delta_t, number_of_overlaps, number_of_eigenstates)
    
    psi = np.ones(dim, dtype = complex)
    psi = evolutionABCN5(dim, H0, H1, psi, A, B, overlap1, gs, t_f, delta_t)
    overlap_with_instantaneous_gs.append(overlap1[0])
    
    
    
    
    # # Plot spectra over time 
    # plt.figure()
    # xAxis3 = np.linspace(0, t_f, number_of_overlaps)
    # plt.xlim(0, t_f)
    # for i in range(number_of_eigenstates):
    #     plt.plot(xAxis3, energy[0][i])
    # for i in range(number_of_eigenstates):
    #     plt.scatter(xAxis3, energy[0][i], s = 100*overlap[0][i])
        
    
    
normalize = mcolors.Normalize(vmin=min_tf, vmax=max_tf)
colormap = cm.jet

    
# Plot overlap with target states
plot5 = plt.figure(5)
adiabaticEvolutionAB(dim, H0, H1, A, B, adiabatic, gs, 1.0)
# xAxis2b = np.linspace(0, 1, len(successProbabilityCN[0]))
# xAxis2c = np.linspace(0, 1, len(successProbabilityRK3[0]))
# xAxis2d = np.linspace(0, 1, len(successProbabilityRK6[0]))
# xAxis2e = np.linspace(0, 1, len(successProbabilityODE[0]))
xAxis2f = np.linspace(0, 1, len(adiabatic[0]))
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("Scaled time s=t/t_f")
plt.ylabel("Overlap with target")
for i in range(num_tf):
    xAxis2a = np.linspace(0, 1, len(overlap_with_target[i]))
    plt.plot(xAxis2a, overlap_with_target[i], color = colormap(normalize(tf[i])), alpha = 0.7)
    # plt.plot(xAxis2b, successProbabilityCN[0], 0.1, color = 'red')
    # plt.plot(xAxis2c, successProbabilityRK3[0], 0.1, color = 'orange')
    # plt.plot(xAxis2d, successProbabilityRK6[0], 0.1, color = 'cyan')
    # plt.plot(xAxis2e, successProbabilityODE[0], 0.1, color = 'green')
plt.plot(xAxis2f, adiabatic[0], color = (0, 0, 0))
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(tf)
plt.colorbar(scalarmappaple)
filename = "Overlap with target state (DW_2000Q_6).png"
plt.savefig(filename, bbox_inches='tight', dpi=1000)


# Plot overlap with the instantaneous ground state
plot6 = plt.figure(6)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("Scaled time s=t/t_f")
plt.ylabel("Overlap with instantaneous ground state")
for i in range(num_tf):
    xAxis6 = np.linspace(0, 1, len(overlap_with_instantaneous_gs[i]))
    plt.plot(xAxis6, overlap_with_instantaneous_gs[i], color = colormap(normalize(tf[i])))
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(tf)
plt.colorbar(scalarmappaple)
filename = "Overlap with instantaneous groundstate (DW_2000Q_6).png"
plt.savefig(filename, bbox_inches='tight', dpi=1000)



# Plot success probability vs annealing time
plot7 = plt.figure(7)
plt.xlim([min(tf), max(tf)])
plt.ylim([0, 1])
plt.xlabel("Annealing time (ns)")
plt.ylabel("Success probability")
successProbability = np.empty(num_tf)
for i in range(num_tf):
    successProbability[i] = overlap_with_target[i][-1]
plt.plot(tf, successProbability)
filename = "Success probability vs annealing time (DW_2000Q_6).png"
plt.savefig(filename, bbox_inches='tight', dpi=1000)


plt.figure()
logical_successProbability = np.empty((num_tf, 2))
for i in range(num_tf):
    logical_successProbability[i][0] = successProbability[i]
    logical_successProbability[i][1] = successProbability[i]
plt.xlim([0, max_RCS])
plt.ylim([0, 1])
xAxis2 = np.linspace(0, max_RCS, 2)
plt.xlabel("Relative chain strength")
plt.ylabel("Success probability")
for i_time in range(num_tf):
    plt.plot(RCS, physical_successProbability[i_time], color = colormap(normalize(tf[i_time])), alpha = 0.7)
for i_time in range(num_tf):
    plt.plot(xAxis2, logical_successProbability[i_time], color = colormap(normalize(tf[i_time])), alpha = 0.7, linestyle = 'dashed')
plt.colorbar(scalarmappaple)
filename = "Success probability vs chain strength (w and wout chains).png"
plt.savefig(filename, bbox_inches='tight', dpi=1000)


plt.show()























#-----------------------General problem solved by QA--------------------------#
# # Set quantum parameters
# number_of_logical_qubits = 3
# number_of_eigenstates = 8
# min_RCS = 0.0
# max_RCS = 2.0
# num_RCS = 100
# RCS = np.linspace(min_RCS, max_RCS, num_RCS)
# t_f = 1.0
# delta_t = 0.01
# s_divisions = 100
# h = defaultdict(int)
# J = defaultdict(int)
# physical_successProbability = []
# spectra_of_H1 = np.empty((number_of_eigenstates, num_RCS))
# minimum_gap = np.empty(num_RCS)
# A, B = fun.schedule("DW_2000Q_6.txt")

# for i in range(number_of_logical_qubits):
#     h[(i)] = 0.0
#     for j in range(i):
#         J[(i, j)] = 0.0

# # Set values for h and J
# # h[(0)] = -1.0
# # h[(1)] = -1.0
# # h[(2)] = -8.0
# # J[(1, 0)] = 4.0
# # J[(2, 0)] = 2.0
# # J[(2, 1)] = 1.0
# for i in range(number_of_logical_qubits):
#     h[(i)] = uniform(-2, 2)
#     for j in range(i):
#         J[(i, j)] = uniform(-1, 1)
    
# logical_H = evaluateHamiltonian(number_of_logical_qubits, h, J)
# logical_gs = groundState(pow(2, number_of_logical_qubits), logical_H)


# # Target graph and minor-embedding
# target_graph = nx.Graph()
# target_graph.add_edge(0, 1)
# target_graph.add_edge(1, 2)
# target_graph.add_edge(2, 3)
# target_graph.add_edge(3, 0)
# minor = {0: [0], 1: [1], 2: [2, 3]}
# number_of_physical_qubits = target_graph.number_of_nodes()
# H0 = makeInitialHamiltonian(number_of_physical_qubits)
# physical_gs = [0 for i in range(len(logical_gs))]
# for i in range(len(logical_gs)):
#     logical_binary_gs = decimalToBinary(number_of_logical_qubits, logical_gs[i])
#     physical_binary_gs = []
#     for logical_qubit in minor:
#         for physical_qubit in minor[logical_qubit]:
#             physical_binary_gs.append(logical_binary_gs[logical_qubit])
#     physical_gs[i] = binaryToDecimal(physical_binary_gs)

# # Scan evolution and minimum gap over RCS
# for i in range(num_RCS):
#     successProbabilityCN = [np.empty(0)]
#     H = generalHamiltonian(target_graph, minor, h, J, RCS[i])
#     dim = len(H)
#     gs = groundState(dim, H)
#     print(gs)
#     H1 = makeFinalHamiltonian(dim, H)
    
#     # Spectrum of H1
#     energy = np.linalg.eigvalsh(H1)
#     for j in range(number_of_eigenstates):
#         spectra_of_H1[j][i] = energy[j]
    
#     # Minimum gap
#     minimum_gap[i], w = spectra(dim, H0, H1, A, B, gs, s_divisions, number_of_eigenstates)
    
#     # Success Probability
#     psi = np.ones(dim, dtype = complex)
#     psi = evolutionABCN2(dim, H0, H1, psi, A, B, successProbabilityCN, physical_gs, t_f, delta_t)
#     physical_successProbability.append(successProbabilityCN[0][-1])
    


# # Plot spectra of H1 vs RCS
# plot1 = plt.figure(1)
# plt.xlim([0, max_RCS])
# plt.xlabel("Relative chain strength")
# plt.ylabel("Energy (GHz)")
# for i in range(number_of_eigenstates):
#     plt.plot(RCS, spectra_of_H1[i])
# # filename = "Spectra of H1 vs RCS.png"
# # plt.savefig(filename, bbox_inches='tight', dpi=300)


# # Plot minimum gap vs RCS
# plot2 = plt.figure()
# plt.xlim([0, max_RCS])
# plt.xlabel("Relative chain strength")
# plt.ylabel("Minumum gap (GHz)")
# plt.plot(RCS, minimum_gap)
# optimal_RCS_gap = RCS[np.argmax(minimum_gap)]
# # filename = "Minimum gap vs RCS.png"
# # plt.savefig(filename, bbox_inches='tight', dpi=300)


# # Plot success probability
# plot3 = plt.figure(3)
# plt.xlim([0, max_RCS])
# plt.ylim([0, 1])
# plt.xlabel("Relative chain strength")
# plt.ylabel("Success probability")
# plt.plot(RCS, physical_successProbability, alpha = 0.7)
# optimal_RCS_prob = RCS[np.argmax(physical_successProbability)]
# # filename = "Success probability vs chain strength (general).png"
# # plt.savefig(filename, bbox_inches='tight', dpi=300)

# print("Optimal RCS in terms of gap: " + str(optimal_RCS_gap))
# print("Optimal RCS in terms of probability: " + str(optimal_RCS_prob))







#---------General problem solved by QA (different random experiments)---------#
# # Set quantum parameters
# number_of_logical_qubits = 4
# number_of_eigenstates = 8
# number_of_random_experiments = 200
# min_RCS = 0.0
# max_RCS = 2.0
# num_RCS = 100
# RCS = np.linspace(min_RCS, max_RCS, num_RCS)
# optimal_RCS_prob = np.empty(number_of_random_experiments)
# optimal_RCS_gap = np.empty(number_of_random_experiments)
# t_f = 1.0
# delta_t = 0.01
# s_divisions = 100
# h = defaultdict(int)
# J = defaultdict(int)
# spectra_of_H1 = np.empty((number_of_eigenstates, num_RCS))
# minimum_gap = np.empty(num_RCS)
# A, B = fun.schedule("DW_2000Q_6.txt")

# for i in range(number_of_logical_qubits):
#     h[(i)] = 0.0
#     for j in range(i):
#         J[(i, j)] = 0.0


# for rnd_exp in range(number_of_random_experiments):
#     physical_successProbability = []
#     # Set values for h and J
#     # h[(0)] = -1.0
#     # h[(1)] = -1.0
#     # h[(2)] = -8.0
#     # J[(1, 0)] = 4.0
#     # J[(2, 0)] = 2.0
#     # J[(2, 1)] = 1.0
#     for i in range(number_of_logical_qubits):
#         h[(i)] = uniform(-2, 2)
#         for j in range(i):
#             J[(i, j)] = uniform(-1, 1)
            
#     logical_H = evaluateHamiltonian(number_of_logical_qubits, h, J)
#     logical_gs = groundState(pow(2, number_of_logical_qubits), logical_H)
    
#     # Target graph and minor-embedding
#     target_graph = nx.Graph()
#     target_graph.add_edge(0, 1)
#     target_graph.add_edge(0, 2)
#     target_graph.add_edge(0, 4)
#     target_graph.add_edge(1, 3)
#     target_graph.add_edge(1, 5)
#     target_graph.add_edge(2, 3)
#     target_graph.add_edge(2, 5)
#     target_graph.add_edge(3, 4)
#     target_graph.add_edge(4, 5)
#     minor = {0: [0], 1: [1], 2: [2, 3], 3: [4, 5]}
#     number_of_physical_qubits = target_graph.number_of_nodes()
#     H0 = makeInitialHamiltonian(number_of_physical_qubits)
#     physical_gs = [0 for i in range(len(logical_gs))]
#     for i in range(len(logical_gs)):
#         logical_binary_gs = decimalToBinary(number_of_logical_qubits, logical_gs[i])
#         physical_binary_gs = []
#         for logical_qubit in minor:
#             for physical_qubit in minor[logical_qubit]:
#                 physical_binary_gs.append(logical_binary_gs[logical_qubit])
#         physical_gs[i] = binaryToDecimal(physical_binary_gs)
    
#     # Scan evolution and minimum gap over RCS
#     for i in range(num_RCS):
#         successProbabilityCN = [np.empty(0)]
#         H = generalHamiltonian(target_graph, minor, h, J, RCS[i])
#         dim = len(H)
#         gs = groundState(dim, H)
#         H1 = makeFinalHamiltonian(dim, H)
        
#         # Spectrum of H1
#         energy = np.linalg.eigvalsh(H1)
#         for j in range(number_of_eigenstates):
#             spectra_of_H1[j][i] = energy[j]
        
#         # Minimum gap
#         minimum_gap[i], w = spectra(dim, H0, H1, A, B, gs, s_divisions, number_of_eigenstates)
        
#         #Success Probability
#         psi = np.ones(dim, dtype = complex)
#         psi = evolutionABCN2(dim, H0, H1, psi, A, B, successProbabilityCN, physical_gs, t_f, delta_t)
#         physical_successProbability.append(successProbabilityCN[0][-1])
        
    
    
#     # # Plot spectra of H1 vs RCS
#     # plt.figure()
#     # plt.xlim([0, max_RCS])
#     # plt.xlabel("Relative chain strength")
#     # plt.ylabel("Energy (GHz)")
#     # for i in range(number_of_eigenstates):
#     #     plt.plot(RCS, spectra_of_H1[i])
#     # # filename = "Spectra of H1 vs RCS.png"
#     # # plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    
#     # # Plot minimum gap vs RCS
#     # plt.figure()
#     # plt.xlim([0, max_RCS])
#     # plt.xlabel("Relative chain strength")
#     # plt.ylabel("Minumum gap (GHz)")
#     # plt.plot(RCS, minimum_gap)
#     # # filename = "Minimum gap vs RCS.png"
#     # # plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    
#     # # Plot success probability
#     # plt.figure()
#     # plt.xlim([0, max_RCS])
#     # plt.ylim([0, 1])
#     # plt.xlabel("Relative chain strength")
#     # plt.ylabel("Success probability")
#     # plt.plot(RCS, physical_successProbability, alpha = 0.7)
#     # # filename = "Success probability vs chain strength (general).png"
#     # # plt.savefig(filename, bbox_inches='tight', dpi=300)
    
#     optimal_RCS_gap[rnd_exp] = RCS[np.argmax(minimum_gap)]
#     optimal_RCS_prob[rnd_exp] = RCS[np.argmax(physical_successProbability)]
    
#     print("Random experiment #" + str(rnd_exp))
#     print(gs)
#     print("Optimal RCS in terms of gap: " + str(optimal_RCS_gap[rnd_exp]))
#     print("Optimal RCS in terms of probability: " + str(optimal_RCS_prob[rnd_exp]))
#     print("")


# plt.figure()
# plt.xlabel("RCS")
# plt.ylabel("# of occurrences")
# plt.title("Optimal RCS in terms of minimum gap")
# plt.hist(optimal_RCS_gap, bins = 20)
# filename = "Optimal RCS in terms of gap 2.png"
# plt.savefig(filename, bbox_inches='tight', dpi=300)

# plt.figure()
# plt.xlabel("RCS")
# plt.ylabel("# of occurrences")
# plt.title("Optimal RCS in terms of probability")
# plt.hist(optimal_RCS_prob, bins = 20)
# filename = "Optimal RCS in terms of probability 2.png"
# plt.savefig(filename, bbox_inches='tight', dpi=300)
# plt.show()


