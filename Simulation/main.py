import math
import time
import sys
import numpy as np
import scipy.integrate
from matplotlib import colors
import matplotlib.pyplot as plt
from graph_things import graph, createGraph, networkToGraph, fileToNetwork, networkToFile, decimalToBinary
from evolution import evolutionCN2, evolutionCN3, evolutionCN4, evolutionCN5, evolutionCN6
from evolution import evolutionABCN2, evolutionABCN3, evolutionABCN4, evolutionABCN5, evolutionABCN6
from evolution import adiabaticEvolution, adiabaticEvolutionAB, spectra
from evolution import evolutionRK2, evolutionRK3, evolutionRK4, evolutionRK5, evolutionRK6
from evolution import evolutionABRK2, evolutionABRK3, evolutionABRK4, evolutionABRK5, evolutionABRK6
from evolution import makeInitialHamiltonian, makeInitialHamiltonian2, makeFinalHamiltonian, groundState
import functions as fun
import networkx as nx
from evolution import evolutionABRK62, evolutionABRK22






#--------------------Histogram plot of the spectrum of H_p--------------------#
# nv = 19
# p = 0.5
# K = 6
# B = 1
# A = (K + 1)*B
# dim = int(pow(2, nv))
# networkG = nx.fast_gnp_random_graph(nv, p)
# G = networkToGraph(networkG)
# H = graph.makeHamiltonian(G, K, A, B)
# energy = np.unique(H)
# r = np.empty(len(energy) - 2)
# for i in range(len(r)):
#     r[i] = np.amin(np.array([(energy[i+2]-energy[i+1])/(energy[i+1]-energy[i]), (energy[i+1]-energy[i])/(energy[i+2]-energy[i+1])]))
    
# averge_r = np.mean(r)



# print("Clique number: " + str(nx.graph_clique_number(networkG)))


# # Plot the graph
# plot1 = plt.figure(1)
# nx.draw(networkG, with_labels = True)
# plt.show() 

# # Plot histogram of energies
# plot2 = plt.figure(2)
# plt.hist(H, bins = 20)
# plt.show()


# # Plot histogram of energy distances
# plot3 = plt.figure(3)
# plt.hist(r, bins = 10)
# plt.show()












#-------------------NP-complete problem with random graphs--------------------#
# # PROBABILITY OF SUCCESS AT A GIVEN TIME
# nv_min = 4  #Minimum number of vertices
# nv_max = 8  #Maximum number of vertices
# p = 0.5  #Probability that there is an edge joining two vertices
# t_f = 15  #Time of the evolution
# delta_t = 0.1  #Step in the Crank-Nicolson/Runge-Kutta algorithm
# iterations = int(t_f/delta_t)  #Number of times the Crank-Nicolson algorithm is called
# number_of_experiments = 20  #Number of simulations for each size
# probability_at_tf = np.empty(number_of_experiments)  #Probabilities of success at a given time
# meanProbability = np.empty(nv_max - nv_min + 1)  #Mean probabilities for each size
# deviationProbability = np.empty(nv_max - nv_min + 1)  #Standard deviation of the probabilities for each size

# for nv in range(nv_min, nv_max + 1):
#     K = int(nv/2) 
#     B = 1
#     A = (K + 1)*B
#     dim = int(pow(2, nv))
#     H0 = makeInitialHamiltonian(nv)
    
#     for i in range(number_of_experiments):
#         # Create graphs
#         networkG = nx.fast_gnp_random_graph(nv, p)
#         G = networkToGraph(networkG)
#         H = graph.makeHamiltonian(G, K, A, B)
        
#         # Create Hamiltonian matrices and initial state
#         gs = groundState(dim, H)
#         H1 = makeFinalHamiltonian(dim, H)
#         psi = np.ones(dim, dtype = complex)
#         successProbability = [np.empty(0)]
        
#         # Make evolution
#         psi = evolutionABCN3(dim, H0, H1, psi, fun.linearA, fun.linearB, successProbability, gs, t_f, delta_t)
        
#         # Probabilities
#         probability = np.empty(dim)
#         for k in range(dim):
#             probability[k] = np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k])
#         probability = probability / float(dim)
#         probability = np.amax(H) * probability
#         xAxis1 = np.linspace(0, dim - 1,  dim)
#         xAxis2 = np.linspace(0, t_f, len(successProbability[0]))
#         probability_at_tf[i] = successProbability[0][-1]
        
#         # Draw stuff
#         plot1 = plt.figure(1)
#         plt.title("Graph #" + str(i + 1) + " (" + str(nx.graph_clique_number(networkG)) + ")")
#         plt.plot(xAxis1, H)
#         plt.plot(xAxis1, probability)
        
#         plot2 = plt.figure(2)
#         plt.title("Graph #" + str(i + 1) + " (" + str(nx.graph_clique_number(networkG)) + ")")
#         plt.xlim([0, t_f])
#         plt.ylim([0, 1])
#         plt.scatter(xAxis2, successProbability[0], 0.1)
        
#         plot3 =plt.figure(3)
#         nx.draw(networkG, with_labels = True)
    
#         plt.show()
    
#     meanProbability[nv - nv_min] = np.mean(probability_at_tf)
#     deviationProbability[nv - nv_min] = np.std(probability_at_tf)
#     print("Graphs with " + str(nv) + " vertices:")    
#     print("Mean probability: " + str(meanProbability[nv - nv_min]))
#     print("Deviation: " + str(deviationProbability[nv - nv_min]))
#     print("")

# xAxis = np.linspace(nv_min, nv_max, nv_max - nv_min + 1)
# plt.figure()
# plt.xlim([nv_min - 1, nv_max + 1])
# plt.ylim([0,1])
# plt.scatter(xAxis, meanProbability, 100)
# plt.show()
    




# AVERAGE TIME TO ACHIVE A GIVEN PROBABILITY OF SUCCESS
# nv_min = 4  #Minimum number of vertices
# nv_max = 5  #Maximum number of vertices
# p = 0.5  #Probability that there is an edge joining two vertices
# pSuccess = 0.2  #Considered probability of success
# delta_t = 0.1  #Step in the Crank-Nicolson algorithm
# number_of_experiments = 20  #Number of simulations for each size
# time_for_pSuccess = np.empty(number_of_experiments)
# meanTime = np.empty(nv_max - nv_min + 1)  #Mean time for each size
# deviationTime = np.empty(nv_max - nv_min + 1)  #Standard deviation of the time for each size

# for nv in range(nv_min, nv_max + 1):
#     K = int(nv/2) 
#     B = 1
#     A = (K + 1)*B
#     dim = int(pow(2, nv))
#     H0 = makeInitialHamiltonian(nv)
    
#     for i in range(number_of_experiments):
#         # Create graphs
#         networkG = nx.fast_gnp_random_graph(nv, p)
#         G = networkToGraph(networkG)
#         H = graph.makeHamiltonian(G, K, A, B)
        
#         # Create Hamiltonian matrices and initial state
#         gs = groundState(dim, H)
#         H1 = makeFinalHamiltonian(dim, H)
#         psi = np.ones(dim, dtype = complex)
#         successProbability = [np.empty(0)]
        
#         # Make evolution
#         t_f = [0.]
#         psi = evolution4(dim, H0, H1, psi, fun.exponentialsqrt, successProbability, pSuccess, gs, t_f, delta_t)
        
#         # Probabilities
#         probability = np.empty(dim)
#         for k in range(dim):
#             probability[k] = np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k])
#         probability = probability / float(dim)
#         probability = np.amax(H) * probability
#         xAxis1 = np.linspace(0, dim - 1,  dim)
#         xAxis2 = np.linspace(0, t_f[0], len(successProbability[0]))
#         time_for_pSuccess[i] = t_f[0]
        
#         # Draw stuff
#         plot1 = plt.figure(1)
#         plt.title("Graph #" + str(i + 1) + " (" + str(nx.graph_clique_number(networkG)) + ")")
#         plt.plot(xAxis1, H)
#         plt.plot(xAxis1, probability)
        
#         plot2 = plt.figure(2)
#         plt.title("Graph #" + str(i + 1) + " (" + str(nx.graph_clique_number(networkG)) + ")")
#         plt.ylim([0, pSuccess + 0.05])
#         plt.scatter(xAxis2, successProbability[0], 0.1)
        
#         plot3 =plt.figure(3)
#         nx.draw(networkG, with_labels = True)
    
#         plt.show()
    
#     meanTime[nv - nv_min] = np.mean(time_for_pSuccess)
#     deviationTime[nv - nv_min] = np.std(time_for_pSuccess)
#     print("Graphs with " + str(nv) + " vertices:")    
#     print("Mean time: " + str(meanTime[nv - nv_min]))
#     print("Deviation: " + str(deviationTime[nv - nv_min]))
#     print("")

# xAxis = np.linspace(nv_min, nv_max, nv_max - nv_min + 1)
# plt.figure()
# plt.xlim([nv_min - 1, nv_max + 1])
# plt.scatter(xAxis, meanTime, 50)
# plt.show()





#---------------------NP-complete problem (plot overlap)----------------------#
# #Create graph
# networkG = nx.fast_gnp_random_graph(5, 0.6)
# G = networkToGraph(networkG)
# H = graph.makeHamiltonian(G, 3, 10, 1)



# #Create Hamiltonian matrices and initial state
# dim = len(H)
# t_f = 50
# delta_t = 0.01
# iterations = int(t_f/delta_t)
# gs = groundState(dim, H)
# H1 = makeFinalHamiltonian(dim, H)
# H0 = makeInitialHamiltonian(len(G.adjacency))
# psi = np.ones(dim, dtype = complex)
# overlap = [np.empty(0)]

# # #Make evolution
# psi = evolutionABRK5(dim, H0, H1, psi, fun.linearA, fun.linearB, overlap, gs, t_f, delta_t)

# # Probabilities
# probability = np.empty(dim)
# for k in range(dim):
#     probability[k] = np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k])
# probability = probability / float(dim)
# probability =  np.amax(H) * probability
# xAxis1 = np.linspace(0, dim - 1,  dim)
# xAxis2 = np.linspace(0, t_f, len(overlap[0]))



# # Draw stuff
# plot1 = plt.figure(1)
# plt.plot(xAxis1, H)
# plt.plot(xAxis1, probability)

# plot2 = plt.figure(2)
# plt.xlim([0,t_f])
# plt.ylim([0,1])
# plt.scatter(xAxis2, overlap, 0.1)

# plot3 = plt.figure(3)
# nx.draw(networkG, with_labels = True)
# plt.show()





#---------------------------NP-complete problem-------------------------------#
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




#---------------------------NP-complete problem-------------------------------#
# #Create graph
# G = createGraph("graph2.txt")
# networkG = graph.toNetwork(G)


# #Create Hamiltonian matrices and initial state
# c = [0]
# K = nx.graph_clique_number(networkG)
# beta = 1.
# alpha = (K + 1)*beta
# H = graph.makeIsingHamiltonian(G, K, alpha, beta, c)
# dim = len(H)
# t_f = 12.0
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

# psi = np.ones(dim, dtype = complex)
# initialT = time.time()
# psi = evolutionABRK2(dim, H0, H1, psi, A, B, successProbabilityRK, gs, t_f, delta_t)
# finalT = time.time()
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

# psi = np.ones(dim, dtype = complex)
# psi = evolutionABRK5(dim, H0, H1, psi, A, B, overlap1, gs, t_f, delta_t)






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
# xAxis3 = np.linspace(0, t_f, number_of_overlaps)
# plt.xlim(0, t_f)
# # for i in range(number_of_eigenstates):
# #     plt.plot(xAxis3, energy[0][i])
# for i in range(number_of_eigenstates):
#     plt.scatter(xAxis3, energy[0][i], s = 100*overlap[0][i])

# divisions = 100
# sVector = np.linspace(0, 1, divisions)
# minimum_gap, energies = spectra(dim, H0, H1, A, B, gs, divisions, number_of_eigenstates)

# plot4 = plt.figure(4)
# plt.xlim(0, 1)
# plt.ylim(-4.1, 0.1)
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
# plt.plot(xAxis2a, successProbabilityRK[0], 0.1, color = 'blue')
# plt.plot(xAxis2b, successProbabilityCN[0], 0.1, color = 'red')
# # plt.plot(xAxis2c, successProbabilityRK3[0], 0.1, color = 'orange')
# # plt.plot(xAxis2d, successProbabilityRK6[0], 0.1, color = 'cyan')
# plt.plot(xAxis2e, successProbabilityODE[0], 0.1, color = 'black')
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





# # Plot schedule functions A(s), B(s)
# plot7 = plt.figure(7)
# xAxis7 = np.linspace(0, t_f, 1000)
# plotA = A(xAxis7/t_f)
# plotB = B(xAxis7/t_f)
# plt.xlim([0,t_f])
# plt.plot(xAxis7, plotA)
# plt.plot(xAxis7, plotB)

# plt.show()



#--------------------------NP-problem with chains-----------------------------#
#Create graph
G = createGraph("graph2.txt")
networkG = graph.toNetwork(G)


#Create Hamiltonian matrices and initial state
c = [0]
K = nx.graph_clique_number(networkG)
beta = 1.
alpha = (K + 1)*beta
min_RCS = 0.1
max_RCS = 5.0
num_RCS = 50
RCS = np.linspace(min_RCS, max_RCS, num_RCS)
min_tf = 1.0
max_tf = 40.0
num_tf = 5
tf = np.linspace(min_tf, max_tf, num_tf)
delta_t = 0.005
number_of_eigenstates = 6
number_of_overlaps = 50
successProbability = []
target_graph = fileToNetwork("chimera.txt")
minor = {0: [4, 2], 1: [7, 3], 2: [6], 3: [0]}
number_of_physical_qubits = 0
for i in range(len(minor)):
    number_of_physical_qubits += len(minor[i])

for i_time in range(num_tf):
    t_f = tf[i_time]
    successProbability_for_a_given_tf = []
    for i in range(num_RCS):
        H = graph.makeHamiltonianWithChains(G, target_graph, minor, K, alpha, beta, c, RCS[i])
        dim = len(H)
        gs = groundState(dim, H)
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
        A = fun.linearA
        B = fun.linearB
        
        
            
        #Make evolution
        adiabaticEvolutionAB(dim, H0, H1, A, B, adiabatic, gs, t_f)
        
        # psi = np.ones(dim, dtype = complex)
        # initialT = time.time()
        # psi = evolutionABRK2(dim, H0, H1, psi, A, B, successProbabilityRK, gs, t_f, delta_t)
        # finalT = time.time()
        # print("RK4: " + str(finalT - initialT))
        
        psi = np.ones(dim, dtype = complex)
        initialT = time.time()
        psi = evolutionABCN2(dim, H0, H1, psi, A, B, successProbabilityCN, gs, t_f, delta_t)
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
                for i in gs:
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
        
        
        
        # Plot overlap with target states
        plt.figure()
        # xAxis2a = np.linspace(0, t_f, len(successProbabilityRK[0]))
        xAxis2b = np.linspace(0, 1, len(successProbabilityCN[0]))
        # xAxis2c = np.linspace(0, t_f, len(successProbabilityRK3[0]))
        # xAxis2d = np.linspace(0, t_f, len(successProbabilityRK6[0]))
        xAxis2e = np.linspace(0, 1, len(successProbabilityODE[0]))
        xAxis2f = np.linspace(0, 1, len(adiabatic[0]))
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel("Annealing scaled time s=t/T")
        plt.ylabel("Overlap with target states")
        # plt.plot(xAxis2a, successProbabilityRK[0], 0.1, color = 'blue')
        plt.plot(xAxis2b, successProbabilityCN[0], 0.1, color = 'red')
        # plt.plot(xAxis2c, successProbabilityRK3[0], 0.1, color = 'orange')
        # plt.plot(xAxis2d, successProbabilityRK6[0], 0.1, color = 'cyan')
        plt.plot(xAxis2e, successProbabilityODE[0], 0.1, color = 'black')
        plt.plot(xAxis2f, adiabatic[0], 0.1, color = 'green')
        
        plt.show()
    
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
    
    successProbability.append(successProbability_for_a_given_tf)

plot8 = plt.figure(8)
plt.xlim([min_RCS, max_RCS])
plt.ylim([0, 1])
plt.xlabel("Relative chain strength")
plt.ylabel("Success probability")
for i_time in range(num_tf):
    plt.plot(RCS, successProbability[i_time])
filename = "Success probability vs chain strength.png"
plt.savefig(filename, bbox_inches='tight', dpi=300)



#---------------------------NP-complete problem (different annealing times)-------------------------------#
# #Create graph
# G = createGraph("graph2.txt")
# networkG = graph.toNetwork(G)


# #Create Hamiltonian matrices and initial state
# c = [0]
# K = nx.graph_clique_number(networkG)
# beta = 1.
# alpha = (K + 1)*beta
# H = graph.makeIsingHamiltonian(G, K, alpha, beta, c)
# dim = len(H)
# min_tf = np.power(1, 1./3.)
# max_tf = np.power(100.0, 1./3.)
# num_tf = 200
# tf = np.linspace(min_tf, max_tf, num_tf)
# tf = np.power(tf, 3)
# delta_t = 0.01
# number_of_eigenstates = 6
# number_of_overlaps = 50
# gs = groundState(dim, H)
# H1 = makeFinalHamiltonian(dim, H)
# H0 = makeInitialHamiltonian(len(G.adjacency))
# overlap_with_instantaneous_gs = []
# overlap_with_target = []
# A = fun.linearA
# B = fun.linearB


# for i_time in range(num_tf):
#     t_f = tf[i_time]
#     successProbabilityRK = [np.empty(0)]
#     successProbabilityRK3 = [np.empty(0)]
#     successProbabilityRK6 = [np.empty(0)]
#     successProbabilityCN = [np.empty(0)]
#     successProbabilityODE = [np.empty(0)]
#     adiabatic = [np.empty(0)]
#     energy = [np.empty([number_of_eigenstates, number_of_overlaps])]
#     overlap = [np.empty([number_of_eigenstates, number_of_overlaps])]
#     overlap1 = [np.empty(0)]
#     iterations = int(t_f/delta_t)
    
#     #Make evolution   
#     psi = np.ones(dim, dtype = complex)
#     initialT = time.time()
#     psi = evolutionABCN2(dim, H0, H1, psi, A, B, successProbabilityRK, gs, t_f, delta_t)
#     finalT = time.time()
#     overlap_with_target.append(successProbabilityRK[0])
#     # print("RK4: " + str(finalT - initialT))
    
#     # psi = np.ones(dim, dtype = complex)
#     # initialT = time.time()
#     # psi = evolutionABCN2(dim, H0, H1, psi, A, B, successProbabilityCN, gs, t_f, delta_t)
#     # finalT = time.time()
#     # print("CN: " + str(finalT - initialT))
    
#     # psi = np.ones(dim, dtype = complex)
#     # initialT = time.time()
#     # psi = evolutionABRK22(dim, H0, H1, psi, A, B, successProbabilityRK3, gs, t_f, delta_t)
#     # finalT = time.time()
#     # print("RK3: " + str(finalT - initialT))
    
#     # psi = np.ones(dim, dtype = complex)
#     # initialT = time.time()
#     # psi = evolutionABRK62(dim, H0, H1, psi, A, B, successProbabilityRK6, gs, t_f, delta_t)
#     # finalT = time.time()
#     # print("RK6: " + str(finalT - initialT))
    
#     # psi = np.ones(dim, dtype = complex)
#     # t = np.linspace(0, t_f, int(t_f/delta_t))
#     # iteration = 0
#     # initialT = time.time()
#     # r = scipy.integrate.ode(fun.f).set_integrator('zvode', method='Adams', with_jacobian=False)
#     # r.set_initial_value(psi, 0).set_f_params(H0, H1, A, B, t_f)
#     # while r.successful() and r.t < t_f - 2*delta_t:
#     #     psi_t = r.integrate(r.t + delta_t)
        
#     #     if (iteration%30 == 0):
#     #         successProbabilityODE[0] = np.append(successProbabilityODE[0], 0.)
#     #         for i in gs:
#     #             successProbabilityODE[0][-1] += np.real(psi_t[i])*np.real(psi_t[i]) + np.imag(psi_t[i])*np.imag(psi_t[i])
           
#     #         successProbabilityODE[0][-1] = successProbabilityODE[0][-1]/float(dim)
#     # finalT = time.time()
#     # print("ODE: " + str(finalT - initialT))
    
#     # psi = np.ones(dim, dtype = complex)
#     # psi = evolutionABRK6(dim, H0, H1, psi, A, B, energy, overlap, t_f, delta_t, number_of_overlaps, number_of_eigenstates)
    
#     psi = np.ones(dim, dtype = complex)
#     psi = evolutionABCN5(dim, H0, H1, psi, A, B, overlap1, gs, t_f, delta_t)
#     overlap_with_instantaneous_gs.append(overlap1[0])
    
    
    
    
#     # # Plot spectra over time 
#     # plt.figure()
#     # xAxis3 = np.linspace(0, t_f, number_of_overlaps)
#     # plt.xlim(0, t_f)
#     # for i in range(number_of_eigenstates):
#     #     plt.plot(xAxis3, energy[0][i])
#     # for i in range(number_of_eigenstates):
#     #     plt.scatter(xAxis3, energy[0][i], s = 100*overlap[0][i])
        
    
    
# # Plot overlap with target states
# plot5 = plt.figure(5)
# adiabaticEvolutionAB(dim, H0, H1, A, B, adiabatic, gs, 1.0)
# # xAxis2b = np.linspace(0, 1, len(successProbabilityCN[0]))
# # xAxis2c = np.linspace(0, 1, len(successProbabilityRK3[0]))
# # xAxis2d = np.linspace(0, 1, len(successProbabilityRK6[0]))
# # xAxis2e = np.linspace(0, 1, len(successProbabilityODE[0]))
# xAxis2f = np.linspace(0, 1, len(adiabatic[0]))
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.xlabel("Annealing scaled time s=t/T")
# plt.ylabel("Overlap with target")
# for i in range(num_tf):
#     c = float(i/num_tf)
#     xAxis2a = np.linspace(0, 1, len(overlap_with_target[i]))
#     plt.plot(xAxis2a, overlap_with_target[i], color = (c, 0, 1-c), alpha = 0.7)
#     # plt.plot(xAxis2b, successProbabilityCN[0], 0.1, color = 'red')
#     # plt.plot(xAxis2c, successProbabilityRK3[0], 0.1, color = 'orange')
#     # plt.plot(xAxis2d, successProbabilityRK6[0], 0.1, color = 'cyan')
#     # plt.plot(xAxis2e, successProbabilityODE[0], 0.1, color = 'black')
# plt.plot(xAxis2f, adiabatic[0], color = (0, 0, 0))
# filename = "Overlap with target state.png"
# plt.savefig(filename, bbox_inches='tight', dpi=300)


# # Plot overlap with the instantaneous ground state
# plot6 = plt.figure(6)
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.xlabel("Annealing scaled time s=t/T")
# plt.ylabel("Overlap with instantaneous ground state")
# for i in range(num_tf):
#     c = float(i/num_tf)
#     xAxis6 = np.linspace(0, 1, len(overlap_with_instantaneous_gs[i]))
#     plt.plot(xAxis6, overlap_with_instantaneous_gs[i], color = (c, 0, 1-c))
# filename = "Overlap with instantaneous groundstate.png"
# plt.savefig(filename, bbox_inches='tight', dpi=300)

# # Plot success probability vs annealing time
# plot7 = plt.figure(7)
# plt.xlim([min(tf), max(tf)])
# plt.ylim([0, 1])
# plt.xlabel("Annealing time")
# plt.ylabel("Success probability")
# successProbability = np.empty(num_tf)
# for i in range(num_tf):
#     successProbability[i] = overlap_with_target[i][-1]
# plt.plot(tf, successProbability)
# filename = "Success probability vs annealing time.png"
# plt.savefig(filename, bbox_inches='tight', dpi=300)




# plt.show()





















# --------------------NP-complete problem (alternative)------------------------#
# #Create graph
# G = createGraph("graph2.txt")
# H = graph.makeHamiltonian3(G, 3, 1)
# networkG = graph.toNetwork (G)


# #Create Hamiltonian matrices and initial state
# dim = len(H)
# t_f = 100
# delta_t = 0.1
# iterations = int(t_f/delta_t)
# gs = groundState(dim, H)
# H1 = makeFinalHamiltonian(dim, H)
# H0 = makeInitialHamiltonian2(dim)
# psi = np.ones(dim, dtype = complex)
# successProbability = [np.zeros(iterations + 1)]

# #Make evolution
# psi = evolution2(dim, H0, H1, psi, fun.exponentialsqrt, successProbability, gs, t_f, delta_t)

# # Probabilities
# probability = np.empty(dim)
# for k in range(dim):
#     probability[k] = np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k])
# probability = probability / float(dim)



# probability =  50*probability
# xAxis1 = np.linspace(0, dim - 1,  dim)
# xAxis2 = np.linspace(0, t_f, iterations + 1)



# # Draw stuff
# plot1 = plt.figure(1)
# plt.plot(xAxis1, H)
# plt.plot(xAxis1, probability)

# plot2 = plt.figure(2)
# plt.xlim([0,t_f])
# plt.ylim([0,1])
# plt.scatter(xAxis2, successProbability[0], 0.1)

# plt.show()

# nx.draw(networkG, with_labels = True)



# -----------------------------NP-hard problem---------------------------------#
# #Create graph
# G = createGraph("graph1.txt")
# H = graph.makeHamiltonian2(G, 20, 3, 1)
# networkG = graph.toNetwork (G)

# #Create Hamiltonian matrices and initial state
# dim = len(H)
# t_f = 1
# delta_t = 0.5
# iterations = int(t_f/delta_t)
# gs = groundState(dim, H)
# H1 = makeFinalHamiltonian(dim, H)
# H0 = makeInitialHamiltonian(len(G.adjacency) + int(np.floor(np.log2(len(G.adjacency)))) + 1)
# psi = np.ones(dim, dtype = complex)
# successProbability = [np.zeros(iterations + 1)]

# #Make evolution
# psi = evolution2(dim, H0, H1, psi, fun.exponentialsqrt, successProbability, gs, t_f, delta_t)

# # Probabilities
# probability = np.empty(dim)
# for k in range(dim):
#     probability[k] = np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k])
# probability = probability / float(dim)



# probability =  50*probability
# xAxis1 = np.linspace(0, dim - 1,  dim)
# xAxis2 = np.linspace(0, t_f, iterations + 1)



# # Draw stuff
# plot1 = plt.figure(1)
# plt.plot(xAxis1, H)
# plt.plot(xAxis1, probability)

# plot2 = plt.figure(2)
# plt.xlim([0,t_f])
# plt.ylim([0,1])
# plt.scatter(xAxis2, successProbability[0], 0.1)

# plt.show()

# nx.draw(networkG, with_labels = True)





# -----------------------NP-hard problem (alternative)-------------------------#
# #Create graph
# G = createGraph("graph1.txt")
# H = graph.makeHamiltonian4(G, 3, 1)
# networkG = graph.toNetwork (G)

# #Create Hamiltonian matrices and initial state
# dim = len(H)
# t_f = 300
# delta_t = 0.5
# iterations = int(t_f/delta_t)
# gs = groundState(dim, H)
# H1 = makeFinalHamiltonian(dim, H)
# H0 = makeInitialHamiltonian2(dim)
# psi = np.ones(dim, dtype = complex)
# successProbability = [np.zeros(iterations + 1)]

# #Make evolution
# psi = evolution2(dim, H0, H1, psi, fun.exponentialsqrt, successProbability, gs, t_f, delta_t)


# # Probabilities
# probability = np.empty(dim)
# for k in range(dim):
#     probability[k] = np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k])
# probability = probability / float(dim)



# probability =  50*probability
# xAxis1 = np.linspace(0, dim - 1,  dim)
# xAxis2 = np.linspace(0, t_f, iterations + 1)



# # Draw stuff
# plot1 = plt.figure(1)
# plt.plot(xAxis1, H)
# plt.plot(xAxis1, probability)

# plot2 = plt.figure(2)
# plt.xlim([0,t_f])
# plt.ylim([0,1])
# plt.scatter(xAxis2, successProbability[0], 0.1)

# plt.show()

# nx.draw(networkG, with_labels = True)



















