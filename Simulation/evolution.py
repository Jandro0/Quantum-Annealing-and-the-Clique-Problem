import numpy as np
import matplotlib.pyplot as plt


# Do one Crank-Nicolson step
def timeplus1(dim, H, psi, delta_t):
    A = np.identity(dim, dtype=complex) - (0. + 1.j)*delta_t/2 * H
    psi = np.matmul(A, psi)

    A = np.linalg.inv(np.identity(dim, dtype=complex) +
                      (0. + 1.j)*delta_t/2 * H)
    psi = np.matmul(A, psi)

    return psi


# Do the Runge-Kutta step
def timeplus1RK(dim, psi, k1, k2, k3, k4, delta_t):
    return psi + float(1/6)*delta_t*(k1 + 2*k2 + 2*k3 + k4)


# Make the diagonal matrix of the final Hamiltonian
def makeFinalHamiltonian(dim, H):
    H1 = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        H1[k][k] = H[k]

    return H1


# Make the matrix of the initial Hamiltonian
def makeInitialHamiltonian(n):
    dim = pow(2, n)
    H0 = np.zeros((dim, dim), dtype=complex)
    sigmaX = np.array([[0, 1], [1, 0]], dtype=complex)
    for k in range(n):
        A = np.identity(pow(2, k), dtype=complex)
        B = np.identity(pow(2, n - k - 1), dtype=complex)
        H0 = H0 - matrixTensor(A, matrixTensor(sigmaX, B))

    return H0


# Make the matrix of the initial Hamiltonian (alternative)
def makeInitialHamiltonian2(dim):
    H0 = np.empty((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            H0[i][j] = -1

    for i in range(dim):
        H0[i][i] = 0

    return H0


# Return the tensor product of two matrices
def matrixTensor(A, B):
    dimA = np.shape(A)
    dimB = np.shape(B)
    C = np.empty((dimA[0]*dimB[0], dimA[1]*dimB[1]), dtype=complex)

    for k1 in range(dimA[0]):
        for k2 in range(dimA[1]):
            for k3 in range(dimB[0]):
                for k4 in range(dimB[1]):
                    C[k1*dimB[0] + k3][k2*dimB[1] + k4] = A[k1][k2]*B[k3][k4]

    return C


#-------------------Evolutions with Gamma (Crank-Nicolson)-----------------------#
# Simulate the evolution and plot probability coeffiecients over time
def evolutionCN2(dim, H0, H1, psi, Gamma, successProbability, gs, t_f, delta_t):
    t = 0
    iteration = 0
    probability = np.empty(dim)
    xAxis = np.linspace(0, dim - 1, dim)

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][iteration] += np.real(psi[i])*np.real(
            psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][iteration] = successProbability[0][iteration] / \
        float(dim)

    plot = plt.figure()

    while (t < t_f):
        H = H1 + Gamma(t) * H0
        psi = timeplus1(dim, H, psi, delta_t)
        t += delta_t
        iteration += 1

        successProbability[0] = np.append(successProbability[0], 0.)
        for i in gs:
            successProbability[0][iteration] += np.real(psi[i])*np.real(
                psi[i]) + np.imag(psi[i])*np.imag(psi[i])

        successProbability[0][iteration] = successProbability[0][iteration] / \
            float(dim)

        # if (iteration%50 == 0):
        #     for k in range(dim):
        #         probability[k] = (np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k]))/float(dim)

        #     plt.xlim(0, dim - 1)9
        #     plt.ylim(0, 1)
        #     plt.plot(xAxis, probability, color = 'black')
        #     plt.pause(0.0000001)
        #     plt.clf()

        #     print(str(t) + '   ' + str(successProbability[0][iteration]))

    return psi


# Simulate the evolution with fixed time
def evolutionCN3(dim, H0, H1, psi, Gamma, successProbability, gs, t_f, delta_t):
    t = 0
    iteration = 0

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][0] += np.real(psi[i]) * \
            np.real(psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][0] = successProbability[0][0]/float(dim)

    while (t < t_f):
        H = H1 + Gamma(t) * H0
        psi = timeplus1(dim, H, psi, delta_t)
        t += delta_t
        iteration += 1

        successProbability[0] = np.append(successProbability[0], 0.)
        for i in gs:
            successProbability[0][iteration] += np.real(psi[i])*np.real(
                psi[i]) + np.imag(psi[i])*np.imag(psi[i])

        successProbability[0][iteration] = successProbability[0][iteration] / \
            float(dim)

    return psi


# Simulate the evolution until a desired probability is achieved
def evolutionCN4(dim, H0, H1, psi, Gamma, successProbability, pSuccess, gs, t_f, delta_t):
    t = 0
    iteration = 0

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][0] += np.real(psi[i]) * \
            np.real(psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][0] = successProbability[0][0]/float(dim)

    while (successProbability[0][iteration] < pSuccess):
        H = H1 + Gamma(t) * H0
        psi = timeplus1(dim, H, psi, delta_t)
        t += delta_t
        iteration += 1

        successProbability[0] = np.append(successProbability[0], 0.)
        for i in gs:
            successProbability[0][iteration] += np.real(psi[i])*np.real(
                psi[i]) + np.imag(psi[i])*np.imag(psi[i])

        successProbability[0][iteration] = successProbability[0][iteration] / \
            float(dim)

    t_f[0] = t

    return psi


# Simulate the evolution and compute the overlap with the instantaneous ground state
def evolutionCN5(dim, H0, H1, psi, Gamma, overlap, gs, t_f, delta_t):
    t = 0
    degeneracy = len(gs)

    w, v = np.linalg.eigh(H1 + Gamma(t) * H0)
    prod = 0
    for i in range(degeneracy):
        vec = np.conjugate(v[:, i])
        dot = np.dot(psi, vec)
        prod += (np.real(dot)*np.real(dot) +
                 np.imag(dot)*np.imag(dot))/float(dim)

    overlap[0] = np.append(overlap[0], prod)

    while (t < t_f):
        H = H1 + Gamma(t) * H0
        psi = timeplus1(dim, H, psi, delta_t)
        t += delta_t

        w, v = np.linalg.eigh(H)
        prod = 0
        for i in range(degeneracy):
            vec = np.conjugate(v[:, i])
            dot = np.dot(psi, vec)
            prod += (np.real(dot)*np.real(dot) +
                     np.imag(dot)*np.imag(dot))/float(dim)
        overlap[0] = np.append(overlap[0], prod)

    return psi


# Simulate the evolution and compare with adiabatic evolution
def evolutionCN6(dim, H0, H1, psi, Gamma, energy, overlap, t_f, delta_t, number_of_overlaps, number_of_eigenstates):
    t = 0
    delta_t_overlap = int(t_f/(delta_t*number_of_overlaps))
    t_for_overlap = 0
    i = 0

    H = H1 + Gamma(0) * H0
    w, v = np.linalg.eigh(H)
    for k in range(number_of_eigenstates):
        vec = np.conjugate(v[:, k])
        dot = np.dot(psi, vec)
        overlap[0][k][0] = (np.real(dot)*np.real(dot) +
                            np.imag(dot)*np.imag(dot))/float(dim)
        energy[0][k][0] = w[k]

    while (t < t_f):
        H = H1 + Gamma(t) * H0
        psi = timeplus1(dim, H, psi, delta_t)
        t += delta_t
        t_for_overlap += 1

        if (t_for_overlap % delta_t_overlap == 0 and i < number_of_overlaps - 1):
            i += 1
            w, v = np.linalg.eigh(H)
            for k in range(number_of_eigenstates):
                vec = np.conjugate(v[:, k])
                dot = np.dot(psi, vec)
                overlap[0][k][i] = (
                    np.real(dot)*np.real(dot) + np.imag(dot)*np.imag(dot))/float(dim)
                energy[0][k][i] = w[k]

    return psi


#-------------------Evolutions with A(t), B(t) (Crank-Nicolson)-------------------#
# Simulate the evolution and plot probability coeffiecients over time
def evolutionABCN2(dim, H0, H1, psi, A, B, successProbability, gs, t_f, delta_t):
    t = 0.
    iteration = 0
    probability = np.empty(dim)
    xAxis = np.linspace(0, dim - 1, dim)

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][iteration] += np.real(psi[i])*np.real(
            psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][iteration] = successProbability[0][iteration] / \
        float(dim)

    plot = plt.figure()

    while (t < t_f):
        H = A(t/t_f)*H0 + B(t/t_f)*H1
        psi = timeplus1(dim, H, psi, delta_t)
        t += delta_t
        iteration += 1

        if (iteration % 1 == 0):
            successProbability[0] = np.append(successProbability[0], 0.)
            for i in gs:
                successProbability[0][-1] += np.real(psi[i])*np.real(
                    psi[i]) + np.imag(psi[i])*np.imag(psi[i])

            successProbability[0][-1] = successProbability[0][-1]/float(dim)

        # if (iteration%50 == 0):
        #     for k in range(dim):
        #         probability[k] = (np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k]))/float(dim)

        #     plt.xlim(0, dim - 1)9
        #     plt.ylim(0, 1)
        #     plt.plot(xAxis, probability, color = 'black')
        #     plt.pause(0.0000001)
        #     plt.clf()

        #     print(str(t) + '   ' + str(successProbability[0][iteration]))

    return psi


# Simulate the evolution with fixed time
def evolutionABCN3(dim, H0, H1, psi, A, B, successProbability, gs, t_f, delta_t):
    t = 0
    iteration = 0

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][0] += np.real(psi[i]) * \
            np.real(psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][0] = successProbability[0][0]/float(dim)

    while (t < t_f):
        H = A(t/t_f)*H0 + B(t/t_f)*H1
        psi = timeplus1(dim, H, psi, delta_t)
        t += delta_t
        iteration += 1

        successProbability[0] = np.append(successProbability[0], 0.)
        for i in gs:
            successProbability[0][iteration] += np.real(psi[i])*np.real(
                psi[i]) + np.imag(psi[i])*np.imag(psi[i])

        successProbability[0][iteration] = successProbability[0][iteration] / \
            float(dim)

    return psi


# Simulate the evolution until a desired probability is achieved
def evolutionABCN4(dim, H0, H1, psi, A, B, successProbability, pSuccess, gs, t_f, delta_t):
    t = 0
    iteration = 0

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][0] += np.real(psi[i]) * \
            np.real(psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][0] = successProbability[0][0]/float(dim)

    while (successProbability[0][iteration] < pSuccess):
        H = A(t/t_f)*H0 + B(t/t_f)*H1
        psi = timeplus1(dim, H, psi, delta_t)
        t += delta_t
        iteration += 1

        successProbability[0] = np.append(successProbability[0], 0.)
        for i in gs:
            successProbability[0][iteration] += np.real(psi[i])*np.real(
                psi[i]) + np.imag(psi[i])*np.imag(psi[i])

        successProbability[0][iteration] = successProbability[0][iteration] / \
            float(dim)

    t_f[0] = t

    return psi


# Simulate the evolution and compute the overlap with the instantaneous ground state
def evolutionABCN5(dim, H0, H1, psi, A, B, overlap, gs, t_f, delta_t):
    t = 0
    degeneracy = len(gs)

    w, v = np.linalg.eigh(A(t/t_f)*H0 + B(t/t_f)*H1)
    prod = 0
    for i in range(degeneracy):
        vec = np.conjugate(v[:, i])
        dot = np.dot(psi, vec)
        prod += (np.real(dot)*np.real(dot) +
                 np.imag(dot)*np.imag(dot))/float(dim)

    overlap[0] = np.append(overlap[0], prod)

    while (t < t_f):
        H = A(t/t_f)*H0 + B(t/t_f)*H1
        psi = timeplus1(dim, H, psi, delta_t)
        t += delta_t

        w, v = np.linalg.eigh(H)
        prod = 0
        for i in range(degeneracy):
            vec = np.conjugate(v[:, i])
            dot = np.dot(psi, vec)
            prod += (np.real(dot)*np.real(dot) +
                     np.imag(dot)*np.imag(dot))/float(dim)
        overlap[0] = np.append(overlap[0], prod)

    return psi


# Simulate the evolution and compare with adiabatic evolution
def evolutionABCN6(dim, H0, H1, psi, A, B, energy, overlap, t_f, delta_t, number_of_overlaps, number_of_eigenstates):
    t = 0
    delta_t_overlap = int(t_f/(delta_t*number_of_overlaps))
    t_for_overlap = 0
    i = 0

    H = A(t/t_f)*H0 + B(t/t_f)*H1
    w, v = np.linalg.eigh(H)
    for k in range(number_of_eigenstates):
        vec = np.conjugate(v[:, k])
        dot = np.dot(psi, vec)
        overlap[0][k][0] = (np.real(dot)*np.real(dot) +
                            np.imag(dot)*np.imag(dot))/float(dim)
        energy[0][k][0] = w[k]

    while (t < t_f):
        H = A(t/t_f)*H0 + B(t/t_f)*H1
        psi = timeplus1(dim, H, psi, delta_t)
        t += delta_t
        t_for_overlap += 1

        if (t_for_overlap % delta_t_overlap == 0 and i < number_of_overlaps - 1):
            i += 1
            w, v = np.linalg.eigh(H)
            for k in range(number_of_eigenstates):
                vec = np.conjugate(v[:, k])
                dot = np.dot(psi, vec)
                overlap[0][k][i] = (
                    np.real(dot)*np.real(dot) + np.imag(dot)*np.imag(dot))/float(dim)
                energy[0][k][i] = w[k]

    return psi


#-------------------------Evolutions with Gamma (Runge-Kutta)--------------------------#
# Simulate the evolution and plot probability coeffiecients over time
def evolutionRK2(dim, H0, H1, psi, Gamma, successProbability, gs, t_f, delta_t):
    t = 0
    iteration = 0
    probability = np.empty(dim)
    xAxis = np.linspace(0, dim - 1, dim)

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][iteration] += np.real(psi[i])*np.real(
            psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][iteration] = successProbability[0][iteration] / \
        float(dim)

    plot = plt.figure()

    while (t < t_f):
        k1 = (0. - 1.j)*np.matmul(H1 + Gamma(t)*H0, psi)
        k2 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t/2))
                                  * H0, psi + float(delta_t/2)*k1)
        k3 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t/2))
                                  * H0, psi + float(delta_t/2)*k2)
        k4 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t))
                                  * H0, psi + float(delta_t)*k3)
        psi = timeplus1RK(dim, psi, k1, k2, k3, k4, delta_t)
        t += delta_t
        iteration += 1

        successProbability[0] = np.append(successProbability[0], 0.)
        for i in gs:
            successProbability[0][iteration] += np.real(psi[i])*np.real(
                psi[i]) + np.imag(psi[i])*np.imag(psi[i])

        successProbability[0][iteration] = successProbability[0][iteration] / \
            float(dim)

        # if (iteration%50 == 0):
        #     for k in range(dim):
        #         probability[k] = (np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k]))/float(dim)

        #     plt.xlim(0, dim - 1)9
        #     plt.ylim(0, 1)
        #     plt.plot(xAxis, probability, color = 'black')
        #     plt.pause(0.0000001)
        #     plt.clf()

        #     print(str(t) + '   ' + str(successProbability[0][iteration]))

    return psi


# Simulate the evolution with fixed time
def evolutionRK3(dim, H0, H1, psi, Gamma, successProbability, gs, t_f, delta_t):
    t = 0
    iteration = 0

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][0] += np.real(psi[i]) * \
            np.real(psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][0] = successProbability[0][0]/float(dim)

    while (t < t_f):
        k1 = (0. - 1.j)*np.matmul(H1 + Gamma(t)*H0, psi)
        k2 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t/2))
                                  * H0, psi + float(delta_t/2)*k1)
        k3 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t/2))
                                  * H0, psi + float(delta_t/2)*k2)
        k4 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t))
                                  * H0, psi + float(delta_t)*k3)
        psi = timeplus1RK(dim, psi, k1, k2, k3, k4, delta_t)
        t += delta_t
        iteration += 1

        successProbability[0] = np.append(successProbability[0], 0.)
        for i in gs:
            successProbability[0][iteration] += np.real(psi[i])*np.real(
                psi[i]) + np.imag(psi[i])*np.imag(psi[i])

        successProbability[0][iteration] = successProbability[0][iteration] / \
            float(dim)

    return psi


# Simulate the evolution until a desired probability is achieved
def evolutionRK4(dim, H0, H1, psi, Gamma, successProbability, pSuccess, gs, t_f, delta_t):
    t = 0
    iteration = 0

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][0] += np.real(psi[i]) * \
            np.real(psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][0] = successProbability[0][0]/float(dim)

    while (successProbability[0][iteration] < pSuccess):
        k1 = (0. - 1.j)*np.matmul(H1 + Gamma(t)*H0, psi)
        k2 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t/2))
                                  * H0, psi + float(delta_t/2)*k1)
        k3 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t/2))
                                  * H0, psi + float(delta_t/2)*k2)
        k4 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t))
                                  * H0, psi + float(delta_t)*k3)
        psi = timeplus1RK(dim, psi, k1, k2, k3, k4, delta_t)
        t += delta_t
        iteration += 1

        successProbability[0] = np.append(successProbability[0], 0.)
        for i in gs:
            successProbability[0][iteration] += np.real(psi[i])*np.real(
                psi[i]) + np.imag(psi[i])*np.imag(psi[i])

        successProbability[0][iteration] = successProbability[0][iteration] / \
            float(dim)

    t_f[0] = t

    return psi


# Simulate the evolution and compute the overlap with the instantaneous ground state
def evolutionRK5(dim, H0, H1, psi, Gamma, overlap, gs, t_f, delta_t):
    t = 0
    degeneracy = len(gs)

    w, v = np.linalg.eigh(H1 + Gamma(t) * H0)
    prod = 0
    for i in range(degeneracy):
        vec = np.conjugate(v[:, i])
        dot = np.dot(psi, vec)
        prod += (np.real(dot)*np.real(dot) +
                 np.imag(dot)*np.imag(dot))/float(dim)

    overlap[0] = np.append(overlap[0], prod)

    while (t < t_f):
        H = H1 + Gamma(t) * H0
        k1 = (0. - 1.j)*np.matmul(H1 + Gamma(t)*H0, psi)
        k2 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t/2))
                                  * H0, psi + float(delta_t/2)*k1)
        k3 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t/2))
                                  * H0, psi + float(delta_t/2)*k2)
        k4 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t))
                                  * H0, psi + float(delta_t)*k3)
        psi = timeplus1RK(dim, psi, k1, k2, k3, k4, delta_t)
        t += delta_t

        w, v = np.linalg.eigh(H)
        prod = 0
        for i in range(degeneracy):
            vec = np.conjugate(v[:, i])
            dot = np.dot(psi, vec)
            prod += (np.real(dot)*np.real(dot) +
                     np.imag(dot)*np.imag(dot))/float(dim)
        overlap[0] = np.append(overlap[0], prod)

    return psi


# Simulate the evolution and compare with adiabatic evolution
def evolutionRK6(dim, H0, H1, psi, Gamma, energy, overlap, t_f, delta_t, number_of_overlaps, number_of_eigenstates):
    t = 0
    delta_t_overlap = int(t_f/(delta_t*number_of_overlaps))
    t_for_overlap = 0
    i = 0

    H = H1 + Gamma(0) * H0
    w, v = np.linalg.eigh(H)
    for k in range(number_of_eigenstates):
        vec = np.conjugate(v[:, k])
        dot = np.dot(psi, vec)
        overlap[0][k][0] = (np.real(dot)*np.real(dot) +
                            np.imag(dot)*np.imag(dot))/float(dim)
        energy[0][k][0] = w[k]

    while (t < t_f):
        H = H1 + Gamma(t) * H0
        k1 = (0. - 1.j)*np.matmul(H1 + Gamma(t)*H0, psi)
        k2 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t/2))
                                  * H0, psi + float(delta_t/2)*k1)
        k3 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t/2))
                                  * H0, psi + float(delta_t/2)*k2)
        k4 = (0. - 1.j)*np.matmul(H1 + Gamma(t + float(delta_t))
                                  * H0, psi + float(delta_t)*k3)
        psi = timeplus1RK(dim, psi, k1, k2, k3, k4, delta_t)
        t += delta_t
        t_for_overlap += 1

        if (t_for_overlap % delta_t_overlap == 0 and i < number_of_overlaps - 1):
            i += 1
            w, v = np.linalg.eigh(H)
            for k in range(number_of_eigenstates):
                vec = np.conjugate(v[:, k])
                dot = np.dot(psi, vec)
                overlap[0][k][i] = (
                    np.real(dot)*np.real(dot) + np.imag(dot)*np.imag(dot))/float(dim)
                energy[0][k][i] = w[k]

    return psi


#--------------------Evolutions with A(t), B(t) (Runge-Kutta)----------------------#
# Simulate the evolution and plot probability coeffiecients over time
def evolutionABRK2(dim, H0, H1, psi, A, B, successProbability, gs, t_f, delta_t):
    t = 0
    iteration = 0
    probability = np.empty(dim)
    xAxis = np.linspace(0, dim - 1, dim)

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][0] += np.real(psi[i]) * \
            np.real(psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][0] = successProbability[0][0]/float(dim)

    plot = plt.figure()

    while (t < t_f - delta_t):
        k1 = (0. - 1.j)*np.matmul(A(t/t_f)*H0 + B(t/t_f)*H1, psi)
        k2 = (0. - 1.j)*np.matmul(A((t + delta_t/2) / t_f)*H0 +
                                  B((t + delta_t/2) / t_f)*H1, psi + delta_t/2 * k1)
        k3 = (0. - 1.j)*np.matmul(A((t + delta_t/2) / t_f)*H0 +
                                  B((t + delta_t/2) / t_f)*H1, psi + delta_t/2 * k2)
        k4 = (0. - 1.j)*np.matmul(A((t + delta_t) / t_f)*H0 +
                                  B((t + delta_t) / t_f)*H1, psi + delta_t * k3)
        psi = psi + float(1/6)*delta_t*(k1 + 2*k2 + 2*k3 + k4)
        t += delta_t
        iteration += 1

        if (iteration % 1 == 0):
            successProbability[0] = np.append(successProbability[0], 0.)
            for i in gs:
                successProbability[0][-1] += np.real(psi[i])*np.real(
                    psi[i]) + np.imag(psi[i])*np.imag(psi[i])

            successProbability[0][-1] = successProbability[0][-1]/float(dim)

        # if (iteration%50 == 0):
        #     for k in range(dim):
        #         probability[k] = (np.real(psi[k])*np.real(psi[k]) + np.imag(psi[k])*np.imag(psi[k]))/float(dim)

        #     plt.xlim(0, dim - 1)9
        #     plt.ylim(0, 1)
        #     plt.plot(xAxis, probability, color = 'black')
        #     plt.pause(0.0000001)
        #     plt.clf()

        #     print(str(t) + '   ' + str(successProbability[0][iteration]))

    return psi


# Simulate the evolution with fixed time
def evolutionABRK3(dim, H0, H1, psi, A, B, successProbability, gs, t_f, delta_t):
    t = 0
    iteration = 0

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][0] += np.real(psi[i]) * \
            np.real(psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][0] = successProbability[0][0]/float(dim)

    while (t < t_f - delta_t):
        k1 = (0. - 1.j)*np.matmul(A(t/t_f)*H0 + B(t/t_f)*H1, psi)
        k2 = (0. - 1.j)*np.matmul(A((t + delta_t/2) / t_f)*H0 +
                                  B((t + delta_t/2) / t_f)*H1, psi + delta_t/2 * k1)
        k3 = (0. - 1.j)*np.matmul(A((t + delta_t/2) / t_f)*H0 +
                                  B((t + delta_t/2) / t_f)*H1, psi + delta_t/2 * k2)
        k4 = (0. - 1.j)*np.matmul(A((t + delta_t) / t_f)*H0 +
                                  B((t + delta_t) / t_f)*H1, psi + delta_t * k3)
        psi = timeplus1RK(dim, psi, k1, k2, k3, k4, delta_t)
        t += delta_t
        iteration += 1

        successProbability[0] = np.append(successProbability[0], 0.)
        for i in gs:
            successProbability[0][iteration] += np.real(psi[i])*np.real(
                psi[i]) + np.imag(psi[i])*np.imag(psi[i])

        successProbability[0][iteration] = successProbability[0][iteration] / \
            float(dim)

    return psi


# Simulate the evolution until a desired probability is achieved
def evolutionABRK4(dim, H0, H1, psi, A, B, successProbability, pSuccess, gs, t_f, delta_t):
    t = 0
    iteration = 0

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][0] += np.real(psi[i]) * \
            np.real(psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][0] = successProbability[0][0]/float(dim)

    while (t < t_f - delta_t):
        k1 = (0. - 1.j)*np.matmul(A(t/t_f)*H0 + B(t/t_f)*H1, psi)
        k2 = (0. - 1.j)*np.matmul(A((t + delta_t/2) / t_f)*H0 +
                                  B((t + delta_t/2) / t_f)*H1, psi + delta_t/2 * k1)
        k3 = (0. - 1.j)*np.matmul(A((t + delta_t/2) / t_f)*H0 +
                                  B((t + delta_t/2) / t_f)*H1, psi + delta_t/2 * k2)
        k4 = (0. - 1.j)*np.matmul(A((t + delta_t) / t_f)*H0 +
                                  B((t + delta_t) / t_f)*H1, psi + delta_t * k3)
        psi = timeplus1RK(dim, psi, k1, k2, k3, k4, delta_t)
        t += delta_t
        iteration += 1

        successProbability[0] = np.append(successProbability[0], 0.)
        for i in gs:
            successProbability[0][iteration] += np.real(psi[i])*np.real(
                psi[i]) + np.imag(psi[i])*np.imag(psi[i])

        successProbability[0][iteration] = successProbability[0][iteration] / \
            float(dim)

    t_f[0] = t

    return psi


# Simulate the evolution and compute the overlap with the instantaneous ground state
def evolutionABRK5(dim, H0, H1, psi, A, B, overlap, gs, t_f, delta_t):
    t = 0
    degeneracy = len(gs)
    iteration = 0

    w, v = np.linalg.eigh(A(t/t_f)*H0 + B(t/t_f)*H1)
    prod = 0
    for i in range(degeneracy):
        vec = np.conjugate(v[:, i])
        dot = np.dot(psi, vec)
        prod += (np.real(dot)*np.real(dot) +
                 np.imag(dot)*np.imag(dot))/float(dim)

    overlap[0] = np.append(overlap[0], prod)

    while (t < t_f - delta_t):
        iteration += 1
        H = A(t/t_f)*H0 + B(t/t_f)*H1
        k1 = (0. - 1.j)*np.matmul(A(t/t_f)*H0 + B(t/t_f)*H1, psi)
        k2 = (0. - 1.j)*np.matmul(A((t + delta_t/2) / t_f)*H0 +
                                  B((t + delta_t/2) / t_f)*H1, psi + delta_t/2 * k1)
        k3 = (0. - 1.j)*np.matmul(A((t + delta_t/2) / t_f)*H0 +
                                  B((t + delta_t/2) / t_f)*H1, psi + delta_t/2 * k2)
        k4 = (0. - 1.j)*np.matmul(A((t + delta_t) / t_f)*H0 +
                                  B((t + delta_t) / t_f)*H1, psi + delta_t * k3)
        psi = timeplus1RK(dim, psi, k1, k2, k3, k4, delta_t)
        t += delta_t

        if (iteration % 10 == 0):
            w, v = np.linalg.eigh(H)
            prod = 0
            for i in range(degeneracy):
                vec = np.conjugate(v[:, i])
                dot = np.dot(psi, vec)
                prod += (np.real(dot)*np.real(dot) +
                         np.imag(dot)*np.imag(dot))/float(dim)
            overlap[0] = np.append(overlap[0], prod)

    return psi


# Simulate the evolution and compare with adiabatic evolution
def evolutionABRK6(dim, H0, H1, psi, A, B, energy, overlap, t_f, delta_t, number_of_overlaps, number_of_eigenstates):
    t = 0
    delta_t_overlap = int(t_f/(delta_t*number_of_overlaps))
    t_for_overlap = 0
    i = 0

    H = A(t/t_f)*H0 + B(t/t_f)*H1
    w, v = np.linalg.eigh(H)
    for k in range(number_of_eigenstates):
        vec = np.conjugate(v[:, k])
        dot = np.dot(psi, vec)
        overlap[0][k][0] = (np.real(dot)*np.real(dot) +
                            np.imag(dot)*np.imag(dot))/float(dim)
        energy[0][k][0] = w[k]

    while (t < t_f - delta_t):
        H = A(t/t_f)*H0 + B(t/t_f)*H1
        k1 = (0. - 1.j)*np.matmul(A(t/t_f)*H0 + B(t/t_f)*H1, psi)
        k2 = (0. - 1.j)*np.matmul(A((t + delta_t/2) / t_f)*H0 +
                                  B((t + delta_t/2) / t_f)*H1, psi + delta_t/2 * k1)
        k3 = (0. - 1.j)*np.matmul(A((t + delta_t/2) / t_f)*H0 +
                                  B((t + delta_t/2) / t_f)*H1, psi + delta_t/2 * k2)
        k4 = (0. - 1.j)*np.matmul(A((t + delta_t) / t_f)*H0 +
                                  B((t + delta_t) / t_f)*H1, psi + delta_t * k3)
        psi = timeplus1RK(dim, psi, k1, k2, k3, k4, delta_t)
        t += delta_t
        t_for_overlap += 1

        if (t_for_overlap % delta_t_overlap == 0 and i < number_of_overlaps - 1):
            i += 1
            w, v = np.linalg.eigh(H)
            for k in range(number_of_eigenstates):
                vec = np.conjugate(v[:, k])
                dot = np.dot(psi, vec)
                overlap[0][k][i] = (
                    np.real(dot)*np.real(dot) + np.imag(dot)*np.imag(dot))/float(dim)
                energy[0][k][i] = w[k]

    return psi


# Compute the overlap assuming adiabatic evolution
def adiabaticEvolution(dim, H0, H1, Gamma, adiabatic, gs, t_f, delta_t):
    times = np.linspace(0, t_f, 100)

    for t in times:
        H = Gamma(t)*H0 + H1
        w, v = np.linalg.eigh(H)
        vec = np.conjugate(v[:, 0])
        adiabatic[0] = np.append(adiabatic[0], 0.)
        for i in gs:
            adiabatic[0][-1] += np.real(vec[i])*np.real(vec[i]) + \
                np.imag(vec[i])*np.imag(vec[i])


# Compute the overlap assuming adiabatic evolution
def adiabaticEvolutionAB(dim, H0, H1, A, B, adiabatic, gs, t_f):
    sVector = np.linspace(0, 1, 100)

    for s in sVector:
        H = A(s)*H0 + B(s)*H1
        w, v = np.linalg.eigh(H)
        vec = np.conjugate(v[:, 0])
        adiabatic[0] = np.append(adiabatic[0], 0.)
        for i in gs:
            adiabatic[0][-1] += np.real(vec[i])*np.real(vec[i]) + \
                np.imag(vec[i])*np.imag(vec[i])


def gap(dim, H0, H1, A, B, gs):
    sVector = np.linspace(0, 1, 100)
    degeneracy = len(gs)
    minimum_gap = 1e100

    for s in sVector:
        H = A(s)*H0 + B(s)*H1
        energy = np.linalg.eigvalsh(H)
        if (minimum_gap > energy[degeneracy] - energy[0]):
            minimum_gap = energy[degeneracy] - energy[0]

    return minimum_gap


# Compute the ground states of a given Hamiltonian
def groundState(dim, H):
    mini = H[0]

    for i in range(dim):
        if (H[i] < mini):
            mini = H[i]

    gs = []

    for i in range(dim):
        if (H[i] == mini):
            gs.append(i)

    return gs


# Simulate the evolution and plot probability coeffiecients over time
def evolutionABRK62(dim, H0, H1, psi, A, B, successProbability, gs, t_f, delta_t):
    t = 0
    iteration = 0

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][0] += np.real(psi[i]) * \
            np.real(psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][0] = successProbability[0][0]/float(dim)

    while (t < t_f - delta_t):
        k1 = (0. - 1.j)*np.matmul(A(t/t_f)*H0 + B(t/t_f)*H1, psi)*delta_t
        k2 = (0. - 1.j)*np.matmul(A((t + delta_t/2)/t_f)*H0 +
                                  B((t + delta_t/2)/t_f)*H1, psi + k1/2)*delta_t
        k3 = (0. - 1.j)*np.matmul(A((t + delta_t/2)/t_f)*H0 +
                                  B((t + delta_t/2)/t_f)*H1, psi + k1/4 + k2/4)*delta_t
        k4 = (0. - 1.j)*np.matmul(A((t + delta_t)/t_f)*H0 +
                                  B((t + delta_t)/t_f)*H1, psi - k2 + 2*k3)*delta_t
        k5 = (0. - 1.j)*np.matmul(A((t + 2*delta_t/3)/t_f)*H0 + B((t +
                                                                   2*delta_t/3)/t_f)*H1, psi + 7*k1/27 + 10*k2/27 + 1*k4/27)*delta_t
        k6 = (0. - 1.j)*np.matmul(A((t + delta_t/5)/t_f)*H0 + B((t + delta_t/5)/t_f)
                                  * H1, psi + 28*k1/625 - k2/5 + 546*k3/625 + 54*k4/625 - 378*k5/625)*delta_t
        psi = psi + (k1/24 + 5*k4/48 + 27*k5/56 + 125*k6/336)
        t += delta_t
        iteration += 1

        if (iteration % 1 == 0):
            successProbability[0] = np.append(successProbability[0], 0.)
            for i in gs:
                successProbability[0][-1] += np.real(psi[i])*np.real(
                    psi[i]) + np.imag(psi[i])*np.imag(psi[i])

            successProbability[0][-1] = successProbability[0][-1]/float(dim)

    return psi


# Simulate the evolution and plot probability coeffiecients over time
def evolutionABRK22(dim, H0, H1, psi, A, B, successProbability, gs, t_f, delta_t):
    t = 0
    iteration = 0

    successProbability[0] = np.append(successProbability[0], 0.)
    for i in gs:
        successProbability[0][0] += np.real(psi[i]) * \
            np.real(psi[i]) + np.imag(psi[i])*np.imag(psi[i])

    successProbability[0][0] = successProbability[0][0]/float(dim)

    while (t < t_f - delta_t):
        k1 = (0. - 1.j)*np.matmul(A(t/t_f)*H0 + B(t/t_f)*H1, psi)
        k2 = (0. - 1.j)*np.matmul(A((t + delta_t/3)/t_f)*H0 +
                                  B((t + delta_t/3)/t_f)*H1, psi + delta_t*k1/3)
        k3 = (0. - 1.j)*np.matmul(A((t + 2*delta_t/3)/t_f)*H0 +
                                  B((t + 2*delta_t/3)/t_f)*H1, psi + 2*delta_t*k2/3)
        psi = psi + delta_t*(k1 + 3*k3)/4
        t += delta_t
        iteration += 1

        if (iteration % 1 == 0):
            successProbability[0] = np.append(successProbability[0], 0.)
            for i in gs:
                successProbability[0][-1] += np.real(psi[i])*np.real(
                    psi[i]) + np.imag(psi[i])*np.imag(psi[i])

            successProbability[0][-1] = successProbability[0][-1]/float(dim)

    return psi
