import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def fraction (t):
    return 100/(t+1)

def exponential (t):
    return 100/np.exp(t)

def exponentialsqrt (t):
    return 100/np.exp(np.sqrt(t))

def linearA (s):
    return 1 - s

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
    




# t_max = 10
# h = 0.01
# N = int(t_max/h)
# y = np.empty(N)
# y[0] = 1
# t = 0.
# for n in range(N - 1):
#     t += h
#     k1 = f(t, y[n])
#     k2 = f(t + h/2, y[n] + h*k1/2)
#     k3 = f(t + h/2, y[n] + h*k2/2)
#     k4 = f(t + h, y[n] + h*k3)
#     y[n + 1] = y[n] + h*(k1 + 2*k2 + 2*k3 + k4)/6
    

# xAxis = np.linspace(0, t_max, N)
# z = (1/4)*(-2*xAxis + 5*np.exp(2*xAxis) - 1)

# plt.figure(1)
# plt.plot(xAxis, z)
# plt.plot(xAxis, y)
    
    
    
    
    
    
    
    
    