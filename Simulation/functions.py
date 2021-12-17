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
    
    
    
    
    
    
    