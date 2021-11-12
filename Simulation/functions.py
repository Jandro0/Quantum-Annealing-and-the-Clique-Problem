import numpy as np
import matplotlib.pyplot as plt

def fraction (t):
    return 100/(t+1)

def exponential (t):
    return 100/np.exp(t)

def exponentialsqrt (t):
    return 100/np.exp(np.sqrt(t))

def linearA (t, t_f):
    return 1 - float(t/t_f)

def linearB (t, t_f):
    return float(t/t_f)



def f(t, y, H0, H1, A, B, t_f):
    return (0. - 1.j)*np.matmul(A(t, t_f)*H0 + B(t, t_f)*H1, y)
    
    

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
    
    
    
    
    
    
    
    
    