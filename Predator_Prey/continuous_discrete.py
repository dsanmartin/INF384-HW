# Lotka-Volterra

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Continuous 
def F(v, t, a, b, c, d):
    dxdt = v[0]*(a - b*v[1])
    dydt = -v[1]*(c - d*v[0])
    return np.array([dxdt, dydt])

def plot(t, V):
    rabbits, foxes = V.T
    f1 = plt.figure()
    plt.plot(t, rabbits, 'r-', label='Rabbits')
    plt.plot(t, foxes  , 'b-', label='Foxes')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.title('Evolution of fox and rabbit populations')
    plt.show()

a = 0.1
b = 0.02
c = 0.3
d = 0.01

t = np.linspace(0, 200,  1000) 
v0 = np.array([40, 9])  

V = odeint(F, v0, t, args=(a, b, c, d))
plot(t, V)
plt.show()

# Discrete
a = c = 0.014
b = 0.6
d = 0.7
nperiods = 365
duration = 90

K = nperiods*duration

H = np.zeros(K+1)
L = np.zeros(K+1)
H[0], L[0] = 10, 10
year = np.zeros(K+1)
year[0] = 1845

t = []
V = []

for k in range(K):
    H[k+1] = H[k] + (b*H[k] - a*L[k]*H[k])/nperiods
    L[k+1] = L[k] + (c*L[k]*H[k] - d*L[k])/nperiods
    year[k+1] = year[k] + 1/nperiods
    
    if k % nperiods == 1:
        V.append([L[k], H[k]])
        t.append(year[k])
        
V = np.array(V)
plot(t, V)