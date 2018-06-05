import numpy as np
import matplotlib.pyplot as plt

# Laplacian with periodic boundary condition
def laplacian(f):
  return np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) + \
    np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) - 4*f
  
def f(rho, gamma):
  return rho * gamma #rho + gamma
  
def F(rho, gamma, f, c, m, hr, hs, lr, ld, h):
  lrho = (1/h) * laplacian(rho)
  lgamma = (1/h) * laplacian(gamma)
  
  f1 = rho * (hr - f(rho, gamma) * hs) + (m/8) * lrho
  f2 = gamma * (f(rho, gamma) * hs * lr - ld) + (m/8) * lgamma
  
  return (1/c) * f1,  (1/c) * f2


def euler(rho0, gamma0, dt, T, parameteres):
  c = parameteres['c']
  m = parameteres['m']
  hr = parameteres['hr']
  hs = parameteres['hs']
  lr = parameteres['lr']
  ld = parameteres['ld']
  h = parameteres['h']
  
  rho_sol = np.zeros((T, rho0.shape[0], rho0.shape[1]))
  gamma_sol = np.zeros((T, gamma0.shape[0], gamma0.shape[1]))
  rho_sol[0], gamma_sol[0] = rho0, gamma0
  
  for t in range(1, T):
    new_rho, new_gamma = F(rho_sol[t-1], gamma_sol[t-1], f, c, m, hr, hs, lr, ld, h)
    rho_sol[t] = rho_sol[t-1] + new_rho * dt
    gamma_sol[t] = gamma_sol[t-1] + new_gamma * dt

  return rho_sol, gamma_sol

# Parameters
Tmax = 1
N = 100
T = 2000
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
t = np.linspace(0, Tmax, T)
h = x[1] - x[0]
dt = t[1] - t[0]

parameters = {
  'c': dt/h**2,
  'm': .1,
  'hr': 1e-1,
  'hs': 1e-1,
  'lr': 1e-1,
  'ld': 1e-1,
  'h': h    
}

X, Y = np.meshgrid(x, y)

# Initial conditions
R = lambda x, y: (np.sin(x) * np.cos(x - y)) ** 2
G = lambda x, y: (np.sin(y + x) * np.cos(x)) ** 2
rho0 = R(X, Y)
gamma0 = G(X, Y)

#rho0 = np.random.rand(N, N)
#gamma0 = np.random.rand(N, N)


# Solve with Euler method
rhos, gammas = euler(rho0, gamma0, dt, T, parameters)

pR = np.zeros(T)
pG = np.zeros(T)

for j in range(T):
  pR[j] = np.sum(rhos[j])
  pG[j] = np.sum(gammas[j])
  
plt.plot(t, pR)
plt.plot(t, pG)
plt.show()

# Plots
for i in range(T):
  if i % 10 == 0:
    plt.subplot(1, 2, 1)
    cf1 = plt.contourf(x, y, rhos[i])
    plt.colorbar(cf1)
    plt.title("Hares")
    plt.subplot(1, 2, 2)    
    cf2 = plt.contourf(x, y, gammas[i])
    plt.colorbar(cf2)
    plt.title("Lynxes")
    plt.tight_layout()
    plt.show()