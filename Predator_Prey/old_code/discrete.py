import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def plotStats(stats):
    plt.figure(figsize=(12, 6))
    t = np.arange(0, len(stats))
    plt.plot(t, stats[:,0], 'r-d')
    plt.plot(t, stats[:,1], 'b-o')
    plt.xlabel('Iterations')
    plt.xticks(np.arange(0, len(stats)))
    plt.grid()
    plt.show()
    
def plotWorld(H, L):
    if (len(L) > 0):
        plt.scatter(L[:, 0] -.5, L[:, 1]-.5, c='r', s=100, marker='d')
    if (len(H) > 0):
        plt.scatter(H[:, 0] -.5, H[:, 1] -.5, c='b')
    plt.xticks(np.arange(0, N))
    plt.yticks(np.arange(0, N))
    plt.grid()
    plt.show()

# Predator and prey movement    
def moveL(L): 
    return (L + np.round(np.random.uniform(-3, 3, size=L.shape))) % N

def moveH(H): 
    return (H + np.round(np.random.uniform(-2, 2, size=H.shape))) % N
    
# Repoduction
def reproduce(P, prob):
    for i in range(len(P)):
        if (np.random.uniform(0, 1) <= prob):
            #P = np.vstack((P, [P[i]]))            
            P = np.vstack((P, np.random.randint(N, size=2)))
    return P

# Death prey
def death(P, prob):
    rp = []
    for i in range(len(P)):
        if (np.random.uniform(0, 1) <= prob):
            rp.append(i)
    return np.delete(P, rp, axis=0)

# Check if wolf eats
def checkFood(L, H, d, probEat, probRep):
    for l in L: # Foreach wolf
        pos = 0 
        for h in H: # Foreach rabbit
            # If rabbit is inside a wolf neighborhood
            if np.linalg.norm(l-h, ord=1) <= d: 
                if (np.random.uniform(0, 1) <= probEat): # Random eat
                    H = np.delete(H, [pos], axis=0) # Remove rabbit
                    
                    # Reproduce wolf if it eats
                    if (np.random.uniform(0, 1) <= probRep):
                        L = np.vstack((L, [l]))
            pos += 1 # pos to handle the rabbits' removal
    return L, H
                    
            

# Parameters
N = 40 # World size
Prc = 1e-1 # Probability of rabbit's reproduction
Pdl = 8e-2 # Probability of wolf's death
Prl = 4e-1 # Probability of wolf's reproduction
Pcl = 5e-1 # Probability of wolf's feeding
d = 2 # Distance for wolf eating

#N = 20
#Prc = 0.3910615977651579 #tasa de reproduccion conejo
#Prl = 0.253552295046609 #tasa de reproduccion lobo
#Pdl = 0.9025033456284932 #tasa de sobrevivencia por dia de lobo hambriento
#Pcl = 0.5

NL = 20 # Number of wolfs
NH = 55 # Number of rabbits

# Random wolfs and rabbits positions
L = np.random.randint(N, size=(NL, 2))
H = np.random.randint(N, size=(NH, 2))

# Iterations
N_iter = 1000

# Keep stats
stats = [[NL, NH]]

for i in range(1, N_iter + 1):
    L = moveL(L) # Move wolfs
    H = moveH(H) # Move rabbits
    H = reproduce(H, Prc) # Reproduce rabbits
    L, H = checkFood(L, H, d, Pcl, Prl) # Wolfs eat rabbits
    L = death(L, Pdl) # Wolfs death after time
    
    # Append tats
    stats.append([len(L), len(H)])
    
    # Show world status
    #plotWorld(H, L)
    
    # Stop iterations 
    if (len(H) >= 0 and len(L) == 0):
        break
    
# Show stats
plotStats(np.array(stats))
    
