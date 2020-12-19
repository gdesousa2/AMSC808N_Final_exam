import numpy as np
import matplotlib.pyplot as plt
from random import sample

################################################################################
# UNIVERSITY OF MARYLAND - FALL 2020                                           #
# FINAL EXAM - AMSC808N: NUMERICAL METHODS FOR MACHINE LEARNING & DATA SCIENCE #
# GUILHERME DE SOUSA - Physics PhD student                                     #
#                                                                              #
# FINAL EXAM: PROBLEM 1 - FUNCTION APPROXIMATION                               #
################################################################################

# DEFINITIONS #
def relu(x):
    return np.maximum(x, 0)

def g(x):
    return 1 - np.cos(x)

def f(a,b):
    xj = np.array([0, 1, 2, 3, 4, 5]) * np.pi/10
    return np.sum((relu(a*xj-b) - g(xj))**2)/12

xj = np.array([0, 1, 2, 3, 4, 5]) * np.pi/10 # training grid

def gradf(a,b):
    da = np.zeros(len(xj))
    db = np.zeros(len(xj))

    for i in range(len(xj)):
        if a*xj[i]-b > 0:
            da[i] = xj[i]
            db[i] = -1
        else:
            da[i] = 0
            db[i] = 0
    
    aux = relu(a*xj-b) - g(xj)
    return np.sum(aux*da/6), np.sum(aux*db/6)

def gradf_stoc(a,b):
    ind = sample([1, 2, 3, 4, 5, 6], 1)[0]

    if a*xj[ind]-b > 0:
        da = xj[ind]
        db = -1
    else:
        da = 0
        db = 0
    
    aux = relu(a*xj[ind]-b) - g(xj[ind])
    return aux*da, aux*db

# PLOT STATIONARY FLAT REGION
plt.figure()
plt.fill_between([0, 1], [0, 2/np.pi], [-1, -1], where=[0, 2/np.pi] > [-1, -1], 
    color='black', alpha=0.35, label='Flat region')
plt.plot(0.37, 0.86, '*', color=f'black', ms=14, label='Global minimum')
plt.xlabel('b')
plt.ylabel('a')
plt.legend()
plt.savefig('flat_region.png')

# INITIAL GUESS #
a = 1
b = 0

# OPTIMAL SOLUTION
a_opt = np.zeros(5)
b_opt = np.zeros(5)
opt_func = np.zeros(5)
for i in range(5):
    xj_opt = xj[i:]
    a_opt[i] = (np.mean(xj_opt*g(xj_opt)) - 
        np.mean(xj_opt)*np.mean(g(xj_opt)))/(np.mean(xj_opt**2)-np.mean(xj_opt)**2)
    b_opt[i] = -np.mean(g(xj_opt)) + a_opt[i]*np.mean(xj_opt)
    opt_func[i] = f(a_opt[i],b_opt[i])

# PLOT OPTIMAL SOLUTION
xplot = np.linspace(0, np.pi/2, 50)
plt.figure()
plt.plot(xplot, g(xplot), label='$1 - \cos x$')
plt.plot(xplot, relu(a_opt[2]*xplot - b_opt[2]), label='$ReLU(a^* x - b^*)$')
plt.legend()
plt.savefig('optimal_solution.png')

####################
# GRADIENT DESCENT #
####################
# ITERATION
alpha_star = 1/(gradf(1,0)[0] - 2*gradf(1,0)[1]/np.pi)
kmax = 1000
alpha = [1e-1, 1.31, .99*alpha_star, alpha_star]

a_iter = [[a] for i in range(len(alpha))]
b_iter = [[b] for i in range(len(alpha))]

for i in range(len(alpha)):
    for k in range(kmax):
        grad = gradf(a_iter[i][-1],b_iter[i][-1])
        a_iter[i].append(a_iter[i][-1] - alpha[i]*grad[0])
        b_iter[i].append(b_iter[i][-1] - alpha[i]*grad[1])

# CONVERGENCE PLOT
plt.figure()
for i in range(len(alpha)):
    plt.plot(b_iter[-1-i], a_iter[-i-1], '--', 
        color=f'C{3-i}', label=r'$\alpha = ${:.2f}'.format(alpha[-i-1]))
    plt.plot(b_iter[-i-1][-1], a_iter[-i-1][-1], '*', color=f'C{3-i}', ms=14)
plt.plot([0, 1], [0, 2/np.pi], 'k--', label='Boundary')
plt.xlabel('b')
plt.ylabel('a')
plt.legend()
plt.savefig('gradient_descent.png')


#######################
# STOCHASTIC GRADIENT #
#######################
# ITERATION
al_ini = 1.5
r = 50
kmax = 1000

a_iter2 = [1]
b_iter2 = [0]

for k in range(kmax):
    al = al_ini / 2**(k/r)
    grad = gradf(a_iter2[-1],b_iter2[-1])
    a_iter2.append(a_iter2[-1] - al*grad[0])
    b_iter2.append(b_iter2[-1] - al*grad[1])

plt.figure()
plt.plot(b_iter2, a_iter2, '--', 
    color=f'C{i+1}', label=r'$\alpha_k = 1.5*2^{-k/50}$')
plt.plot(b_iter2[-1], a_iter2[-1], '*', color=f'C{i+1}', ms=14)
plt.plot([0, 1], [0, 2/np.pi], 'k--', label='Boundary')
plt.xlabel('b')
plt.ylabel('a')
plt.legend()
plt.savefig('stochastic_grad.png')