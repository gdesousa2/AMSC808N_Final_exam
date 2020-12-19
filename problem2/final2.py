import numpy as np
import matplotlib.pyplot as plt
from random import sample
from scipy.optimize import root
from mpmath import polylog
from utils import *

################################################################################
# UNIVERSITY OF MARYLAND - FALL 2020                                           #
# FINAL EXAM - AMSC808N: NUMERICAL METHODS FOR MACHINE LEARNING & DATA SCIENCE #
# GUILHERME DE SOUSA - Physics PhD student                                     #
#                                                                              #
# FINAL EXAM: PROBLEM 2 - EPIDEMIC ANALYSYS                                    #
################################################################################

# Generating functions #
def G0(x):
    return Li(al,x) / Li(al,1)

def G1(x):
    return Li(al-1,x) / (x * Li(al-1, 1))

def find_giant(x,T):
    return x - G1(1-T+T*x)

# Power-law distribution Random Graph #
al = 2.2
T = 0.4
Tc = Li(al-1,1) / (Li(al-2,1) - Li(al-1,1))
print(f'Critical epidemic alpha = {al}, T = {T}: Tc = {Tc:.2f}')

# Fraction of nodes in giant component
u = root(find_giant, x0=0.5, args=1).x[-1]
S = 1 - G0(u)
print(f'Giant component alpha = {al}, T = {T}: S = {S:.2f}')

# Fraction of infected nodes
uT = root(find_giant, x0=0.5, args=T).x[-1]
ST = 1 - G0(1-T+T*uT)
print(f'Infected nodes alpha = {al}, T = {T}: S(T) = {ST:.2f}')

# Critical vaccination
vc = max(0, 1 - Tc/T)
print(f'Critical vaccination alpha = {al}, T = {T}: vc = {vc:.2f}')
print()
print()

# POWER-LAW NETWORK SIMULATIONS #
# FRACTION OF INFECTED T = 0.4
stat = 1000 # statistics for average random graph
n = 2000
Ttime = 500
giant = np.zeros(stat)
infected = np.zeros(stat)
for k in range(stat):
    graph, edges = MakePowerLawRandomGraph(n,al)
    giant[k] = frac_giant(graph)
    infected[k] = evolve_graph(graph, edges, Ttime, T, v=0)[1]/n
print(f'(Average) Fraction giant component alpha = {al}, T = {T}: ', end='')
print(f'S = {np.mean(giant):.3f} +/- {np.std(giant):.3f}')
print(f'(Average) Fraction infected nodes alpha = {al}, T = {T}: ', end='')
print(f'I = {np.mean(infected):.3f} +/- {np.std(infected):.3f}')


# CRITICAL VALUE Tc, NO VACCINATION
stat = 100
n = 2000
Ttime = 500
graph, edges = MakePowerLawRandomGraph(n,al)
T_array = np.linspace(0, 1, 20)
infected_Tc = np.zeros(len(T_array))
for k in range(len(T_array)):
    s = 0
    for i in range(stat):
        s += evolve_graph(graph, edges, Ttime, T_array[k], v=0)[1]/n
    infected_Tc[k] = s / stat

plt.figure()
plt.plot(T_array, infected_Tc, '^--', ms=14, label=f'{n} nodes')
plt.xlabel('Transmission $T$')
plt.ylabel('Infected fraction')
plt.legend()
plt.savefig('transmission_crit.png')


# CRITICAL VALUE vc, VACCINATION w/ T = 0.4
stat = 100
n = 2000
Ttime = 500
#graph, edges = MakePowerLawRandomGraph(n,al)
v_array = np.linspace(0, 1, 20)
infected_vc = np.zeros(len(v_array))
for k in range(len(v_array)):
    s = 0
    for i in range(stat):
        s += evolve_graph(graph, edges, Ttime, T=0.4, v=v_array[k])[1]/n
    infected_vc[k] = s / stat

plt.figure()
plt.plot(T_array, infected_vc, '^--', ms=14, label=f'{n} nodes, $T = 0.4$')
plt.xlabel('Vaccination fraction $v$')
plt.ylabel('Infected fraction')
plt.legend()
plt.savefig('vaccination_crit.png')


# DISCRETE-TIME SIR SIMULATION
stat = 100 
Ttime = 500
n_time = [100, 500, 1000, 2000, 5000]
infected_time = [np.zeros((stat,Ttime)) for i in range(len(n_time))]
infected_time_total = [np.zeros(stat) for x in range(len(n_time))]
for m in range(len(n_time)):
    graph, edges = MakePowerLawRandomGraph(n_time[m],al)
    for i in range(stat):
        time_series, tot = evolve_graph(graph, edges, Ttime, T=0.4, v=0)
        time_series = time_series/n
        tot = tot/n
        infected_time[m][i,:len(time_series)] = time_series
        infected_time_total[m][i] = tot

plt.figure()
plt.plot(infected_time[-1][0,:], color='C3', label=f'{n_time[-1]} nodes, $T = 0.4$')
for k in range(1,stat):
    plt.plot(infected_time[-1][k,:], color='C3')
plt.xlabel('Time')
plt.ylabel('Fraction infected per time step')
plt.xlim(0,25)
plt.legend()
plt.savefig('All_100_time.png')

plt.figure()
for i in range(len(n_time)):
    plt.plot(np.mean(infected_time[i], axis=0), label=f'{n_time[i]} nodes, $T = 0.4$')
plt.xlim(0,25)
plt.xlabel('Time')
plt.ylabel('Fraction infected per time step')
plt.legend()
plt.savefig('Average_100_time.png')

plt.figure()
plt.plot(n_time, [np.mean(x) for x in infected_time_total], '^--', ms=14, label=f'$T = 0.4$')
plt.xlabel('Graph size (nodes)')
plt.ylabel('Total Fraction infected')
plt.legend()
plt.savefig('Total_infected.png')