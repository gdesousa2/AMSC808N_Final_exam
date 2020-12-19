import numpy as np
from numpy.random import choice
from random import sample

# Vertex class
class Vertex:
    def __init__(self, color, d, parent):
        self.color = color
        self.d = d
        self.parent = parent
        self.f = d

# Depth-First Search algorithm
def DFS(graph):
    v_list = {u:Vertex('white',0,'NIL') for u in graph}
    time = 0

    for v in graph:
        if v_list[v].color == 'white':
            time, v_list = DFS_VISIT(graph,v_list,v,time)
    
    return v_list
    
# Depth-First Search - Auxiliary algorithm
def DFS_VISIT(graph, v_list, node, time):
    time = time + 1
    v_list[node].d = time
    v_list[node].color = 'gray'
    for v in graph[node]:
        if v_list[v].color == 'white':
            v_list[v].parent = node
            time, v_list = DFS_VISIT(graph, v_list, v, time)
    v_list[node].color = 'black'
    time = time + 1
    v_list[node].f = time

    return time, v_list

def frac_giant(graph):
    # find connected components
    n = len(graph)
    v_list = DFS(graph)
    conn_list = np.zeros(n)
    for i in range(n):
      if v_list[i].parent == 'NIL':
        conn_list[i] = 1
    ind_not_root = np.argwhere(conn_list < 1)
    ind_not_root = [x[0] for x in ind_not_root]
    conn_list = np.ones(n)

    iter = 0
    itermax = n
    while [x[0] for x in np.argwhere(conn_list < 1)] != ind_not_root:
      for j in ind_not_root:
        conn_list[v_list[j].parent] += conn_list[j]
        conn_list[j] = 0
      if iter > itermax: break
      iter += 1
    
    S = np.sort(conn_list)[-1]/n
    return S

# Polylogarithm function
def Li(al,x):
    s = 0
    for k in range(1,40):
        s += x**k/k**al
    return s

# Creates a power-law distribution graph
def MakePowerLawRandomGraph(n,a):
    # Input:
    # n = number of vertices
    # a = exponent of power-law distribution pk ~ k^(-a)
    #
    # Output:
    # G = adjacency list
    # edges = edge list
    K = int(np.round(np.power(n,1/(a-1))))
    k = np.array([x+1 for x in range(K)], dtype='float64')
    p = k**(-a)
    psum = np.sum(p)
    p = p / psum # renormalized distribution

    graph = {x:[] for x in range(n)}
    edges = []
    degree = np.zeros(n)

    for i in range(n):
        degree[i] = choice([x for x in range(1,K+1)],1,p=p)

    deg = degree
    kmax = 15 * n * Li(a-1,1) / Li(a,1)
    k = 0
    while (np.sum(deg)/n > 0.05) and (k < kmax):
        k = k + 1
        n1 = choice([x for x in range(n)], p=deg/np.sum(deg))
        n2 = choice([x for x in range(n)], p=deg/np.sum(deg))
        if (n1 == n2) or ([n1, n2] in edges):
            continue
        edges.append([n1, n2])
        graph[n1].append(n2)
        graph[n2].append(n1)
        deg[n1] = deg[n1] - 1
        deg[n2] = deg[n2] - 1
    
    return graph, edges

# dynamics on infected nodes
def evolve_graph(graph, edges, Ttime, T, v):
    n = len(graph)
    orig = sample([x for x in range(n)], 1)[0]
    vaccinated = sample([x for x in range(n)], int(v*n))
    recovered = []
    infected = [[orig]] # initial infected node

    t = 0
    while (infected[-1] != []) and (t < Ttime):
        inf = []
        for i in infected[-1]:
            for j in graph[i]:
                if (not (j in vaccinated)) and (not (j in recovered)):
                    if np.random.rand() < T:
                        inf.append(j)
        inf = list(set(inf))
        recovered = recovered + infected[-1]
        infected = infected + [inf]
        t = t + 1
    
    return np.array([len(x) for x in infected]), len(set(recovered))



# Fraction of infected nodes
def num_infected(edges,T):
    infected = []
    for e in edges:
        if np.random.rand() < T:
            infected = infected + e
    infected = set(infected)

    return len(infected)
