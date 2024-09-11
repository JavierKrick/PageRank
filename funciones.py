import numpy as np
from decompositionLU import decompositionLU
from scipy.linalg import solve_triangular

def leer_archivo(input_file_path):

    f = open(input_file_path, 'r')
    n = int(f.readline())
    m = int(f.readline())
    W = np.zeros(shape=(n,n))
    for _ in range(m):
        line = f.readline()
        i = int(line.split()[0]) - 1
        j = int(line.split()[1]) - 1
        W[j,i] = 1.0
    f.close()
    
    return W



def calcularRanking(M, p):
    npages = M.shape[0]
    rnk = np.arange(0, npages) # ind{k] = i, la pagina k tienen el iesimo orden en la lista.
    scr = np.zeros(npages) # scr[k] = alpha, la pagina k tiene un score de alpha 
    # CODIGO
    I = np.eye(npages)

    #Armo c_j
    sum_links = M.sum(axis=0)

    D = np.zeros((npages,npages))

    #Construyo a la matriz D
    for j in range(npages):
        cj = sum_links[j]
        if cj!=0:
            D[j,j] = 1/cj
        else :
            D[j,j] = 0
    
    A = I - p * M @ D
    L,U = decompositionLU(A)
    e = np.ones((npages,1))

    #Resuelov la sig ecuacion
    #Ux = y
    #Ly = b
    y = solve_triangular(L,e,lower=True)
    x = solve_triangular(U,y)

    #Score normalizado
    scr = x.flatten()/x.sum()

    #Calculo el ranking utilizando de forma auxiliar los indices de la lista de score ordenado
    aux_rnk = list(np.sort(scr))
    rnk = []
    for s in scr:
        rank = np.abs(aux_rnk.index(s)-npages+1)
        rnk.append(rank)

    return rnk, scr

def obtenerMaximoRankingScore(M, p):
    output = -np.inf
    # calculo el ranking y los scores
    rnk, scr = calcularRanking(M, p)
    output = np.max(scr)
    
    return output

