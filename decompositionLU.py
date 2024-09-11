import numpy as np
#from scipy.linalg import lu


def decompositionLU(M):
    n = M.shape[0] #tamaño de la matriz
    U = M.copy() #Asigna U como una copia de la matriz M
    L = np.eye(n) #prepara la diagonal de unos para L

    for col in range(n-1): #itera por columas
        for fil in range(col+1, n): #para cada fila de dicha columna
            factor = U[fil][col] / U[col][col]  #define el factor por el cual se operan las filas en la diagonalización
            L[fil][col] = factor #actualiza la fila de L con el el factor
            U[fil] -= U[col] * factor #actualiza las filas en U haciendo que esté más cerca de ser triangular superior

    return L, U
    

def positiva(M):
    #Devuelve True si M es definida positiva
    n = M.shape[0]
    res = True
    for i in range(2,n+1):
        sub_matrix = M[:i,:i]
        det_sub = np.linalg.det(sub_matrix)
        res = res and det_sub>0
    
    return res



