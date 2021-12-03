import vqa_gradients

import numpy as np
from numpy import tensordot as td

n_qubits = 6

H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
X = np.array([[0,1],[1,0]])

M = td(H,td(H,td(H,td(H,td(H,H)))))
Psi = np.zeros((2**n_qubits,)); Psi[0] = 1
G = np.diag(np.random.choice([0,1,2,3,4]))
R = len(np.unique(np.diff(G)))

E = np.vectorize(
    lambda x:
    np.real( td(td( td(Psi, np.e**(-1j*x*G)), M), td(Psi, np.e**(1j*x*G))) ))



x = np.linspace(1,10,num=2*R+1)
y = E(x)
s = Series(x,y)
s.plot(function=E)
            
