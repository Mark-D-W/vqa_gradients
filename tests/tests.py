import vqa_gradients

import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group
import functools as ft
tp = lambda a,b: np.tensordot(a,b,axes=0)

n_qubits = 6

H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
X = np.array([[0,1],[1,0]])

M = ft.reduce(tp, np.repeat(H, n_qubits))
Psi = np.zeros((2**n_qubits,1)); Psi[0] = 1
G = np.diag(np.random.choice([0,1,2,3,4], 2**n_qubits))
R = len(np.unique(np.diff(G)))
rnd1 = unitary_group.rvs(2**n_qubits)
rnd2 = unitary_group.rvs(2**n_qubits)

E = np.vectorize(lambda param:
                 np.real(
                     ft.reduce(np.dot,
                         [Psi.conj().T,
                          rnd1.conj().T,
                          expm(-1j*param*G).conj().T,
                          rnd2.conj().T,
                          G,
                          rnd2.conj().T,
                          expm(1j*param*G),
                          rnd1,
                          Psi])))



x = np.linspace(1,10,num=2*R+1)
y = E(x)
s = vqa_gradients.Series(x,y)
s.plot(function=E)

rnd = np.random.random()*9 + 1
print(s.gradient(rnd))
print(sp.misc.derivative(E,rnd,dx=1e-6)[0,0])
            
