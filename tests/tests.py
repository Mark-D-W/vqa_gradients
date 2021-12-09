import vqa_gradients

import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group
from scipy.misc import derivative
import functools as ft
tp = lambda a,b: np.tensordot(a,b,axes=0)


n_qubits = 3

H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
X = np.array([[0,1],[1,0]])

M = ft.reduce(tp, np.repeat(H, n_qubits))
Psi = np.zeros((2**n_qubits,1)); Psi[0] = 1
G = np.diag(np.random.choice([1,2,3,4,5,6,7,8,9,10], 2**n_qubits))
W = vqa_gradients.complete_graph(2**n_qubits)/2**n_qubits
#W = np.diag(np.random.choice([1,2,3,4,5,6,7,8,9,10], 2**n_qubits))
R = vqa_gradients.Optimise().find_R_from_qualities(np.linalg.eigvals(W))
rnd1 = unitary_group.rvs(2**n_qubits)
rnd2 = unitary_group.rvs(2**n_qubits)


def E(param):
    U = ft.reduce(np.matmul,
              [rnd2,
               expm(-1j*param*W),
               rnd1,
               Psi])
    U_dagger = U.conj().T
    return ft.reduce(np.matmul, [U_dagger,G,U])
E = np.vectorize(E)
   

x = np.array([(2*np.pi*i)/(2*R+1) for i in range(-R,R+1)])
y = E(x)
s = vqa_gradients.Series(x,y)
s.plot(function=E)

rnd = np.random.random()*6-3
print(s.gradient(rnd))
print(derivative(E,rnd,dx=1e-6)[0,0])
            
