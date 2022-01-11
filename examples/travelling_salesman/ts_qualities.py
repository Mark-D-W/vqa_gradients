# run using: mpirun -N 2 python3 "ts.py"


import numpy as np
import pandas as pd
import networkx as nx
from mpi4py import MPI


comm = MPI.COMM_WORLD

DEBUG=False

###############################################################
###############################################################
## Travelling salesman functions
###############################################################

def ts_cost(n_nodes=5):
    # Define the cost table in the zeroth process only and broadcast it to the other processes
    if comm.Get_rank() == 0:
        # Define the cost table: the cost of moving from node i to node j is given by cost_table[i][j]
        cost_table = [[0 for j in range(0, n_nodes)] for i in range(0, n_nodes)]
        G = nx.Graph()
        for i in range(0, n_nodes):
            for j in range(0, n_nodes):
                if cost_table[j][i] != 0: continue # don't redo assignment
                G.add_node(str(j)) # this will add all the nodes we need
                if (i==j):
                    cost_table[i][j] = 0
                    G.add_edge(str(i), str(j), cost=0)
                else:
                    rand_cost = np.random.randint(low = 1, high = 1000)
                    cost_table[i][j] = rand_cost
                    cost_table[j][i] = rand_cost
                    G.add_edge(str(i), str(j), cost=rand_cost)
                            
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        edge_labels = nx.get_edge_attributes(G, 'cost')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=0.3)
        plt.savefig("./tmp/graph.png")
        #plt.show()
    else:
        cost_table = None

    cost_table = comm.bcast(cost_table, root=0)

    return cost_table




# Effectivly converts base 10 integer num to base new_base. 
# For a system with n nodes, we can represent an arbitray path by a number in base n. Hence we can enumerate all the possible paths in base to which is easy to work with and still have a bijection to the corresponding path via base conversion.
def to_path(num, new_base, max_path_length):
    path = []

    for i in range(max_path_length-1, -1, -1): # count down to 0 from the maximum path length
        if num >= new_base**i:
            path_elem = math.floor(num / new_base**i)
            path.append(path_elem)
            num -= path_elem * new_base**i
        else:
            path.append(0)
            #print("num = {} : Path = {}".format(num_tmp, path))        

    return path




def num_elems_less(less_than, elems):
    lst = elems.copy()
    lst.sort()

    for i in range(len(lst)):
        if lst[i] >= less_than:
            #print("lst={}; less_than={}; i={}".format(lst, less_than, i))
            return i

                
def to_ts_path(index, permutations, DEBUG=False):
    path = []
    idx = index
    
    elems = permutations.copy()
    elems.sort()
    elems.reverse()

        # Take the largest element to be the starting and ending element. This eliminated degeneracy in closed paths starting at different elements.
    path.append(max(elems))
    elems.remove(max(elems))
        
    for i in range(len(elems)):
        if DEBUG: print("\n\tDigit {}".format(i))
        for elem in elems:
            if DEBUG: print("Is it {}?".format(elem))
                        
            max_num_smaller_perms = num_elems_less(elem, elems)*(math.factorial(len(elems)-1))
                        
            if DEBUG: print("There exists {}({}-1)! = {} smaller options".format(num_elems_less(elem, elems), len(elems), max_num_smaller_perms))
                        
            if (idx >= max_num_smaller_perms):
                idx -= max_num_smaller_perms
                path.append(elem)
                elems.remove(elem)
                elems.sort()
                elems.reverse()
                if DEBUG: print("Therefore it is {}, so: path={}; idx={}; elems={}\n".format(elem, path, idx, elems))
                break
            else:
                if DEBUG: print("Therefore not {}\n".format(elem))

    path.append(path[0]) # Make the path closed
    if DEBUG: print(path)
    return np.array(path)




                
        
def make_qualities(N, local_i, local_i_offset, cost_table, seed=1, binary_qualities=True, binary_cutoff=3000):
    ret_lst = np.zeros(local_i)
    penalty = max(max(cost_table))

    for i in range(local_i_offset, local_i_offset + local_i):
        valid_path = (i < math.factorial(n_nodes))
        path = to_ts_path(i, list(range(n_nodes-1, -1, -1)))
            
        path_cost = 0
        if valid_path:
            for idx in range(0, len(path)-1):
                path_cost += cost_table[path[idx]][path[idx+1]] 
            else:
                for idx in range(0, len(path)-1):
                    path_cost += penalty

            if not binary_qualities:
                ret_lst[i - local_i_offset] = int(path_cost)
            else:
                ret_lst[i - local_i_offset] = 1 if int(path_cost)<binary_cutoff else -1
                    
    return ret_lst





def ts_qualities():
    return None




if __name__ == "__main__":
    cost_table = ts_cost()
    make_qualities(cost_table=cost_table)
