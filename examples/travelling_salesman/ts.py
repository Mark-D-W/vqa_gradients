# This is a program to simulate the collatz conjecture using the qwoa algorithm
# run using: mpirun -N 2 python3 "ts.py"

from mpi4py import MPI
import numpy as np
import quop_mpi as qu
import math
import networkx as nx
import matplotlib.pyplot as plt
import time # used for naming stuff with the current date and time
import os
import sys

comm = MPI.COMM_WORLD

###############################################################
# Defining execution of program
###############################################################
run_type = "qwoa"
n_nodes = 5
p = 10

#n_paths = math.factorial(n_nodes)
n_paths = math.factorial(n_nodes-1)
n_qubits = math.ceil(math.log(n_paths, 2)) #number of qubits required to represent the n_nodes

initial_state_used = "equal"
optimisation_used = "BFGS"

use_parameter_shift_rule = True
binary_qualities = True
binary_cutoff = 3000
use_gammas_p = True
use_ts_p = False
#s_shift = int(sys.argv[1])
s_shift = math.pi / 2
d_shift = 0.01

# defin the graph being walked over
walked_graph_name = "complete"
walked_graph = np.ones(2**n_qubits)
walked_graph[0] = 0 

# For qwoa_benchmark only
pc = [5,10,15]
repeats = 1
###############################################################

# "Housekeeping" things
np.random.seed(1)
time_name = time.ctime().replace(" ", "_")




###############################################################
###############################################################
# Travelling salesman functions
###############################################################

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




                
        
def routing(N, local_i, local_i_offset, seed=1):
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

		# The 2 and -2 are required so that $U$ is of the correct form
                if not binary_qualities:
                        ret_lst[i - local_i_offset] = int(path_cost)
                else:
                        ret_lst[i - local_i_offset] = 1 if int(path_cost)<binary_cutoff else -1
                
        return ret_lst



###############################################################

###############################################################
###############################################################
# SIMULATE THE TRAVELLING SALESMAN PROBLEM WITH THE QWOA
###############################################################

# Run the qwoa or qaoa simulation   
if run_type == "qwoa":
	np.random.seed(1)
	def x0(p):
		return np.random.uniform(low = 0, high = 2*np.pi, size = 2 * p)

	qwoa = qu.MPI.qwoa(n_qubits, comm)

	qwoa.log_results("log", "default", action = "a")
	try:
		os.rename("log.csv", "./tmp/log.csv")
	except:
		print("ts.py: Cannot locate file log.csv")



	qwoa.set_graph(walked_graph)

	qwoa.set_qualities(routing)
	print("Finished setting qualities")

	qwoa.set_initial_state(name = initial_state_used)

	# Find the gradient with the parameter shift rule
	#jacobian = qwoa.find_jacobian(x0(p))
	# Choose the optimiser
	if use_parameter_shift_rule and binary_qualities:
		qwoa.set_gradient_method(use_gammas = use_gammas_p, use_ts = use_ts_p, s=s_shift, d=d_shift)

		qwoa.set_optimiser('scipy', {'method':optimisation_used, 'jac':qwoa.parameter_shift_rule})
	else:
		qwoa.set_gradient_method()
		qwoa.set_optimiser('scipy', {'method':optimisation_used})

	qwoa.plan()
	qwoa.execute(x0(p))
	qwoa.destroy_plan()

	#qwoa.save("./tmp/qwoa_routing_{}".format(time_name), "example_config", action = "w")
	qwoa.save("data", "default", action = "w")
	#os.rename("data.h5", "./tmp/data.h5")
	qwoa.print_result()


elif run_type == "qwoa_benchmark":
	 np.random.seed(1)
	 def x0(p, seed=1):
                 return np.random.uniform(low = 0, high = 2*np.pi, size = 2 * p)


	 print(run_type)
	 qwoa = qu.MPI.qwoa(n_qubits,comm)
	 qwoa.set_initial_state(name = initial_state_used)
	 qwoa.set_optimiser('scipy', {'method':optimisation_used})

	 qwoa.set_graph(walked_graph)

	 qwoa.log_results("log","default",action="a")

	 try:
		 os.rename("log.csv", "./tmp/log.csv")
	 except:
		 print("ts.py: Cannot locate file log.csv")

	 qwoa.plan()

	 qwoa.benchmark(
		 pc,
		 repeats,
		 param_func = x0,
		 qual_func = routing, 
		 filename = "qwoa_complete_equal",
		 label = "qwoa_" + str(n_qubits))
	 qwoa.destroy_plan()

	 qwoa.save("data", "default", action = "w")
	 os.rename("data.h5", "./tmp/data.h5")
	 qwoa.print_result()

elif run_type == "qaoa":
	np.random.RandomState(1)

	def x0(p):
                 return np.random.uniform(low = 0, high = 2*np.pi, size = 2 * p)

	qaoa = qu.MPI.qaoa(n_qubits, comm)

	qaoa.log_results("log", "qaoa", action = "a")


	qaoa.set_initial_state(name = initial_state_used)

	qaoa.set_qualities(routing)

	qaoa.execute(x0(p))
	qaoa.save("data", "default", action = "w")
	#os.rename("data.h5", "./tmp/data.h5")

	qaoa.print_result()


###############################################################

###############################################################
###############################################################
# WRITE INFO TO info.txt FILE
###############################################################

infile = open("./tmp/info.txt", "w")

infile.write(
        "time_name="+str(time_name)+"\n"+
        "run_type="+str(run_type)+"\n"+ 
        "p="+str(p)+"\n"+
        "n_nodes="+str(n_nodes)+"\n"+
        "n_qubits="+str(n_qubits)+"\n"+
        "initial_state_used="+str(initial_state_used)+"\n"+
        "optimisation_used="+str(optimisation_used)+"\n"+
	"use_parameter_shift_rule="+str(use_parameter_shift_rule)+"\n"+
	"binary_qualities="+str(binary_qualities)+"\n"+
	"binary_cutoff="+str(binary_cutoff)+"\n"+
	"use_gammas_p="+str(use_gammas_p)+"\n"+
	"use_ts_p="+str(use_ts_p)+"\n"+
	"s_shift="+str(s_shift)+"\n"+
	"d_shift="+str(d_shift)+"\n"+
        "pc="+str(pc)+"\n"+
        "repeats="+str(repeats)+"\n"+
        "walked_graph_name="+str(walked_graph_name)+"\n"
        )

infile.close()
