"""
@author: OUYA Boris
@description: This project intends to simulate random walks (1D and 2D)
We will define 
- E = set of integers :: the state space
- (X(n)) with n , a natural number and X0 = 0 , a serie of random 
(Xn) with n, a natural number and X0 = 0 , simulating a 1D random walk. 
The tasks at hand are:
 - program random walks
 - give a graphical representation of a simulated walk if needed 
 - calculate probabiliy of being in position k after n steps in the case of a 1D random Walk

Bonus: For my own amusement, I decided to make the program console-friendly with multiple parameters so 
that we never have to tweak the program to have the desired outputs

There is also the possibility to save the intermediary results (in case of exact calculation) in a text file. This could lead to some improvements:
- saving instead a pickle object so that we can easily retrieve and use it in further calculations
- using previously saved memory to speed up subsequent calculations memory (if at some point, we calculated P(Xn = k) and saved all the 
calculations in memory file, we could retrieve the memory file and use it in other calculations)

But, I will stop here for now, as this will overcomplicate things (and I already did).




Syntax:
    python3 main.py n k [l]
    python3 main.py -s n
    python3 main.py -{opts}*p{opts}* n k l
    python3 main.py -{opts}*e{opts}*[m filename] n k
Opts:
    -t: time
    -s: simulate
    -e: exact
    -p: approximate 
    -m: memory (effective  if exact activated)
    Default filename: memory.txt
    By default, if nor exact nor simulate are activated and l is not specified, approximate is activated and l=1000\n\n\
    -n: number of steps
    -k : final position
    -l : number of simulations"
"""



import numpy as np
import matplotlib.pyplot as plt
import time

import sys


syntax = "Syntax:\n\
    python3 main.py n k [l]\n\
    python3 main.py -s n\n\
    python3 main.py -{opts}*p{opts}* n k l\n\
    python3 main.py -{opts}*e{opts}*[m filename] n k\nOpts:\n\
    -t: time\n\
    -s: simulate\n\
    -e: exact\n\
    -p: approximate \n\
    -m: memory (effective  if exact activated\n\
    Default filename: memory.txt\n\
    By default, if nor exact nor simulate are activated and l is not specified, approximate is activated and l=1000\n\n\
    -n: number of steps\n\
    -k : final position\n\
    -l : number of simulations"





class SyntaxicError(Exception):
    pass
def take_one_step(seeded=False,seed=0):
    """
        Return 1 or -1 with equiprobability

        if seeded is set to True, the seed argument is used as the seed of the 
        pseudo-random number generator
    """

    if seeded:
        np.random.seed(seed)
    return 1 if np.random.random() > 0.5 else -1

def take_n_steps_1D(n,seeded=False,seed=0,show=False):
    """ 
        Prints the final position after n steps of a 1D-random walk as well as the 
        average number of steps before revisiting the origin
        If the origin is never visited, prints infinity instead

        if show is set to True, the function prints out a plot with 
        the subsquent positions of the agent 

        if seeded is set to True, the seed argument is used as the seed of the 
        pseudo-random number generator
    """

    X = 0

    number_of_steps_since_last_visit_of_origin = 0
    record_of_number_of_steps_before_revisiting_the_origin = []

    if show:
        positions = [0]
        m = 0

    for _ in range(n):
        number_of_steps_since_last_visit_of_origin += 1
        X += take_one_step(seeded=seeded,seed=seed)

        if X == 0:
            record_of_number_of_steps_before_revisiting_the_origin.append(number_of_steps_since_last_visit_of_origin)
            number_of_steps_since_last_visit_of_origin = 0

        if show:
            if abs(m) < abs(X):
                m = abs(X)
            positions.append(X)

    avg_time_before_revisiting_the_origin = sum(record_of_number_of_steps_before_revisiting_the_origin) / \
                                                    len(record_of_number_of_steps_before_revisiting_the_origin)
    
    print(f"Final position after {n} steps: {X}")
    print(f"Average number of steps before revisiting the origin {avg_time_before_revisiting_the_origin} steps.")
    if show:
        plt.plot(np.arange(n+1),positions,marker=".")
        plt.title(f"Random walk of {n} steps")
        plt.xlabel("Steps")
        plt.ylabel("Position from origin")
        plt.ylim(-m-1,m+1)
        plt.show()

    
        

def take_n_steps_2D(n,seeded=False,seed=0,show=False):
    """
        Prints the final position after n steps of a 1D-random walk as well as the 
        average number of steps before revisiting the origin
        If the origin is never visited, prints infinity instead

        if show is set to True, the function prints out a plot with 
        the subsquent positions of the agent 

        if seeded is set to True, the seed argument is used as the seed of the 
        pseudo-random number generator
    """

    X = [0,0]

    if show:
        positions = [[0,0]]
        m = 0

    for _ in range(n):
        if (np.random.random()>0.5):
            X[0] += take_one_step(seeded=seeded,seed=seed)
        else:
            X[1] += take_one_step(seeded=seeded,seed=seed)
        
        if show:
            maximum = max(abs(X[0]),abs(X[1]))
            if abs(m) < maximum:
                m = maximum
            positions.append(X[:])


    
    if show:
        plt.scatter([0],[0],marker="x")
        plt.plot([v[0] for v in positions],[v[1] for v in positions],marker=".")
        plt.title(f"Random walk of {n} steps")
        plt.xlabel("Steps")
        plt.ylabel("Position from origin")
        plt.ylim(-m-1,m+1)
        plt.show()

    print(f"Final position after {n} steps: {X}")

def estimate_probability(n,k,m=100):
    def simulate_n_steps():
        X = 0
        for _ in range(n):
            X += take_one_step()

        return X
    
    number_of_times_the_goal_was_reached = 0

    for _ in range(m):
        if simulate_n_steps() == k:
            number_of_times_the_goal_was_reached += 1

    return number_of_times_the_goal_was_reached / m


def calculate_probability_by_recurrence(n,k):
    """
        Recursively computes the probability P(Xn = k).
    """

    if n < abs(k):
        return 0
    if (n == 1)  and abs(k) == 1:
        return 1/2
    if (n == 1) and abs(k) > 1:
        return 0
    if (n==0) and (k == 0):
        return 1
    
    return 1/2 * calculate_probability_by_recurrence(n-1,k-1) + 1/2 * calculate_probability_by_recurrence(n-1,k+1)



def calculate_probability_dfs(n,k):
    """
        Computes P(Xn = k)

        Calculating the probabilities for this problem is analoguous to 
    visiting nodes of a tree with the Depth First Search Rule.
    We visit only the relevant possibilities and memorize the calculated probabilites.

   
    """
    def is_unreachable(n,k):
        return n < abs(k)  or (n % 2 != k % 2)
    
    def is_terminal(n,k):
        return is_unreachable(n,k) or ((n==1) and abs(k) == 1) or (n==0 and k == 0)

    
    Pile = [(n,k)]

    n0 = n
    k0 = k
    memory = {(0,0):1,(1,-1):1/2,(1,1):1/2}

    while len(Pile) > 0:
        n,k = Pile[-1]

        if is_terminal(n,k):
            Pile.pop()
            if is_unreachable(n,k): 
                memory[(n,k)] = 0
        else:
            try: #trying to access P(Xn=k)
                memory[(n,k)] 
            except KeyError:  # P(Xn=k) was not memorize : we have to calculate it
                try: #trying to access the P(X{n-1} = k-1)
                    prob1 = memory[(n-1,k-1)] 
                except KeyError: # if this probability was not memorized, 
                                # we tried to access P(Xn=k) for the first time.
                                # we then have to calculate the relevant probabilities
                    Pile.append((n-1,k+1))
                    Pile.append((n-1,k-1))
                else: 
                    try: # we try to calculate P(X{n-1} = k+1)
                        prob2 = memory[(n-1,k+1)]
                    except KeyError: # if it was not calculated, we add the task to the stack
                        Pile.append((n-1,k+1))
                    else: # we can calculate P(Xn = k) and  memorize it
                        memory[(n,k)] = 0.5*(prob1 + prob2)
                        Pile.pop()
            else:
                prob1 = memory[(n-1,k-1)]
                prob2 = memory[(n-1,k+1)]
                memory[(n,k)] = 0.5*(prob1 + prob2)
                Pile.pop()
                

    return memory[(n0,k0)],memory   


if __name__ == "__main__":

    """
    OPTIONS:
    -t : time 
    -s : simulate
    -e : exact calculation
    -p : approximation by averaging over multiple simulations
    -m : save memory in the file indicated
    """
    
    timeEnable = False
    simulateEnable = False
    exactEnable = False
    printEnable = False
    memoryEnable = False
    approximateEnable = False

    syntaxicError = False

    l = 1000
    # exemple python3 main.py -st 1000 2 
                              #op  #n   #k

    try:
        if len(sys.argv) >= 2:
            if sys.argv[1][0] == "-": #options enabled
                opts = sys.argv[1][1:]
                if "t" in opts:
                    timeEnable = True
                if "s" in opts:
                    simulateEnable = True
                if "e" in opts:
                    exactEnable = True
                    if "m" == opts[-1]:
                        memoryEnable = True
                if "p" in opts:
                    approximateEnable = True
                if "s" in opts:
                    simulateEnable = True

                if len(sys.argv) == 3 and simulateEnable:
                    try:
                        n = int(sys.argv[2])
                    except ValueError:
                        raise SyntaxicError()
                
                elif len(sys.argv) >= 3:
                    if memoryEnable:
                        if len(sys.argv) < 5:
                            raise SyntaxicError()
                        else:
                            memoryFile = sys.argv[2]
                            if approximateEnable:
                                if len(sys.argv) < 6:
                                    raise SyntaxicError
                                else:
                                    if len(sys.argv) == 6:
                                        try:
                                            n = int(sys.argv[3])
                                            k = int(sys.argv[4])
                                            l = int(sys.argv[5])
                                        except ValueError | IndexError:
                                            raise SyntaxicError()
                                    else:
                                        raise SyntaxicError()
                            else:
                                if len(sys.argv) == 5:
                                    try:
                                        n = int(sys.argv[3])
                                        k = int(sys.argv[4])
                                    except ValueError | IndexError:
                                        raise SyntaxicError()
                                else:
                                    raise SyntaxicError()

                    else:
                        if len(sys.argv) < 5:
                            try:
                                n = int(sys.argv[2])
                                k = int(sys.argv[3])
                            except ValueError or IndexError:
                                raise SyntaxicError()
                             
                        else :
                            if len(sys.argv) == 5:
                                try:
                                    n = int(sys.argv[2])
                                    k = int(sys.argv[3])
                                    l = int(sys.argv[4])
                                except ValueError:
                                    raise SyntaxicError()
                            else:
                                raise SyntaxicError()
            else:
                if len(sys.argv) >= 3: # only numerical parameters
                    try:
                        n = int(sys.argv[1])
                        k = int(sys.argv[2])
                    except ValueError:
                        raise SyntaxicError()
                    if len(sys.argv) == 4:
                        try:
                            l = int(sys.argv[3])
                        except ValueError:
                            raise SyntaxicError()
                    if len(sys.argv) > 4:
                        raise SyntaxicError()
                else:
                    raise SyntaxicError()
        else:
            raise SyntaxicError()

    except SyntaxicError:
        print("Bad arguments!")  
        print(syntax)
        exit(1)       
   
   
            
    if not exactEnable and not approximateEnable and not simulateEnable:
        approximateEnable = True
        

    if timeEnable:

        if approximateEnable:
            print("==================Approximation==================")
            deb = time.perf_counter_ns()
            probability = estimate_probability(n,k,l)
            fin = time.perf_counter_ns()

            print(f"Estimated probability of being at positon {k} after {n} steps ({l} simulations) : {probability}")
            print(f"Simulation time: {(fin - deb)/1000000000:.5f}")
        if exactEnable:
            print("==================Exact calculation==================")
            deb = deb = time.perf_counter_ns()
            probability,memory = calculate_probability_dfs(n,k)
            fin = time.perf_counter_ns()
            print(f"Exact probability of being at positon {k} after {n} steps : {probability}")
            print(f"Calculation time: {(fin - deb)/1000000000:.5f}")

            
    else:
        if approximateEnable:
            print("==================Approximation==================")
            probability = estimate_probability(n,k,l)
            print(f"Estimated probability of being at positon {k} after {n} steps ({l} simulations) : {probability}")
        if exactEnable:
            print("==================Exact calculation==================")
            probability,memory = calculate_probability_dfs(n,k)
            print(f"Exact probability of being at positon {k} after {n} steps : {probability}")


    if memoryEnable:
        memory = dict(sorted(memory.items(),key=lambda item:item[0]))  # tri par ordre lexicographique
        with open(memoryFile,"w") as f:
            for key,value in memory.items():
                f.write(f"{key}:{value}\n")
        print(f"Memory saved in {memoryFile}")
        print(f"{len(memory)} lines saved")

    if simulateEnable:
        print("==================Simulation==================")
        take_n_steps_1D(n,show=True)

