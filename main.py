import numpy as np
import matplotlib.pyplot as plt
import time

def take_one_step(seeded=False,seed=0):
    """
        Return 1 or -1 with equiprobability

        if seeded is set to True, the seed argument is used as the seed of the 
        pseudo-random number generator
    """

    if seeded:
        np.random.seed(seed)
    return 1 if np.random.random() > 0.5 else -1

def take_n_steps(n,seeded=False,seed=0,show=False):
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
    if show:
        plt.scatter(np.arange(n+1),positions,marker=".")
        plt.title(f"Random walk of {n} steps")
        plt.xlabel("Steps")
        plt.ylabel("Position from origin")
        plt.ylim(-m-1,m+1)
        plt.show()

    print(f"Final position after {n} steps: {X}")
    print(f"Average time before a revisit of the origin {avg_time_before_revisiting_the_origin}")
        


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

    if n < abs(k):
        return 0
    if (n == 1)  and abs(k) == 1:
        return 1/2
    if (n == 1) and abs(k) > 1:
        return 0
    
    return 1/2 * calculate_probability_by_recurrence(n-1,k-1) + 1/2 * calculate_probability_by_recurrence(n-1,k+1)


def calculate_probability_iterative(n,k):    

    def is_unreachable(n,k):
        return n < abs(k) or (n==1 and abs(k) > 1)
    
    def is_terminal(n,k):
        return is_unreachable(n,k) or ((n==1) and abs(k) == 1)
    
    stack = []

    prob = 0.0
    p = 0
    stop = False
    while not stop:
        if is_terminal(n,k):
            if (n==1) and abs(k) == 1: #reachable
                prob += pow(1/2,p+1)
            if len(stack) == 0:
                stop = True
            else:
                n,k,p = stack.pop()
                n -= 1
                k += 1
                p += 1
        else:
            stack.append((n,k,p))
            n -= 1
            k -= 1
            p += 1

    return prob
                





if __name__ == "__main__":
    print(estimate_probability(20,2,10000))


    """deb = time.time()
    print(calculate_probability_by_recurrence(100,2))
    fin = time.time()

    print(f"Temps de calcul: {fin - deb}")
"""
    deb = time.perf_counter_ns()
    print(calculate_probability_iterative(20,2))
    fin = time.perf_counter_ns()
    print(f"Temps de calcul: {(fin - deb)/1000000000:.5f}")