from math import sqrt, log2
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j0
from variable import *
import time
import csv
#In this program, we will use q as a np array list, shape of q will be [3*(N+1)] each 3 continous element will denote x,y, and DTS rate t

def distance2(x, y):
    return (x[0]-y[0])**2 + (x[1]-y[1])**2


def distance(x, y):
    return sqrt(distance2(x, y))
def e_c(q):
    e_fly = 0
    for i in range(0, int((len(q) - 3)/3)):
        dis = distance([q[3*i+3]*20,q[3*i+4]*10],[q[i*3]*20,q[i*3+1]*10])
        e_fly_i = P_0 * (delta_t + k_1 * dis**2) + P_1 * sqrt(sqrt(delta_t**4 + k_2**2 * dis**4) -
                                                        k_2 * dis**2) + k_3 * dis**3 / delta_t**2 + q[3*i+2] * delta_t * (P_b + P_u)
        e_fly = e_fly + e_fly_i

    return e_fly
def e_h(q):
    e_h = 0
    for i in range(0,int ((len(q)-3)/3)):
        e_h_i = miu*(1 - q[i*3 + 2])*delta_t*omega_0*P_WPT/(H**2 + distance2([q[i*3]*20,q[i*3+1]*10], w_s))**a2
        e_h = e_h + e_h_i
    return e_h
def rate(q):
    r_u = 0  # total rate from source -> uav
    r_d = 0  # total rate from uav -> destination
    for i in range(0,int ((len(q)-0)/3)):
        d_su2 = distance2([q[3*i]*20,q[3*i+1]*10], w_s)  # distance^2 from uav to source
        d_du2 = distance2([q[3*i]*20,q[3*i+1]*10], w_d)  # distance^2 from uav to destination

        # R_u at time slot i
        # formula (20)
        bar_sigma_u2 = sigma_u2 + (1-Z2)*sigma_u2
        r_u_i = log2(1 + (theta * P_s*Z2) / ((H**2 + d_su2)**a2)*bar_sigma_u2) #add noise to rate to user
        r_u = r_u + 1 * q[3*i+2] * delta_t * r_u_i

        _o = ((H**2 + d_su2)**a2 * (H**2 + d_du2)**a2)
        # formula (21)
        r_d_i1 = log2(1+(theta*(n_u*omega_0*P_s+Z2*P_u_bar*(H**2 + d_su2)**a2)) / _o) 

        # Data transmission rate from UAV to d if it cached a part of f file
        r_d_i2 = log2(1 + theta * P_u / (H**2 + d_du2)**a2)
        # R_d at time slot i
        r_d_i = r_d_i1
        # r_d_i = r_d_i1 + r_d_i2
        r_d = r_d + 1 * q[3*i+2] * delta_t * r_d_i
    return r_u+sigma*S,r_d
def fitness(q):
    e1 = e_c(q)
    e2 = e_h(q)
    delta_d = V*delta_t

    if e2 < e1:
        
        return 0
    for i in range(0,int ((len(q)-3)/3)):
        j = i + 1
        dis = distance([q[3*j]*20,q[3*j+1]*10],[q[3*i]*20,q[3*i+1]*10])
        if dis > delta_d:
            return 0
    return - rate(q)[1]


def init_q(mean_x,de_x,de_y):
    mid = int((N-1)/2)
    x = np.random.normal(loc=mean_x,scale=de_x,size=(1,N-1))
    y1 = abs(np.random.normal(loc=0,scale=de_y/2,size = mid + 1))
    y2 = abs(np.random.normal(loc=0,scale=de_y/2,size = mid + 1))
    y1 = np.sort(y1)[::-1]
    y2 = np.sort(y2)

    y = np.zeros((1,N-1))
    
    for i in range(N-1):
        if i < (N-1)/2:
            y[0][i] = y1[i]
        elif i == mid:
            y[0][i] = 0
        else:
            y[0][i] = y2[i-mid]






    x = np.sort(x)

    # Create q array
    q = np.zeros((3*N + 3,))
    q[0], q[1], q[2] = 0, 1, 0
    for i in range(N-1):
        q[3*i+3], q[3*i+4], q[3*i+5] = x[0][i], y[0][i], 0
    q[-3], q[-2], q[-1] = 1, 1, 0



    for i in range(1, N + 1):
        temp =  miu*delta_t*omega_0*P_WPT
        Xi = temp / ((H**2 + distance2([q[3*i]*20,q[3*i+1]*10], w_s))**a2)
        e_con = e_c(q[0:3*i+30])
        e_har = e_h(q[0:3*i+30])
        delta_E_need = (e_con - e_har)*1.2
        max_tau =  1 - (delta_E_need / Xi)
        threshold = min(max_tau, 1)
        if threshold > 0:
            tau = random.uniform(threshold*0.8, threshold)
        else:
            #print("Khong du nang luong tai slot thu: ", i)
            break
        q[3*i + 2] = tau
    #print('E_C',e_c(q),'E_H',e_h(q))
    #show_figure(q)
    q = np.array(q)
    q = np.clip(q, 0, 1)
    return q

def init_population(num):
    population = []
    q = []
    for i in range(0,num):
        while(1):
            mean_x = random.uniform(0.2,0.8)
            de_x = random.uniform(0.2,0.5)
            de_y = random.uniform(0.2,0.5)
            q = init_q(mean_x,de_x,de_y)
            tmp = fitness(q)
            if tmp < -S:
                population.append(q)
                break
            else:
                continue
    return population


# Define decode function to ensure correct format of individuals
def decode(individual):
    # Set index 0, 1, 2 to 1, 1, 0 respectively
    individual[0:3] = [0, 1, 0]
    # Set index -3, -2 to 1, 1 respectively
    individual[-3] = 1
    individual[-2] = 1
    # Clip all elements to be within [0,1]
    individual = np.clip(individual, 0, 1)
    return individual
def PSO(num_pop=100, num_gen=3000, c1=1.32, c2=1.32, w=0.24,v_max = 0.5):
    start_time = time.time()
    # Initialize particles with random positions and velocities
    particles =  init_population(num_pop)
    velocities = np.zeros((num_pop, 3*N+3))
    # Initialize best positions and fitness values
    best_positions = particles.copy()
    best_fitness_values = [fitness(particle) for particle in particles]
    # Initialize global best position and fitness value
    global_best_position = particles[np.argmin(best_fitness_values)].copy()
    global_best_fitness_value = min(best_fitness_values)
    # Initialize v_max
    v_max = v_max
    # Initialize mutation rate
    mutation_rate = 0.05
    # Initialize adaptive inertia weight parameters
    w_min = 0.1
    w_max = 0.9
    w = w_max
    w_damp = 0.99
    # Initialize early stop counter
    early_stop_counter = 0
    # Iterate through iterations
    for i in range(num_gen):
        # Update velocities and positions of particles
        for j in range(num_pop):
            # Update velocity
            velocities[j] = w * velocities[j] + c1 * np.random.rand() * (best_positions[j] - particles[j]) + c2 * np.random.rand() * (global_best_position - particles[j])
            # Clip velocity
            velocities[j] = np.clip(velocities[j], -v_max, v_max)
            # Update position
            particles[j] += velocities[j]
            particles[j] = decode(particles[j])

            # Evaluate fitness of new position
            fitness_value = fitness(particles[j])
            # Update best position and fitness value for particle if necessary
            if fitness_value < best_fitness_values[j]:
                best_positions[j] = particles[j].copy()
                best_fitness_values[j] = fitness_value
            # Update global best position and fitness value if necessary
            if fitness_value < global_best_fitness_value:
                global_best_position = particles[j].copy()
                global_best_fitness_value = fitness_value
                # Reset early stop counter if new global best is found
                early_stop_counter = 0
            else:
                # Increase early stop counter if global best is not updated
                early_stop_counter += 1
        # Decrease v_max and increase w when iteration increases
        if i % 10 == 0:
            v_max *= 0.9
            w *= w_damp
            w = max(w_min, w)
        # Mutate particles
        for j in range(num_pop):
            if np.random.rand() < mutation_rate:
                # Gaussian mutation
                particles[j] += np.random.normal(0, 0.1, size=3*N+3)
                particles[j] = decode(particles[j])

        # Write result to csv per 10 inter
        if (i+1) % 10 == 0 and (i+1) > 0:
            algorithm_name = 'PSO'
            current_best_fitness = global_best_fitness_value
            # Save values to csv file
            with open(f'result_inter.csv', mode='a',newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i+1, T, P_WPT, S, P_u, eta, algorithm_name, current_best_fitness])
        print(f"Iteration {i+1}: Best fitness value = {global_best_fitness_value}")
        # Early stop if global best is not updated after 25 iterations
        if early_stop_counter >= 25:
            print("Early stop at iteration", i+1)
            break
    end_time = time.time()
    # Return global best position
    return global_best_position,fitness(global_best_position), end_time - start_time





# To visualize x,y,z of the best_individual, we can extract the values of x, y, and z from the array and plot them using matplotlib library. Here is the code to do so:

import matplotlib.pyplot as plt
def visualize(best_individual):
    # Extract x, y, and z values from best_individual array
    x = [best_individual[3*i] for i in range(int(len(best_individual)/3))]
    y = [best_individual[3*i+1] for i in range(int(len(best_individual)/3))]
    t = [best_individual[3*i+2] for i in range(int(len(best_individual)/3))]

    value = fitness(best_individual)



    # Plot the points as a scatter plot
    plt.scatter(x, y, c=t, cmap='viridis')

    # Add a line connecting the points
    for i in range(len(x)-1):
        plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color='black')

    # Add a colorbar to explain the color of the z variable
    plt.colorbar(label='t')

    # Add labels to the x and y axes
    plt.xlabel('x')
    plt.ylabel('y')

    # Add fitness value to chart
    plt.title(f'Fitness Value: {value}')

    # Show the plot
    plt.show()
def polynomial_mutation(individual, eta, low, up, indpb):
    """
    This function applies polynomial mutation to the input individual.
    :param individual: The individual to apply mutation to
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: The minimum value allowed for the individual
    :param up: The maximum value allowed for the individual
    :param indpb: Independent probability for each attribute to be mutated
    :return: A tuple of one individual
    """
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            x = individual[i]
            x = min(max(x + random.gauss(0, 1) * eta, low), up)
            individual[i] = x
    return individual


# Here is an implementation of a genetic algorithm with SBX crossover, adaptive Gaussian mutation, and parent selection that chooses 30% top, 30% bottom, and 40% mid individuals:

def GA(num_pop=100, num_gen=3000, crossover_rate=0.9, mutation_rate=0.1):
    start_time = time.time()
    # Initialize population
    population = init_population(num_pop)
    # Initialize best fitness value
    best_fitness = float('inf')
    # Initialize counter for number of generations with no improvement
    no_improvement_count = 0
    # Iterate through generations
    for i in range(num_gen):
        # Evaluate fitness of population
        fitness_values = [fitness(individual) for individual in population]
        # Return individual with best fitness
        best_individual = population[np.argmin(fitness_values)]
        best = fitness(best_individual)
        print(f"Iteration {i+1}: Best fitness value = {best}")
        # Write result to csv per 10 inter
        if (i+1) % 10 == 0 and (i+1) > 0:
            algorithm_name = 'GA'
            current_best_fitness = best
            # Save values to csv file
            with open(f'result_inter.csv', mode='a',newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i+1, T, P_WPT, S, P_u, eta, algorithm_name, current_best_fitness])
        # Check if best fitness has improved
        if best < best_fitness:
            best_fitness = best
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        # Check if early stop condition is met
        if no_improvement_count == 40:
            print("Early stop at iteration", i+1)
            break
        # Select parents
        sorted_population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0])]
        num_top = int(num_pop * 0.3)
        num_bot = int(num_pop * 0.3)
        num_mid = num_pop - num_top - num_bot
        top_individuals = sorted_population[:num_top]
        bot_individuals = sorted_population[-num_bot:]
        mid_individuals = sorted_population[num_top:-num_bot]
        parents = []
        for j in range(num_pop):
            if j < num_top:
                parents.append(random.choice(top_individuals))
            elif j < num_top + num_mid:
                parents.append(random.choice(mid_individuals))
            else:
                parents.append(random.choice(bot_individuals))
        # Create offspring using SBX crossover
        offspring = []
        for j in range(num_pop):
            parent1, parent2 = random.sample(parents, 2)
            child = np.zeros_like(parent1)
            for k in range(len(child)):
                if np.random.rand() < 0.5:
                    if parent1[k] < parent2[k]:
                        child[k] = parent1[k] + (parent2[k] - parent1[k]) * np.random.rand()
                        
                    else:
                        child[k] = parent2[k] + (parent1[k] - parent2[k]) * np.random.rand()
                        
                else:
                    child[k] = parent1[k]
                    
            child = decode(child)
            offspring.append(child)
        # Apply adaptive Gaussian mutation
        for j in range(num_pop):
            eta = 1 / np.sqrt(2 * np.sqrt(num_pop))
            low = 0
            up = 1
            indpb = 1 / (3 * N + 3)
            offspring[j] = polynomial_mutation(offspring[j], eta, low, up, indpb)
            if np.random.rand() < mutation_rate:
                # Gaussian mutation
                offspring[j] += np.random.normal(0, 0.1, size=3*N+3)
                offspring[j] = decode(offspring[j])
        # Normalize offspring
        offspring = np.clip(offspring, 0, 1)
        # Evaluate fitness of offspring
        offspring_fitness_values = [fitness(individual) for individual in offspring]
        # Replace population with offspring if offspring has better fitness
        for j in range(num_pop):
            if offspring_fitness_values[j] < fitness_values[j]:
                population[j] = offspring[j]
    # Decode all elements in population
    population = [decode(child) for child in population]


    # Evaluate fitness of final population
    fitness_values = [fitness(individual) for individual in population]
    # Return individual with best fitness
    best_individual = population[np.argmin(fitness_values)]
    best_individual = decode(best_individual)
    end_time = time.time()
    return best_individual, fitness(best_individual), end_time - start_time


def DE(num_pop=100, num_gen=3000, F=0.5, CR=0.7):
    start_time = time.time()
    # Initialize population
    population = init_population(num_pop)
    # Iterate through generations
    for i in range(num_gen):
        # Evaluate fitness of population
        fitness_values = [fitness(individual) for individual in population]
        # Return individual with best fitness
        best_individual = population[np.argmin(fitness_values)]
        best = fitness(best_individual)
        print(f"Iteration {i+1}: Best fitness value = {best}")
                # Write result to csv per 10 inter
        if (i+1) % 10 == 0 and (i+1) > 0:
            algorithm_name = 'DE'
            current_best_fitness = best
            # Save values to csv file
            with open(f'result_inter.csv', mode='a',newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i+1, T, P_WPT, S, P_u, eta, algorithm_name, current_best_fitness])
        # Create offspring using differential evolution
        offspring = []
        for j in range(num_pop):
            # Select three random individuals
            a, b, c = random.sample(population, 3)
            # Create mutant vector
            mutant = a + F * (b - c)
            # Create trial vector
            trial = np.zeros_like(mutant)
            for k in range(len(trial)):
                if np.random.rand() < CR:
                    trial[k] = mutant[k]
                else:
                    trial[k] = population[j][k]
            # Add trial vector to offspring
            offspring.append(decode(trial))
        # Normalize offspring
        offspring = np.clip(offspring, 0, 1)
        # Evaluate fitness of offspring
        offspring_fitness_values = [fitness(individual) for individual in offspring]
        # Replace population with offspring if offspring has better fitness
        for j in range(num_pop):
            if offspring_fitness_values[j] < fitness_values[j]:
                population[j] = offspring[j]
    # Evaluate fitness of final population
    fitness_values = [fitness(individual) for individual in population]
    # Return individual with best fitness
    best_individual = population[np.argmin(fitness_values)]
    end_time = time.time()
    return best_individual,fitness(best_individual),end_time - start_time





def run_all():
    algorithms_list = [PSO() for i in range(3)]
    algorithms_name = ['PSO' for i in range(3)]
    for j in range(len(algorithms_list)):
        best_position,best_fitness,time = algorithms_list[j]
        algorithm_name =  algorithms_name[j]# get algorithm name
        with open('results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([j+1,T,algorithm_name, time, best_fitness])
            file.flush() # flush the buffer to ensure the data is written to file



     
def grid_search_PSO():
    v_max = np.linspace(0.1, 1, 10)
    c1_c2 = np.linspace(0.5, 2, 10)
    best_fitness = float('inf')
    best_v_max = None
    best_c1_c2 = None
    for v in v_max:
        for c in c1_c2:
            pso = PSO(v_max=v, c1=c, c2=c)
            _, fitness, _ = pso
            if fitness < best_fitness:
                best_fitness = fitness
                best_v_max = v
                best_c1_c2 = c
    print(f"Best v_max: {best_v_max}")
    print(f"Best c1_c2: {best_c1_c2}")
    print(f"Best fitness: {best_fitness}")
 
def grid_search_GA(num_pop=[50, 100, 150, 200], num_gen=[3000, 4000, 5000], mutation_rate=[0.01, 0.05, 0.1, 0.15], crossover_rate=[0.5, 0.7, 0.9]):
    best_fitness = float('inf')
    best_num_pop = None
    best_num_gen = None
    best_mutation_rate = None
    best_crossover_rate = None
    for pop in num_pop:
        for gen in num_gen:
            for rate in mutation_rate:
                for cross in crossover_rate:
                    ga = GA(num_pop=pop, num_gen=gen, mutation_rate=rate, crossover_rate=cross)
                    _, fitness, _ = ga
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_num_pop = pop
                        best_num_gen = gen
                        best_mutation_rate = rate
                        best_crossover_rate = cross
    print(f"Best num_pop: {best_num_pop}")
    print(f"Best num_gen: {best_num_gen}")
    print(f"Best mutation_rate: {best_mutation_rate}")
    print(f"Best crossover_rate: {best_crossover_rate}")
    print(f"Best fitness: {best_fitness}")



