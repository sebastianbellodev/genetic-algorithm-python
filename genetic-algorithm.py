import random
import math
import matplotlib.pyplot as plt

## Problem definition
## An array of 10 numbers represents an individual
## Each element of the array is a number between -5.12 and 5.12
## The population will be of 1000 individuals
## There will be 1000 generations
## The probability of mutation will be 0.4 

POPULATION_SIZE = 2000
GENERATION_SIZE = 5000
ALPHA = 1
MUTATION_RATE = 0.4
PARENTS_SELECTED = 2000
EXECUTIONS = 1

## Create an individual, which is an array of 10 random numbers between -5.12 and 5.12
def generate_individual():
    _individual = []
    for _ in range(10):
        _number = round(random.uniform(-5.120, 5.120), 3)
        _individual.append(_number)
    return _individual

## Create a population of 1000 individuals, each individual is an array of 10 numbers
def generate_population():
    _population = []
    for i in range(POPULATION_SIZE):
        _individual = generate_individual()
        _fitness = fitness(_individual)
        _population.append((_individual, _fitness))
    return _population

## Fitness function of an individual
def fitness(individual):
    _sum = 0
    for i in range(0,9):
        _sum += math.pow(individual[i], 2) - (10 * math.cos(2 * math.pi * individual[i]))
    _result = 10 * len(individual) + _sum
    return round(_result,3)

## Variation using BLX-Alpha crossover
## Return two offspring
def blx_a_crossover(parentA, parentB):
    _cmax = 0
    _cmin = 0
    _alpha = ALPHA
    _interval = 0
    _childx = []
    _childy = []
    for i in range(0,9):
        _cmax = max(parentA[i], parentB[i])
        _cmin = min(parentA[i], parentB[i])
        _interval = abs(_cmax) - abs(_cmin)
        _childx.append(round(random.uniform(_cmin - _alpha * _interval, _cmax + _alpha * _interval),3))
        _childy.append(round(random.uniform(_cmin - _alpha * _interval, _cmax + _alpha * _interval),3))
    _childx = uniform_mutation(_childx)
    _childy = uniform_mutation(_childy)
    _childx = check_interval(_childx)
    _childy = check_interval(_childy)
    return _childx, _childy

## Mutation is uniform, each number of the individual has a probability of 0.4 of being mutated
def uniform_mutation(child):
    for i in range(0,9):
        _rand = random.uniform(0,1)
        if _rand < MUTATION_RATE:
            #_random_gaussian = round(random.gauss(child[i], .5),3)
            _random = round(random.uniform(-1.120, 1.120),3)
            child[i] = child[i] + _random
    return child

## Verify that the values of the individual are within the allowed interval [-5.120, 5.120]
def check_interval(child):
    for i in range(0,9):
        if child[i] < -5.120:
            child[i] = -5.120
        if child[i] > 5.120:
            child[i] = 5.120
        child[i] = round(child[i],3)
    return child

## Selection of parents by roulette
## Return the selected parents
def roulette_selection(population):
    _parents = []
    _wheel_position = 0
    _population_probabilities = computation_probability(population)
    for _ in range(PARENTS_SELECTED):
        _rand = random.uniform(0,1)
        _wheel_weight = 0
        ## Set the position to move the wheel
        _wheel_position += _wheel_position + _rand
        _wheel_position = _wheel_position if _wheel_position <= 1 else (_wheel_position - 1)      
        for i, _individual_probabilities in enumerate(_population_probabilities):
            _wheel_weight += _individual_probabilities
            if _wheel_position >= _wheel_weight:
                _parents.append(population[i])
                break
            else:
                continue
    return _parents

## Aux for parent selection
def computation_probability(population):
    _probabilities = []
    for individual in population:
        _aptitude_value = individual[1]
        _probability = 1 / _aptitude_value
        _probabilities.append(_probability)
    _sum_prob = sum(_probabilities)
    _normalization_prob = [_prob / _sum_prob for _prob in _probabilities]
    return _normalization_prob


# def roulette_selection(population):
#     _parents = []
#     _population_probabilities = computation_probability(population)
#     for i in range(0,PARENTS_SELECTED+1):
#         _rand = random.uniform(0, 1)
#         _wheel_weight = 0
#         _wheel_position = 0
#         _wheel_position += _rand
#         for i, _individual_probabilities in enumerate(_population_probabilities):
#             _wheel_weight += _individual_probabilities
#             if _wheel_position >= _wheel_weight:
#                 _parents.append(population[i])
#                 break
#             else:
#                 continue
#     return _parents

## Define parent reproduction
## Return the children
## Cross the parents, mutate and check the interval, also calculate the fitness of the children
def reproduction(parents):
    _childrens = []
    for i in range(0, len(parents)-1,2):
        _parentA = parents[i][0]
        _parentB = parents[i+1][0]
        _childA, _childB = blx_a_crossover(_parentA, _parentB)
        _childrens.append((_childA, fitness(_childA)))
        _childrens.append((_childB, fitness(_childB)))
    return _childrens

## Set the elitism strategy. Only the best individuals are kept
def replace_population(population, childrens):
    _population_index_sorted = sorted(range(len(population)), key=lambda x: population[x][1])
    
    _population_sorted = sorted(population, key=lambda x: x[1])
    _new_population = []
    _new_population.append(_population_sorted[0])
    for i in range(1, len(childrens)):
        _new_population.append(childrens[i])
    # _new_population = []
    # _new_population = population.append(population[0])
    # ## Combine the current population and the generated children
    # _population = population + children
    # ## Calculate the number of individuals to keep based on the original population size
    # num_to_keep = len(population)
    # ## Find the indices of the worst individuals (those with the highest fitness value)
    # worst_indices = sorted(range(len(_population)), key=lambda i: _population[i][1])[-num_to_keep:]
    # ## Create a new population by removing the worst individuals
    # new_population = [ind for i, ind in enumerate(_population) if i not in worst_indices]
    return _new_population

# def genetic_algorithm():
#     _population = generate_population()
#     for _ in range(GENERATION_SIZE):
#         _parents = roulette_selection(_population)
#         _children = reproduction(_parents)
#         _population = replace_population(_population, _children)
#     return _population

# print(genetic_algorithm())

## Set genetic algorithm
## Return the final population, the best results of each generation and the worst results of each generation
def genetic_algorithm():
    _population = generate_population()
    best_results = []
    worst_results = []
    for i in range(GENERATION_SIZE):
        _parents = roulette_selection(_population)
        print("Parents: ", len(_parents))
        _children = reproduction(_parents)
        print("Children: ", len(_children))
        _population = replace_population(_population, _children)
        print("Population: ", len(_population))
        ## Save the best and worst result of the generation
        best_result = min(_population, key=lambda x: x[1])
        worst_result = max(_population, key=lambda x: x[1])
        print("Generation {}: Best result: {}, Worst result: {}".format(i+1, best_result, worst_result))
        best_results.append(best_result)
        worst_results.append(worst_result)
    ## plot_results(best_results, worst_results)
    return _population, best_results, worst_results

def plot_results(best_results, worst_results):
    generations = range(1, GENERATION_SIZE + 1)
    best_fitness = [result[1] for result in best_results]
    worst_fitness = [result[1] for result in worst_results]
    
    plt.plot(generations, best_fitness, label='Best fitness of the generation')
    plt.plot(generations, worst_fitness, label='Worst fitness of the generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evaluation of individuals in each generation')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
population, best_results, worst_results = genetic_algorithm() 
plot_results(best_results, worst_results)  

# def plot_results_final(best_results_list, worst_results_list):
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.title("30 Best Results")
#     generations = range(1, GENERATION_SIZE + 1)
#     for i, best_results in enumerate(best_results_list):
#         generations = range(1, len(best_results) + 1)
#         best_fitness = [result[1] for result in best_results]
#         plt.plot(generations, best_fitness, label=f'Iteration {i+1}')
#     plt.xlabel('Generation')
#     plt.ylabel('Fitness')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.title("30 Worst Results")
#     for i, worst_results in enumerate(worst_results_list):
#         generations = range(1, len(worst_results) + 1)
#         worst_fitness = [result[1] for result in worst_results]
#         plt.plot(generations, worst_fitness, label=f'Iteratioon {i+1}')
#     plt.xlabel('Generation')
#     plt.ylabel('Fitness')
#     plt.legend()

#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# ## Run the genetic algorithm
# best = []
# worst = []
# for i in range(EXECUTIONS):
#     final_population, best_results, worst_results = genetic_algorithm()
#     best.append(best_results)
#     worst.append(worst_results)

# plot_results_final(best, worst)