
import random
import math
import matplotlib.pyplot as plt


##Caracteristicas
##Individuo es un arreglo de tamaño 10
##La población será de 1000 individuos
##La cantidad de generaciones será de 1000
##La probabilidad de mutación será de 0.4
##Los individuos son arreglos de 10 numeros los cuales deben ser −5.12 ≤ x ≤ 5.12

POPULATION_SIZE = 1000
GENERATION_SIZE = 10000
ALPHA = 1
MUTATION_RATE = 0.4
PARENTS_SELECTED = 200

##Crea un individuo , el cual es un areglo de 10 numeros aleatorios entre -5.12 y 5.12
def generate_individual():
    _individual = []
    for _ in range(10):
        _number = round(random.uniform(-5.120, 5.120),3)
        _individual.append(_number)
    return _individual
##Crea una población de 1000 individuos que son arreglos
def generate_population():
    _population = []
    for i in range(POPULATION_SIZE):
        _individual = generate_individual()
        _fitness = fitness(_individual)
        _population.append((_individual, _fitness))
    return _population
##Función de aptitud de un individuo, solo regresa el valor de la función
def fitness(individual):
    _result = 0
    for i in range(0,9):
        _sum = 0
        for i in range(0,9):
            _sum += math.pow(individual[i], 2) - (10 * math.cos(2 * math.pi * individual[i]))
        _result += 100 + _sum
    return round(_result,3)

##Cruza de dos individuos, regresa dos hijos
def cruza_BLX_A(parentA, parentB):
    _cmax = 0
    _cmin = 0
    _alpha = ALPHA
    _interval = 0
    _childx = []
    _childy = []
    for i in range(0,9):
        _cmax = max(parentA[i], parentB[i])
        _cmin = min(parentA[i], parentB[i])
        _interval = _cmax - _cmin
        _childx.append(round(random.uniform(_cmin - _alpha * _interval, _cmax + _alpha * _interval),3))
        _childy.append(round(random.uniform(_cmin - _alpha * _interval, _cmax + _alpha * _interval),3))
    _childx = check_interval(_childx)
    _childy = check_interval(_childy)
    _childx = uniform_mutation(_childx)
    _childy = uniform_mutation(_childy)
    return _childx, _childy

##Mutación de un individuo, regresa el individuo mutado
def uniform_mutation(child):
    for i in range(0,9):
        _rand = random.uniform(0,1)
        if _rand < MUTATION_RATE:
            child[i] = round(random.uniform(-5.120, 5.120),3)
    return child

##Checa que los valores del individuo esten en el intervalo permitido [-5.120, 5.120]
def check_interval(child):
    for i in range(0,9):
        if child[i] < -5.120:
            child[i] = -5.120
        if child[i] > 5.120:
            child[i] = 5.120
    return child
##Selección de padres por ruleta, regresa los padres seleccionados
def roulette_selection(population):
    _parents = []
    _wheel_position = 0
    _population_probabilities = computation_probability(population)
    for _ in range(PARENTS_SELECTED):
        _rand = random.uniform(0,1)
        _wheel_weight = 0
        ##where to move the wheel
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

##Auxiliar para la selección de padres
def computation_probability(population):
    _probabilities = []
    for individual in population:
        _aptitude_value = individual[1]
        _probabilitie = 1 / _aptitude_value
        _probabilities.append(_probabilitie)
    _sum_prob = sum(_probabilities)
    _normalization_prob = [_prob / _sum_prob for _prob in _probabilities]
    return _normalization_prob
##Reproducción de los padres, regresa los hijos, se hace la cruza de los padres, mutación y se checa el intervalo, ademas se calcula el fitness de los hijos
def reproduction(parents):
    _children = []
    for i in range(0, len(parents)-1,2):
        _parentA = parents[i][0]
        _parentB = parents[i+1][0]
        _childA, _childB = cruza_BLX_A(_parentA, _parentB)
        _children.append((_childA, fitness(_childA)))
        _children.append((_childB, fitness(_childB)))
    return _children
##Reemplazo de la población, se juntan los hijos con la población actual, se eliminan los peores individuos y se regresa la nueva población
def replace_population(population, children):
    # Combine la población actual y los hijos generados
    _population = population + children
    # Calcule la cantidad de individuos a mantener en función del tamaño original de la población
    num_to_keep = len(population)
    # Encuentre los índices de los peores individuos (los de mayor valor de fitness)
    worst_indices = sorted(range(len(_population)), key=lambda i: _population[i][1])[-num_to_keep:]
    # Cree una nueva población eliminando los peores individuos
    new_population = [ind for i, ind in enumerate(_population) if i not in worst_indices]
    return new_population

# def genetic_algorithm():
#     _population = generate_population()
#     for _ in range(GENERATION_SIZE):
#         _parents = roulette_selection(_population)
#         _children = reproduction(_parents)
#         _population = replace_population(_population, _children)
#     return _population


# print(genetic_algorithm())
##Algoritmo genético, regresa la población final, los mejores resultados de cada generación y los peores resultados de cada generación
def genetic_algorithm():
    _population = generate_population()
    best_results = []
    worst_results = []
    for i in range(GENERATION_SIZE):
        _parents = roulette_selection(_population)
        _children = reproduction(_parents)
        _population = replace_population(_population, _children)
        
        # Guardar el mejor y el peor resultado de la generación
        best_result = min(_population, key=lambda x: x[1])
        worst_result = max(_population, key=lambda x: x[1])
        print("Generación {}: Mejor resultado: {}, Peor resultado: {}".format(i+1, best_result, worst_result))
        best_results.append(best_result)
        worst_results.append(worst_result)
        
    plot_results(best_results, worst_results)
    return _population, best_results, worst_results

def plot_results(best_results, worst_results):
    generations = range(1, GENERATION_SIZE + 1)
    best_fitness = [result[1] for result in best_results]
    worst_fitness = [result[1] for result in worst_results]
    
    plt.plot(generations, best_fitness, label='Mejor resultado de la generación')
    plt.plot(generations, worst_fitness, label='Peor resultado de la generación')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Evaluación de los individuos en cada generación')
    plt.legend()
    plt.grid(True)
    plt.show()

# # Ejemplo de uso
final_population, best_results, worst_results = genetic_algorithm()

#print("\nMejores resultados de cada generación:")
# for generation, result in enumerate(best_results, start=1):
#     print("Generación {}: {}".format(generation, result))

# print("\nPeores resultados de cada generación:")
# for generation, result in enumerate(worst_results, start=1):
#     print("Generación {}: {}".format(generation, result))