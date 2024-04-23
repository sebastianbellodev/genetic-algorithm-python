import random

PARENTS_SELECTED = 10

def roulette_selection(population):
    _parents = []
    _wheel_position = 0
    _population_probabilities = computation_probability(population)
    
    print("Selection probabilities:")
    for i, prob in enumerate(_population_probabilities):
        print("Individual {}: {:.4f}".format(i+1, prob))
    
    for _ in range(PARENTS_SELECTED):
        _rand = random.uniform(0, 1)
        print("\nRandom number generated:", _rand)
        _wheel_weight = 0
        
        ## Move the roulette
        _wheel_position += _rand
        _wheel_position = _wheel_position if _wheel_position <= 1 else (_wheel_position - 1)
        print("Roulette position after movement:", _wheel_position)
        
        for i, _individual_probabilities in enumerate(_population_probabilities):
            _wheel_weight += _individual_probabilities
            if _wheel_position <= _wheel_weight:
                _parents.append(population[i])
                print("Selected individual:", population[i])
                break
            else:
                continue
    return _parents

def computation_probability(population):
    total_fitness = sum(individual[1] for individual in population)
    probabilities = [individual[1] / total_fitness for individual in population]
    return probabilities

## Testing example
population = [([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10), 
              ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 8),
              ([3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 5),
              ([4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 3),
              ([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 7),
              ([6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 9),
              ([7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 6),
              ([8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 4),
              ([9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 2),
              ([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 1)]

selected_parents = roulette_selection(population)
print("\nSelected parents:")
print(selected_parents)
