#!/usr/bin/env python3

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from keras.backend.tensorflow_backend import set_session
import numpy
import os
import random
import tensorflow as tf

MIN_INDIVIDUAL_SIZE = 2
MAX_INDIVIDUAL_SIZE = 7
POPULATION_SIZE = 50
GENERATIONS_NUM = 1000

toolbox = base.Toolbox()

def init_tensorflow():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    set_session(session)

def random_individual():
    size = random.randint(2, MAX_INDIVIDUAL_SIZE)
    ind = creator.Individual()
    for layerIndex in range(size):
        ind.append(random_layer())
    return ind

def random_layer():
    layer = creator.Layer()
    layer["filters"] = random_filter_num()
    layer["kernel_size"] = random_kernel_size()
    return layer

def random_filter_num():
    return random.randint(1, 16)

def random_kernel_size():
    return random.randint(0, 5) * 2 + 1

def evaluate(ind):
    valLosses = list()
    lossDiffs = list()
    for i in range(3):
        model = binary_filter.Model(ind)
        valLoss, trainLoss = model.train_model()
        valLosses.append(valLoss)
        lossDiffs.append(abs(valLoss - trainLoss))
        clear_session()
    valLoss = sum(valLosses) / len(valLosses)
    lossDiffs = sum(lossDiffs) / len(lossDiffs)
    return valLoss, lossDiffs

def mate(ind1, ind2):
    ind1Size = len(ind1)
    ind2Size = len(ind2)
    while True:
        ind1LowerSize = random.randint(1, ind1Size - 1)
        ind1UpperSize = ind1Size - ind1LowerSize
        ind2LowerSize = random.randint(1, ind2Size - 1)
        ind2UpperSize = ind2Size - ind2LowerSize
        newInd1Size = ind1LowerSize + ind2UpperSize
        newInd2Size = ind2LowerSize + ind1UpperSize
        newInd1SizeValid = newInd1Size <= MAX_INDIVIDUAL_SIZE
        newInd2SizeValid = newInd2Size <= MAX_INDIVIDUAL_SIZE
        if newInd1SizeValid and newInd2SizeValid:
            ind1Lower = ind1[:ind1LowerSize]
            ind1Upper = ind1[ind1LowerSize:]
            ind2Lower = ind2[:ind2LowerSize]
            ind2Upper = ind2[ind2LowerSize:]
            newInd1 = creator.Individual(ind1Lower + ind2Upper)
            newInd2 = creator.Individual(ind2Lower + ind1Upper)
            return newInd1, newInd2

def mutate(ind):
    mutant = toolbox.clone(ind)
    del mutant.fitness.values

    mutations = {
        mutate_filter_num: 10,
        mutate_kernel_size: 10
    }

    if len(mutant) < MAX_INDIVIDUAL_SIZE:
        mutations[mutate_append_layer] = 1
    if len(mutant) > MIN_INDIVIDUAL_SIZE:
        mutations[mutate_remove_layer] = 1

    funcs = list(mutations.keys())
    weights = [mutations[func] for func in funcs]
    mutation = random.choices(funcs, weights)[0]
    mutation(mutant)

    return mutant,

def mutate_filter_num(ind):
    layer = random.choice(ind)
    layer["filters"] = random_filter_num()

def mutate_kernel_size(ind):
    layer = random.choice(ind)
    layer["kernel_size"] = random_kernel_size()

def mutate_append_layer(ind):
    ind.append(random_layer())

def mutate_remove_layer(ind):
    ind.pop()

def main():
    random.seed(42)

    init_tensorflow()

    creator.create("FitnessMin", base.Fitness, weights=(-2.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    creator.create("Layer", dict)

    toolbox.register("individual", random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POPULATION_SIZE)
    hallOfFame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    population, log = algorithms.eaSimple(
        population, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS_NUM,
        stats=stats, halloffame=hallOfFame, verbose=True)

if __name__ == "__main__":
    main()
