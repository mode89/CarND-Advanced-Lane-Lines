#!/usr/bin/env python3

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from keras.backend.tensorflow_backend import set_session
import numpy
import random
import tensorflow as tf

MAX_INDIVIDUAL_SIZE = 10
POPULATION_SIZE = 10
GENERATIONS_NUM = 1000

toolbox = base.Toolbox()

def init_tensorflow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    set_session(session)

def create_random_individual():
    size = random.randint(2, MAX_INDIVIDUAL_SIZE)
    ind = creator.Individual()
    for layerIndex in range(size):
        layer = creator.Layer()
        layer["filters"] = toolbox.random_filter_num()
        layer["kernel_size"] = toolbox.random_kernel_size()
        ind.append(layer)
    return ind

def evaluate(ind):
    value = 0
    for layer in ind:
        value += (layer["filters"] + layer["kernel_size"])
    return value,

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
    layer = random.choice(mutant)
    layerAttr = random.choice(list(layer.keys()))
    if layerAttr == "filters":
        layer["filters"] = toolbox.random_filter_num()
    elif layerAttr == "kernel_size":
        layer["kernel_size"] = toolbox.random_kernel_size()
    else:
        raise RuntimeError("Unknown layer attribute")
    del mutant.fitness.values
    return mutant,

def main():
    random.seed(42)

    init_tensorflow()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    creator.create("Layer", dict)

    toolbox.register("random_filter_num",
        lambda: random.randint(1, 16))
    toolbox.register("random_kernel_size",
        lambda: random.randint(0, 3) * 2 + 1)
    toolbox.register("individual", create_random_individual)
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