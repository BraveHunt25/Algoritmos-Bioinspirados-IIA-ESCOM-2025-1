import operator
import math
import random

import numpy

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset_2 = gp.PrimitiveSet("2VAR", 2)
pset_2.addPrimitive(operator.sub, 2)
pset_2.addPrimitive(operator.mul, 2)
pset_2.addPrimitive(protectedDiv, 2)
pset_2.addPrimitive(operator.neg, 1)
pset_2.addPrimitive(math.cos, 1)
pset_2.addPrimitive(math.sin, 1)
pset_2.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
pset_2.renameArguments(ARG0='x', ARG1='y')

creator.create("FitnessMin2", base.Fitness, weights=(-1.0,))
creator.create("Individual2", gp.PrimitiveTree, fitness=creator.FitnessMin2)

toolbox2 = base.Toolbox()
toolbox2.register("expr", gp.genHalfAndHalf, pset=pset_2, min_=1, max_=2)
toolbox2.register("individual2", tools.initIterate, creator.Individual2, toolbox2.expr)
toolbox2.register("population2", tools.initRepeat, list, toolbox2.individual2)
toolbox2.register("compile", gp.compile, pset=pset_2)

def evalSymbReg(individual, points):
    func2 = toolbox2.compile(expr=individual)
    # La función de coste es: x**3 * 5y**2 + x/2
    sqerrors = [(func2(x, y) - (x**3 * 5 * y**2 + x/2))**2 for x, y in points]
    return math.fsum(sqerrors) / len(points),                                    # Cambiamos la suma para obtener funcionalidad a costa de precisión

toolbox2.register("evaluate", evalSymbReg, points=[(x/10., y/10.) for x in range(-5,5) for y in range(-5,5)])       # Se cambian los rangos para evitar desbordamiento de memoria
toolbox2.register("select", tools.selTournament, tournsize=3 )                                      # Se usó ruleta pero se desborda la memoria por los cálculos intermedios
toolbox2.register("mate", gp.cxOnePoint)
toolbox2.register("expr_mut", gp.genFull, min_=0, max_=2)                           # se mantiene corto para evitar más errores de los necesarios
toolbox2.register("mutate", gp.mutUniform, expr=toolbox2.expr_mut, pset=pset_2)

toolbox2.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox2.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(318)

    # Generar la población
    pop2 = toolbox2.population2(n=300)
    hof2 = tools.HallOfFame(1)

    # Estadísticas
    stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size2 = tools.Statistics(len)
    mstats2 = tools.MultiStatistics(fitness=stats_fit2, size=stats_size2)
    mstats2.register("avg", numpy.mean)
    mstats2.register("std", numpy.std)
    mstats2.register("min", numpy.min)
    mstats2.register("max", numpy.max)

    # Algoritmo evolutivo
    pop2, log2 = algorithms.eaSimple(pop2, toolbox2, 0.5, 0.1, 40, stats=mstats2,
                                   halloffame=hof2, verbose=True)
    
    return pop2, log2, hof2

if __name__ == "__main__":
    main()