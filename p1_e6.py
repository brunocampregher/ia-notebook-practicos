# (práctica 1 - ejercicio 6) TSP con MOGA utilizando deap
from deap import base, creator, tools, algorithms
import random
import itertools
import matplotlib.pyplot as plt

# [ACLARACIÓN]: para la resolución del ejercicio modelamos las ciudades con
# un grafo completo y no dirigido.

# hiperparametros
POPULATION_SIZE = 30  # tamaño de la población. El limite es la cant de permutaciones de las ciudades sin contar de la que se parte
GENERATIONS = 40      # cantidad de generaciones a evolucionar
CROSOVER_PROB = 0.8   # probabilidad de crossover
MUTATION_PROB = 0.04  # probabildiad de mutación
CANT_CITIES = 5       # cantidad de ciudades
START_CITY = 0        # ciudad de la que se parte

# matriz para guardar las distancias de una ciudad a otra
CITIES_DIST = [
#   0   1   2   3   4
   [0,  10, 40, 1 , 10], # 0
   [10, 0,  4,  12, 20], # 1
   [40, 4,  0,  6 , 4 ], # 2
   [1,  12, 6,  0 , 8 ], # 3
   [10, 3,  8,  9 , 0 ]  # 4
]

# matriz para guardar el tiempo asocidado de ir de una ciudad a otra por el tráfico
CITIES_TIMES = [
#    0  1   2  3   4
    [0, 20, 4, 12, 8 ], # 0
    [20, 0, 10, 6, 10], # 1
    [4, 10, 0, 16, 20], # 2
    [12, 6, 16, 0, 5 ], # 3
    [3,  9, 14, 10, 0]  # 4
]

# cada individuo/cromosoma es una lista que denota un posible recorrido.
# ej: el individuo [0,1,2,3,0] representa el recorrido de ir de la
# ciudad 0 a la 1, luego de la 1 a la 2, despues de la 2 a la 3 y
# finalmente de la 3 a la 0.

# definimos el tipo de fitness y el individuo
creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0)) # queremos minimizar ambos objetivos
creator.create("Individual", list, fitness=creator.FitnessMin)

# toolbox para generar individuos y población
toolbox = base.Toolbox()

# para crear la población inicial lo que hago es obtener una lista con todas
# las ciudades menos la de arranque y luego voy generando una permutación
# aleatoria hasta completar la población. Al final antes de agregar un individuo
# le concateno la ciudad de donde se parte al principio y la final.
def create_population():
    cities = list(range(CANT_CITIES))
    cities.remove(START_CITY)
    poblacion = []

    for _ in range(POPULATION_SIZE):
        # genera una permutación aleatoria de las ciudades
        ruta_intermedia = random.sample(cities, len(cities))
        ruta = [START_CITY] + ruta_intermedia + [START_CITY]
        poblacion.append(creator.Individual(ruta))

    return poblacion

# la población es una lista de individuos
toolbox.register("population", create_population)

# obtener el costo de ir de la ciudad A a la B (distancia)
def get_cost(cityA, cityB):
    return CITIES_DIST[cityA][cityB]

# obtener el tiempo de ir de A y B
def get_time(cityA, cityB):
    return CITIES_TIMES[cityA][cityB]

# fitness costo por distancia
def evalCost(individual):
   cost = 0
   for i in range(CANT_CITIES):
       cost += get_cost(individual[i], individual[i+1])
   return cost

# fitness tiempo por tráfico
def evalTime(individual):
    time = 0
    for i in range(CANT_CITIES):
        time += get_time(individual[i], individual[i+1])
    return time

# evalua un individuo suando las 2 fitness
def eval(individual):
    return evalCost(individual), evalTime(individual)

toolbox.register("evaluate", eval)

# función de crossover usando 1 point crossover modificado. primero se
# excluye la ciudad de inicio para evitar crear individuos inválidos.
# Entonces se parten los 2 padres (padre1 y padre2) en un punto. Luego
# para generar el hijo1 se toma el primer pedazo del del padre1 y se rellena
# el resto con el padre2 de forma que no queden repetidos para evitar
# individuos inválidos. Luego para generar el hijo2 se toma la primer parte
# de padre2 y se rellena con el resto del padre1 de igual forma sin dejar
# ciudades repetidas.
def crossover(p1, p2):
    # excluyo la ciudad de inicio (y de fin)
    p1_inner = p1[1:-1]
    p2_inner = p2[1:-1]

    # genero el punto para partir:
    point = random.randint(1, len(p1_inner) - 1)

    # genero la primer mitad de cada hijo:
    offspring1 = [START_CITY] + p1_inner[:point]
    offspring2 = [START_CITY] + p2_inner[:point]

    # ahora relleno el resto de cada hijo con el otro padre respectivamente:
    for city in p2_inner:
        if city not in offspring1: # chequeo para evitar ciudades repetidas
            offspring1.append(city)
    for city in p1_inner:
        if city not in offspring2:
            offspring2.append(city)

    offspring1.append(START_CITY)
    offspring2.append(START_CITY)

    return creator.Individual(offspring1), creator.Individual(offspring2)

# registro la función crossover
toolbox.register("mate", crossover)

# operador de mutación: intercambia 2 ciudades de lugar menos
# la ciudad de inicio (y de fin que es la misma) para evitar individuos invalidos
def mutate(individual):
    index1 = random.randint(1, len(individual)-2)
    index2 = random.randint(1, len(individual)-2)
    individual[index1], individual[index2] = individual[index2], individual[index1]
    return individual,

# registro la función mutación
toolbox.register("mutate", mutate)   # mutation bit flip

# utilizo NSGA2 como algoritmo para la selección
toolbox.register("select", tools.selNSGA2)

# función para graficar todas las posibles soluciones y el frente de pareto que encontro NSGA-II
def graficar(hof):
    # genero todas las posibles soluciones (permutaciones)
    cities = list(range(CANT_CITIES))
    cities.remove(START_CITY)
    permutaciones = list(itertools.permutations(cities))
    permutaciones = [list(p) for p in permutaciones]

    todas_las_soluciones = []
    for perm in permutaciones:
        ruta = [START_CITY] + perm + [START_CITY]
        todas_las_soluciones.append(creator.Individual(ruta))

    # para cada individuo le calculo cada fitness y
    # genero todos los puntos (evalCost(i-esimo_individuo),evalTime(i-esimo_individuo))
    todas_eval = [eval(ind) for ind in todas_las_soluciones]

    # de todos los puntos obtengo los que estan en el frente de pareto
    pareto_eval = [ind.fitness.values for ind in hof]

    # listas para graficar
    xs_todas, ys_todas = [], []
    xs_pareto, ys_pareto = [], []

    for (x, y) in todas_eval:
        if (x, y) in pareto_eval: # si estan en el frente de pareto los guardo aca para pintar de verde
            xs_pareto.append(x)
            ys_pareto.append(y)
        else:                    # sino los guardo aca para pintar de azul
            xs_todas.append(x)
            ys_todas.append(y)

    # gráfico
    plt.figure(figsize=(7,5))
    plt.scatter(xs_todas, ys_todas, color='blue', label='Todas las soluciones')
    plt.scatter(xs_pareto, ys_pareto, color='green', s=100, edgecolor='black', label='Frente de Pareto')

    plt.xlabel('Distancia total')
    plt.ylabel('Tiempo total')
    plt.title('Todas las soluciones (azul) y Frente de Pareto - NSGA-II (verde)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    random.seed(42)
    pop = toolbox.population()
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: tuple(sum(f)/len(f) for f in zip(*fits)))
    stats.register("min", lambda fits: tuple(min(f) for f in zip(*fits)))
    stats.register("max", lambda fits: tuple(max(f) for f in zip(*fits)))

    # evolucionar con NSGA-II
    algorithms.eaMuPlusLambda(pop, toolbox, mu=30, lambda_=30, cxpb=CROSOVER_PROB, mutpb=MUTATION_PROB, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    print("\n--- Frente de Pareto ---")
    for ind in hof:
        print(ind, ind.fitness.values)

    graficar(hof)
main()
