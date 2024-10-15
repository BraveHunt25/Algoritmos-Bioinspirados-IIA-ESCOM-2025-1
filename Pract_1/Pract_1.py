# Hernández Jiménez Erick Yael 2023630748
import numpy as np
import copy
from typing import Tuple
PRECIOS = np.array([10,8,12,6,3,2,2])
COSTOS = np.array([4,2,5,5,2,1.5,1])
PC = 0.85
PM = 0.1
UMBRAL = 0.5
GENERACIONES = 50
LIMITE = 1000 # Agregué este máximo debido al tiempo de ejecución de búsqueda de mutaciones adecuadas
def funcion_costo(cromosoma: np.array) -> float:
    resultado: float = np.dot(PRECIOS, cromosoma)
    return resultado
def funcion_restriccion(cromosoma: np.array) -> float:
    resultado: float = np.dot(COSTOS, cromosoma)
    return resultado
def generar_poblacion(poblacion: list, num_individuos: int = 10) -> list:
    for _ in range(num_individuos):
        cromosoma = np.zeros(7, dtype=int)
        while (funcion_restriccion(cromosoma) <= 0) or (funcion_restriccion(cromosoma) > 30):
            cromosoma = np.random.randint(0, 11, size=7)
            cromosoma[1] = np.random.randint(3,11)
            cromosoma[3] = np.random.randint(2,11) 
        poblacion.append(cromosoma)
        # print(cromosoma, f"agregado con costo de {funcion_restriccion(cromosoma)}\n")
    return poblacion
def calcular_aptitudes(poblacion: np.array) -> np.array:
    aptitudes_individuales = np.array([funcion_costo(individuo) for individuo in poblacion], dtype=float)
        # print(f"La aptitud del individuo '{poblacion[i]}' es de {aptitudes_individuales[i]}")
        # aptitud_total += np.dot(poblacion[i], precios)
    return aptitudes_individuales
def probabilidades_acumuladas(aptitudes_individuales: np.array) -> np.array:
    aptitud_total = np.sum(aptitudes_individuales)
    probabilidad_acumulada: np.array = np.zeros(len(aptitudes_individuales), dtype=float)
    probabilidad_acumulada[0] = aptitudes_individuales[0] / aptitud_total
    for i in range(1, len(probabilidad_acumulada)):
        probabilidad_acumulada[i] = probabilidad_acumulada[i-1] + (aptitudes_individuales[i] / aptitud_total)
    # print("Las probabilidades son:\n")
    # for i, probabilidad in enumerate(probabilidad_acumulada):
    #     print(f"Individuo en la posición {i} tiene una probabilidad de {probabilidad}")
    return probabilidad_acumulada
def seleccion_padres(poblacion: list, prob_acum: np.array) -> list:
    padres: list = []
    indices_a_eliminar: list = []

    # Seleccionar el primer padre
    frecuencia_1: float = np.random.rand()
    individuo_1: int = np.searchsorted(prob_acum, frecuencia_1)
    individuo_1 = min(individuo_1, len(poblacion) - 1)
    padre_1 = poblacion[individuo_1]
    padres.append(padre_1)
    indices_a_eliminar.append(individuo_1)

    # Seleccionar el segundo padre, asegurándose de que sea diferente al primero
    individuo_2: int = individuo_1
    while individuo_2 == individuo_1:
        frecuencia_2: float = np.random.rand()
        individuo_2 = np.searchsorted(prob_acum, frecuencia_2)
        individuo_2 = min(individuo_2, len(poblacion) - 1)
    padre_2 = poblacion[individuo_2]
    padres.append(padre_2)
    indices_a_eliminar.append(individuo_2)
    for i in sorted(indices_a_eliminar, reverse=True):
        del poblacion[i]
    return padres
def cruza(pareja: list) -> list:
    cromosoma_cruza: np.array = np.random.rand(len(pareja[0]))
    hijos: list = []
    hijo_1: np.array = np.zeros(len(pareja[0]), dtype=int)
    hijo_2: np.array = np.zeros(len(pareja[0]), dtype=int)
    # for i in range(len(cromosoma_cruza)):
    #     print(f"Posición {i} = {cromosoma_cruza[i]}")
    for i in range(len(cromosoma_cruza)):
        if cromosoma_cruza[i] < UMBRAL:
            hijo_1[i] = pareja[0][i]
            hijo_2[i] = pareja[1][i]
        else:
            hijo_2[i] = pareja[0][i]
            hijo_1[i] = pareja[1][i]      
    # print("Los hijos generados son:")
    # print(hijo_1)
    # print(hijo_2)
    # Evaluando quienes son los mejores
    # print("Comparando hijos con padres")
    candidatos: list = [hijo_1, hijo_2] + list(pareja)
    evaluaciones = [funcion_costo(candidato) for candidato in candidatos]
    # for candidato, evaluacion in zip(candidatos, evaluaciones):
    #     print(f"El candidato {candidato} es valuado en {evaluacion}")
    indices_mejores = np.argsort(evaluaciones)[-2:]
    hijos = [candidatos[indice] for indice in indices_mejores]
    return hijos
def mutacion(individuo: list) -> np.array:
    # for _ in range(LIMITE): 
    while True:
        cromosoma_mutacion: np.array = np.random.rand(len(individuo))
        mutado: np.array = individuo.copy()
        mutado[0] = np.where(cromosoma_mutacion[0] < PM, np.random.randint(0, 10), mutado[0])
        mutado[1] = np.where(cromosoma_mutacion[1] < PM, np.random.randint(3, 10), mutado[1])
        mutado[2] = np.where(cromosoma_mutacion[2] < PM, np.random.randint(0, 10), mutado[2])
        mutado[3] = np.where(cromosoma_mutacion[3] < PM, np.random.randint(2, 10), mutado[3])
        mutado[4:] = np.where(cromosoma_mutacion[4:] < PM, np.random.randint(0, 10, size=len(mutado[4:])), mutado[4:])
        if 0 < funcion_restriccion(mutado) <= 30:
            return mutado
    return individuo
print("Generando población...")
poblacion = []
poblacion = generar_poblacion(poblacion)
for i, individuo in enumerate(poblacion):
    print(f"Individuo {i}: {individuo}")
for i in range(GENERACIONES):
    vieja_generacion = copy.deepcopy(poblacion)
    nueva_generacion: list = []
    print('-'*20, f" Generación {i+1}", '-'*20)
    # print('-'*50, "\nSeleccionando los pares de padres...")
    padres: list = []
    while len(poblacion) > 1:
        # print('-'*50, "\nCalculando aptitudes...")
        aptitudes_1 = calcular_aptitudes(poblacion)
        # for i, aptitud in enumerate(aptitudes_1):
        #     print(f"Individuo {i}: {aptitud}")
        # print('-'*50, "\nCalculando aptitud total...")
        # print(f"La aptitud total de la población es {aptitud_total_1}")
        # print('-'*50, "\nCalculando probabilidades acumuladas para la ruleta...")
        probabilidades_acumuladas_1 = probabilidades_acumuladas(aptitudes_1)
        # for i, probabilidad in enumerate(probabilidades_acumuladas_1):
        #     print(f"Individuo {i} con probabilidad acumulada de {probabilidad}")
        padres.append(seleccion_padres(poblacion, probabilidades_acumuladas_1))
        # print("Los pares generados hasta el momento son:")
        # for i, pareja in enumerate(padres):
        #     print(f"Pareja {i} = {pareja}")
        # print("La población ahora contiene los siguientes individuos:\n", '='*50)
        # for i, individuo in enumerate(poblacion):
        #     print(f"Individuo {i}: {individuo}")
    # print("Los pares generados hasta el momento son:")
    # for i, pareja in enumerate(padres):
    #     print(f"Pareja {i} = {pareja}")
    # print('-'*50, "\nCruzando las parejas...")
    for pareja in padres:
        se_cruzan = np.random.rand()
        if se_cruzan < PC:
            # print("Cruzando la pareja")
            # print("De los padres:")
            # for padre in pareja:
            #     print(padre)
            # print("Agregando ganadores a la siguiente generación")
            nuevos_hijos = cruza(pareja)
            # for ganador in nuevos_hijos:
            #     print(ganador)
            nueva_generacion.extend(nuevos_hijos)
            # print('*'*30)
        else:
            # print("No se cruzan, así que mutan") # página 27
            for padre in pareja:
                mutado = mutacion(padre)
                # print("Añadiendo individuo a la siguiente generación")
                nueva_generacion.append(mutado)
    print("Los hijos que se tienen son:")
    for hijo in nueva_generacion:
        print(hijo)
    if all(np.array_equal(viejo, nuevo) for viejo, nuevo in zip(vieja_generacion, nueva_generacion)):
        print("Población sin cambios, deteniendo...")
        break
    poblacion = nueva_generacion

    
print("Los resultados son:")
for resultado in nueva_generacion:
    print(f"{resultado}: {funcion_costo(resultado)} galleones con un peso de {funcion_restriccion(resultado)}")

# Con límites
# [1 7 1 3 0 4 4]
# [1 6 1 2 2 6 8]
# [1 5 2 3 4 0 6]
# [2 8 1 2 2 2 7]
# 
# Sin límites:
# Pasaron 20 minutos sin cumplir las condiciones. Se generó el siguiente arreglo en la última generación completa 18:
# [2 5 1 3 3 1 8]
# [2 5 1 3 3 1 8]
# [2 5 1 3 3 0 8]
# [2 5 1 3 3 1 8]
# [2 5 1 3 3 1 8]
# [2 5 1 3 3 1 8]
# [2 5 1 3 3 1 8]
# [2 5 1 3 3 1 8]
# [0 5 0 3 0 1 1]
# [0 5 0 2 3 1 2]
# 
# [0 5 0 3 4 7 4]
# [0 5 1 3 4 7 4]
# [0 5 1 3 4 0 4]
# [0 5 1 3 4 7 4]
# [0 5 1 3 4 7 4]
# [0 5 1 3 4 7 4]
# [0 3 1 3 0 0 4]
# [0 3 1 3 0 0 4]
# [0 5 1 3 0 0 0]
# [0 5 1 3 0 0 0]
#
# [1 7 1 2 3 2 3]: 109
# [1 7 1 2 3 2 3]: 109
# [0 3 1 2 3 2 0]: 61
# [1 3 0 2 3 2 1]: 61
# [1 7 1 2 3 2 3]: 109
# [1 7 1 2 3 2 3]: 109
# [0 4 0 2 3 2 3]: 63
# [1 3 0 2 3 1 1]: 59
# [1 7 1 2 3 2 3]: 109
# [1 7 1 2 3 2 3]: 109
#
# [1 6 1 2 2 3 6]: 106
# [1 6 1 2 2 3 6]: 106
# [1 6 1 2 2 0 0]: 88
# [1 6 1 2 2 3 6]: 106
# [1 6 1 2 0 3 0]: 88
# [1 6 1 2 2 0 0]: 88
# [1 6 0 2 2 0 0]: 76
# [1 6 0 2 2 0 6]: 88
# [0 6 0 2 0 3 1]: 68
# [0 6 0 2 0 3 1]: 68