# # Proyecto: Minimización con algoritmo de abejas a hiperparámetros de bosque aleatorio de clasificación 
# **Hernández Jiménez Erick Yael 2023630748**
# **Algortimos Bioinspirados de Ingeniería en Inteligencia Artificial 2025-1**
# **Escuela Superior de Cómputo**
# Descripción:
# Este archivo `ipynb` implementa el algoritmo bio inspirado en abejas para minimizar los hiperparámetros que regulan el comportamiento de bosques aleatorios de clasificación para un conjunto de datos limpio y sin datos faltantes, usando particularmente [*agaricus-lepiota*](https://archive.ics.uci.edu/dataset/73/mushroom).

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

def gain(values: list, dataset: DataFrame)->float:
    r'''
    - values: list -> [n_estimators, max_features, max_samples, max_depth, max_leaf_nodes]
    - dataset: DataFrame -> Conjunto de datos a evaluar
    '''
    # Consideraremos un promedio tanto de la evaluación con los datos de entrenamiento como de los datos de prueba
    bosque = RandomForestClassifier(criterion="gini", bootstrap=True, oob_score=True, n_estimators=values[0], max_features=values[1], max_samples=values[2], max_depth=values[3], max_leaf_nodes=values[4])
    bosque.fit(dataset.drop("class", axis=1), dataset["class"])
    score: float = bosque.score(dataset.drop("class", axis=1), dataset["class"])
    score += bosque.oob_score_
    score /= 2
    return score

# No se incluirá la función de peso en este caso, ya que no es necesario para el problema de clasificación
class Worker():
    def __init__(self, limits: list[tuple]):
        r'''
        Inicializa la trabajadora con la función `xij = lj + r*(uj - lj)`, el límite en 0 
        y reevalúa en caso de que el resultado de la exploración sobrepase el límite 
        de peso de la mochila.
        '''
        # Inicializa la fuente de comida
        self.content: list[int] = [int(d_min + np.random.rand()*(d_max - d_min)) if isinstance(d_min, int) else d_min + np.random.rand()*(d_max - d_min) for d_min, d_max in limits]
        # Inicializa el límite de la solución en 0
        self.limit = 0
        
    def __str__(self):
        return f"{self.content}"

class Hive():
    r'''
    Clase que almacena toda la colmena con las soluciones a la función.

    La implementación está aplicado concretamente para el problema de minimización de hiperparámetros 
    de clasificación con bosque aleatorio para cualquier conjunto de datos limpio. E inicializa a la 
    abeja más valiosa (mvb) como la primera generada.
    '''
    def __init__(self, p_size: int, limits: list[tuple], dataset: DataFrame):
        self.population: list[Worker] = [
            # Inicializamos la abeja con 7 variables
            Worker(limits) for _ in range(p_size//2)]
        self.mvb: list = self.population[0].content
        self.range: list[tuple] = limits
        self.dataset: DataFrame = dataset

    def __str__(self):
        result: str = ""
        for dbx, bee in enumerate(self.population):
            result += f"{f"[{dbx}]": >5} {bee}\n"
        return result
    
    def worker_iteration(self)->None:
        r'''
        Función que ejecuta el ciclo correspondiente a las abejas obreras
        '''
        for index in range(len(self.population)):
            # Índice para la variable a explotar
            j: int = np.random.randint(low = 0, high = len(self.population[0].content))
            # Índice para la solución a mejorar
            k: int = np.random.randint(low = 0, high = len(self.population) - 1)
            # Evasión de autoreferenciado
            while k == index:
                k = np.random.randint(low = 0, high = len(self.population) - 1)
            # Generación de random
            r: float = np.random.uniform(low = -1, high = 1)
            value_i_j: int | float = self.population[index].content[j]
            value_k_j: int | float = self.population[k].content[j]
            if isinstance(value_i_j, float):
                value_i_j_t= float(np.clip(value_i_j + r*(value_i_j - value_k_j), min = self.range[j][0], max = self.range[j][1]))
            elif isinstance(value_i_j, int):
                value_i_j_t = int(np.clip(int(value_i_j + r*(value_i_j - value_k_j)), min = self.range[j][0], max = self.range[j][1]))
            temp_bee: list = [temp if dx != j else value_i_j_t for dx, temp in enumerate(self.population[index].content)]
            
            original_gains: float = gain(self.population[index].content, self.dataset)*1_000_000
            new_gains: float = gain(temp_bee, self.dataset)*1_000_000
            
            #print(f"{self.population[index]} gains: {original_gains}")
            #print(f"{temp_bee} gains: {new_gains}")
            if new_gains >= original_gains:
                self.population[index].content = temp_bee
                self.population[index].limit = 0
            else:
                self.population[index].limit += 1
            #print(f"{self.population[index]}. Límite = {self.population[index].limit}")
            #print('*'*100, '\n')

    def roulette_sel(self)->tuple[int, Worker]:
        r'''
        Función que aplica el algoritmo de selección por ruleta con la población de la colmena
        '''
        # Inicializamos el fitness total en 0
        total_fitness: float = 0
        # Inicializamos la lista de probabilidades con el fitness de cada abeja
        fitness: list[float] = [gain(worker.content, self.dataset) for worker in self.population]
        # Calculamos el fitness total
        for worker in fitness:
            total_fitness += worker
        # print(total_fitness)
        # print(fitness)
        # Dividimos el fitness particular entre el fitness total
        probabilities: list[float] = [worker/total_fitness for worker in fitness]
        # print(probabilities)
        # Inicializamos las probabilidades acumuladas con la primera probabilidad
        prob_acum: list[float] = [probabilities[0]]
        # Iteramos sobre el resto de probabilidades para acumularlas
        for index, prob in enumerate(probabilities[1:]):
            prob_acum.append(prob_acum[index] + prob)
        # print(prob_acum)
        # Seleccionamos un random
        r_sel = np.random.random()
        # print(f"{r_sel=}")
        # Inicializamos el índice de la obrera seleccionada como la primera de la lista
        selected: int = 0
        # Iteramos sobre las probabilidades acumuladas
        for i, prob, in enumerate(prob_acum):
            # Rompemos el ciclo si el random es menor que la probabilidad acumulada
            if r_sel < prob:
                break
            # De lo contrario, reasignamos a la obrera asignada como la que le sigue
            else:
                selected += 1
        #print(f"{f"Seleccionando ({selected})": <25}{self.population[selected]}")
        # Regresamos la obrera seleccionada
        return (selected, self.population[selected])

    def unlooker_iteration(self)->None:
        r'''
        Función que ejecuta el ciclo correspondiente a las abejas observadoras
        '''
        # Iteramos sobre la población de la colmena
        for _ in range(len(self.population)):
            # Seleccionamos a la abeja a mejorar por ruleta
            guide: tuple[int, Worker] = self.roulette_sel()
            # Seleccionamos la variable a mejorar
            j: int = np.random.randint(low = 0, high = len(guide[1].content))
            # Seleccionamos a la abeja guía por azar
            k: int = np.random.randint(low = 0, high = len(self.population))
            # Evasión de auterreferenciado
            while(k == guide[0]):
                k = np.random.randint(low = 0, high = len(self.population))
            # Extraemos el valor de la seleccionada por ruleta en j
            w_i_j: int | float = guide[1].content[j]
            # Extraemos el valor de la seleccionada por azar en j
            w_k_j: int | float = self.population[k].content[j]
            # Inicializamos un número aleatorio
            r: float = np.random.uniform(low = -1, high = 1)
            # Aseguramos los límites
            if isinstance(w_i_j, float):
                w_i_j_t = float(np.clip(w_i_j + r*(w_i_j - w_k_j), min = self.range[j][0], max = self.range[j][1]))
            elif isinstance(w_i_j, int):
                w_i_j_t = int(np.clip(int(w_i_j + r*(w_i_j - w_k_j)), min = self.range[j][0], max = self.range[j][1]))
            # Generamos la los valores de la abeja temporalmente
            temp_bee: list = [temp if dx != j else w_i_j_t for dx, temp in enumerate(guide[1].content)]
            
            original_gains: float = gain(guide[1].content, self.dataset)*1_000_000
            new_gains: float = gain(temp_bee, self.dataset)*1_000_000
            
            #print(f"{guide[1]} gains: {original_gains}")
            #print(f"{temp_bee} gains: {new_gains}")
            if new_gains >= original_gains:
                self.population[guide[0]].content = temp_bee
                self.population[guide[0]].limit = 0
            else:
                self.population[guide[0]].limit += 1
            #print(f"{self.population[guide[0]]}. Límite = {self.population[guide[0]].limit}")
            #print('*'*100, '\n')
    
    def explorer_iteration(self)->None:
        r'''
        Función que ejecuta el ciclo correspondiente a las abejas exploradoras
        '''
        for i in range(len(self.population)):
            # Reducimos el límite de la abeja a 3
            if self.population[i].limit >= 3:
                if gain(self.mvb, self.dataset) <= gain(self.population[i].content, self.dataset):
                    self.mvb = self.population[i].content
                self.population[i] = Worker(self.range)
            else: 
                continue
        print(f"Mejor resultado histórico: {self.mvb}\nGanancia: {gain(self.mvb, self.dataset)}")

if __name__ == "__main__":
    dataset = pd.read_csv("./dataset/mushroom/agaricus-lepiota.csv")
    print(dataset.head())
    
    numerical_dataset = dataset.copy()
    
    for column in numerical_dataset.columns:
        if numerical_dataset[column].dtype == 'object':
            numerical_dataset[column] = numerical_dataset[column].astype('category')
            
            print(f"Equivalencias para la columna '{column}':")
            equivalencias = dict(enumerate(numerical_dataset[column].cat.categories))
            print(equivalencias)
            
            numerical_dataset[column] = numerical_dataset[column].cat.codes
            
        elif numerical_dataset[column].dtype == 'category':
            
            print(f"Equivalencias para la columna '{column}':")
            equivalencias = dict(enumerate(numerical_dataset[column].cat.categories))
            print(equivalencias)
            
            numerical_dataset[column] = numerical_dataset[column].cat.codes
    
    print(numerical_dataset.head())
    
    # [n_estimators, max_features, max_samples, max_depth, max_leaf_nodes]
    colmena: Hive = Hive(40, [(60, 150), (0.1, 0.9), (0.01, 0.99), (5, 7), (15, 30)], numerical_dataset)
    max_interaciones : int = 50
    print(colmena)
    for i in range(max_interaciones):
        print(f" {i}/{max_interaciones} {'='*i}>{' '*(max_interaciones-i-1)} {datetime.now().strftime("%H:%M:%S")}")
        #print("\n\n", f" Ciclo de obreras {i} ".center(100, '-'))
        colmena.worker_iteration()
        #print("\n\n", f" Ciclo de observadoras {i} ".center(100, '-'))
        colmena.unlooker_iteration()
        colmena.explorer_iteration()
    
    print(f"Mejor abeja: {colmena.mvb}. Ganancia: {gain(colmena.mvb, numerical_dataset)}\nPoblación final:\n{colmena}")
    
    bosque = RandomForestClassifier(criterion="gini", bootstrap=True, oob_score=True, n_estimators=colmena.mvb[0], max_features=colmena.mvb[1], max_samples=colmena.mvb[2], max_depth=colmena.mvb[3], max_leaf_nodes=colmena.mvb[4])
    
    bosque.fit(numerical_dataset.drop("class", axis=1), numerical_dataset["class"])
    print(bosque.score(numerical_dataset.drop("class", axis=1), numerical_dataset["class"]))
    print(bosque.oob_score_)
    print((bosque.score(numerical_dataset.drop("class", axis=1), numerical_dataset["class"]) + bosque.oob_score_)/2)
    
    # for arbol in bosque.estimators_:
    #     plt.figure(figsize=(25, 15))
    #     tree.plot_tree(arbol, feature_names=numerical_dataset.columns[:-1], class_names=["edible", "poisonous"], filled=True)
    #     plt.show()