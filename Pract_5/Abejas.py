import numpy as np
from numpy.typing import NDArray

def gain(items: list)->int:
    # Consideremos que los precios son:
    prices: list[int] = [10,8,12,6,3,2,2]
    gains: int = 0
    for price, dx in zip(prices, items):
        gains += price*dx
    return gains

def weight(items: list)->float:
    # Consideremos que los pesos son:
    weights: list[int] = [4,2,5,5,2,1.5,1]
    t_weight: int = 0
    for dw, dx in zip(weights, items):
        t_weight += dw*dx
    return t_weight


class Worker():
    def __init__(self, min: list[int], max: int):
        r'''
        Inicializa la trabajadora con la función `xij = lj + r*(uj - lj)`, el límite en 0 
        y reevalúa en caso de que el resultado de la exploración sobrepase el límite 
        de peso de la mochila.
        '''
        # Inicializa la fuente de comida
        self.content: list[int] = [int(d_min + np.random.rand()*(max - d_min)) for d_min in min]
        # Reevalúa en caso de que la solución explorada sobrepase el límite de la mochila
        while(weight(self.content) > 30):
            self.content: list[int] = [int(d_min + np.random.rand()*(max - d_min)) for d_min in min]
        # Inicializa el límite de la solución en 0
        self.limit = 0
        
    def __str__(self):
        return f"{self.content}"

class Hive():
    r'''
    Clase que almacena toda la colmena con las soluciones a la función.

    La implementación está aplicado concretamente para el problema de la mochila de 
    objetos de Harry Potter. E inicializa a la abeja más valiosa (mvb) como la primera generada.
    '''
    def __init__(self, p_size: int, min: list[int], max: int):
        self.population: list[Worker] = [
            # Inicializamos la abeja con 7 variables
            Worker(min, max) for _ in range(p_size//2)]
        self.mvb: list[int] = self.population[0].content

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
            k = np.random.randint(low = 0, high = len(self.population) - 1)
            # Evasión de autoreferenciado
            while k == index:
                k = np.random.randint(low = 0, high = len(self.population) - 1)
            # Generación de random
            r: float = np.random.uniform(low = -1, high = 1)
            value_i_j: int = self.population[index].content[j]
            value_k_j: int = self.population[k].content[j]
            if j != 1 and j != 3:
                value_i_j_t: int = int(np.clip(int(value_i_j + r*(value_i_j - value_k_j)), min = 0, max = 10))
            elif j == 1:
                value_i_j_t: int = int(np.clip(int(value_i_j + r*(value_i_j - value_k_j)), min = 3, max = 10))
            else:
                value_i_j_t: int = int(np.clip(int(value_i_j + r*(value_i_j - value_k_j)), min = 2, max = 10))
            temp_bee: list = [temp if dx != j else value_i_j_t for dx, temp in enumerate(self.population[index].content)]
            
            # Evasión de exceso de peso
            while(weight(temp_bee) > 30):
                r = np.random.uniform(low = -1, high = 1)
                if j != 1 and j != 3:
                    value_i_j_t: int = int(np.clip(int(value_i_j + r*(value_i_j - value_k_j)), min = 0, max = 10))
                elif j == 1:
                    value_i_j_t: int = int(np.clip(int(value_i_j + r*(value_i_j - value_k_j)), min = 3, max = 10))
                else:
                    value_i_j_t: int = int(np.clip(int(value_i_j + r*(value_i_j - value_k_j)), min = 2, max = 10))
                temp_bee: list = [temp if dx != j else value_i_j_t for dx, temp in enumerate(self.population[index].content)]

            print(f"{self.population[index]} gains: {gain(self.population[index].content)}")
            print(f"{temp_bee} gains: {gain(temp_bee)}")
            if gain(temp_bee) > gain(self.population[index].content):
                self.population[index].content = temp_bee
                self.population[index].limit = 0
            else:
                self.population[index].limit += 1
            print(f"{self.population[index]}. Límite = {self.population[index].limit}. Peso = {weight(self.population[index].content)}")
            print('*'*30)

    def roulette_sel(self)->tuple[int, Worker]:
        r'''
        Función que aplica el algoritmo de selección por ruleta con la población de la colmena
        '''
        # Inicializamos el fitness total en 0
        total_fitness: float = 0
        # Inicializamos la lista de probabilidades con el fitness de cada abeja
        fitness: list[float] = [gain(worker.content) for worker in self.population]
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
        print(f"Seleccionando ({selected}){self.population[selected]}...")
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
            w_i_j: float = guide[1].content[j]
            # Extraemos el valor de la seleccionada por azar en j
            w_k_j: float = self.population[k].content[j]
            # Inicializamos un número aleatorio
            r: float = np.random.uniform(low = -1, high = 1)
            # Aseguramos los límites
            if j != 1 and j != 3:
                value_i_j_t: int = int(np.clip(int(w_i_j + r*(w_i_j - w_k_j)), min = 0, max = 10))
            elif j == 1:
                value_i_j_t: int = int(np.clip(int(w_i_j + r*(w_i_j - w_k_j)), min = 3, max = 10))
            else:
                value_i_j_t: int = int(np.clip(int(w_i_j + r*(w_i_j - w_k_j)), min = 2, max = 10))
            # Generamos la los valores de la abeja temporalmente
            temp_bee: list = [temp if dx != j else value_i_j_t for dx, temp in enumerate(guide[1].content)]
            # Evasión de exceso de peso
            while(weight(temp_bee) > 30):
                r = np.random.uniform(low = -1, high = 1)
                if j != 1 and j != 3:
                    value_i_j_t: int = int(np.clip(int(w_i_j + r*(w_i_j - w_k_j)), min = 0, max = 10))
                elif j == 1:
                    value_i_j_t: int = int(np.clip(int(w_i_j + r*(w_i_j - w_k_j)), min = 3, max = 10))
                else:
                    value_i_j_t: int = int(np.clip(int(w_i_j + r*(w_i_j - w_k_j)), min = 2, max = 10))
                temp_bee: list = [temp if dx != j else value_i_j_t for dx, temp in enumerate(guide[1].content)]
            
            print(f"{guide[1]} gains: {gain(guide[1].content)}")
            print(f"{temp_bee} gains: {gain(temp_bee)}")
            if gain(temp_bee) > gain(guide[1].content):
                self.population[guide[0]].content = temp_bee
                self.population[guide[0]].limit = 0
            else:
                self.population[guide[0]].limit += 1
            print(f"{self.population[guide[0]]}. Límite = {self.population[guide[0]].limit}. Peso = {weight(self.population[guide[0]].content)}")
            print('*'*30)
    
    def explorer_iteration(self, min: list[int], max: int)->None:
        r'''
        Función que ejecuta el ciclo correspondiente a las abejas exploradoras
        '''
        for i in range(len(self.population)):
            if self.population[i].limit >= 5:
                if gain(self.mvb) < gain(self.population[i].content):
                    self.mvb = self.population[i].content
                self.population[i] = Worker(min, max)
            else: 
                continue
        print(f"Mejor resultado histórico: {self.mvb}\nGanancia: {gain(self.mvb)}. Peso: {weight(self.mvb)}")


if __name__ == '__main__':
    colmena: Hive = Hive(40, [0,3,0,2,0,0,0], 10)
    print(colmena)
    for i in range(50):
        print("\n\n", f"Ciclo de obreras {i}".center(50, '-'))
        colmena.worker_iteration()
        print("\n\n", f"Ciclo de observadoras {i}".center(50, '-'))
        colmena.unlooker_iteration()
        colmena.explorer_iteration([0,3,0,2,0,0,0], 10)