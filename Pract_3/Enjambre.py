'''
Práctica 3: Optimización con enjambre - PSO
Hernández Jiménez Erick Yael
'''
import numpy as np
from numpy import ndarray
import math

class Particula():
    def __init__(self):
        # Velocidad actual de la partícula
        self.velocidad: ndarray = np.random.uniform(-5, 5, (1, 2))
        # Coordenadas de la posición actual de la partícula
        self.pos: ndarray = np.random.uniform(-5, 5, (1, 2))
        # Coordenadas de la mejor posición de la partícula
        self.pbest: ndarray = self.pos.copy()

    def __str__(self):
        return f"({self.pos[0,0]:.2f}, {self.pos[0,1]:.2f}).   \tP_best: ({self.pbest[0,0]:.2f}, {self.pbest[0,1]:.2f}).    \tVelocidad actual: ({self.velocidad[0,0]:.2f}, {self.velocidad[0,1]:.2f})."
    
    def actualizar_posicion(self) -> None:
        nueva_posicion: ndarray = np.add(self.pos, self.velocidad)
        self.pos = nueva_posicion
        return

class Enjambre(Particula):
    def __init__(self, n_poblacion: int = 20, a: float = 0.8, b_1: float = 0.7, b_2: float = 1.2, r_1: float = 0.5, r_2: float = 0.3):
        '''
        Inicializa la partícula con la opción de definir las constantes que serán útiles para calcular las velocidades
        siguientes en cada generación

        a: float. Factor de inercia 
        b_1: float. Coeficiente de aceleración/factor de aprendizaje local
        b_2: float. Coeficiente de aceleración/factor de aprendizaje global
        r_1: float. Aleatorio local
        r_2. float. Aleatorio global
        '''
        self.A: float = a
        self.B_1: float = b_1
        self.B_2: float = b_2
        self.R_1: float = r_1
        self.R_2: float = r_2
        # Lista de partículas de población
        self.poblacion: list[Particula] = []
        # Coordenadas de la mejor posición

        # Iteración para inicializar las partículas
        for _ in range(0, n_poblacion):
            self.poblacion.append(Particula())
        
        self.gbest: ndarray = self.poblacion[0].pos.copy()
        self.encontrar_mejor_global()
        return
    
    def __str__(self):
        resultado: list[str] = []
        for index, particula in enumerate(self.poblacion):
            resultado.append(f"{index}: {particula} \t {self.evaluacion(particula.pos):.2f}")
        resultado.append(f"Mejor posición: ({self.gbest[0,0]:.2f}, {self.gbest[0,1]:.2f}) evaluado con: {self.evaluacion(self.gbest):.2f}")
        return "\n".join(resultado)
    
    def evaluacion(self, pos: ndarray) -> float:
        '''
        Evalúa las coordenadas actuales en la función x^2 + y^2 + 25(sin(x) + sin(y))
        pos: arreglo [x,y]. Coordendas a evaluar
        '''
        return (pos[0,0]**2 + pos[0,1]**2 + (25 * (math.sin(pos[0,0]) + math.sin(pos[0,1]))))
    
    def actualizar_velocidades(self) -> None:
        '''
        Obtiene la velocidad en el tiempo siguiente y las actualiza para todas las partículas en la población
        '''
        v_t = lambda x_p_best, x_act, x_g_best, x_vel: self.A * x_vel + self.B_1 * self.R_1 * (x_p_best - x_act) + self.B_2 * self.R_2 * (x_g_best - x_act)
        for particula in self.poblacion:
            v_t_sig: ndarray = np.array([[v_t(particula.pbest[0,0], particula.pos[0,0], self.gbest[0,0], particula.velocidad[0,0]),
                                           v_t(particula.pbest[0,1], particula.pos[0,1], self.gbest[0,1], particula.velocidad[0,1])]], dtype=np.float64)
            particula.velocidad = v_t_sig
        return
    
    def actualizar_posiciones(self) -> None:
        for particula in self.poblacion:
            particula.actualizar_posicion()
        return
    
    def encontrar_mejor_global(self) -> None:
        for particula in self.poblacion:
            # Si las coordendas de esta partícula son mejores, se reemplazan en el mejor global
            if self.evaluacion(self.gbest) > self.evaluacion(particula.pos):
                self.gbest = particula.pos.copy()
        return
    
    def actualizar_mejor_local(self) -> None:
        for particula in self.poblacion:
            if self.evaluacion(particula.pbest) > self.evaluacion(particula.pos):
                particula.pbest = particula.pos.copy()

if __name__ == '__main__':
    enjambre: Enjambre = Enjambre()
    for i in range(0, 49):
        print("\n", '*'*30, f"\t Generación {i}\t", '*'*30)
        print(enjambre)
        enjambre.actualizar_velocidades()
        enjambre.actualizar_posiciones()
        enjambre.actualizar_mejor_local()
        enjambre.encontrar_mejor_global()

