import numpy as np

class Nodo():
    r'''
    ### Nodo
    **Clase que contiene al nombre del nodo**
    '''
    def __init__(self, nombre: str):
        r'''
        - nombre: str = cadena con el nombre del nodo
        '''
        self.nombre = nombre

    def __eq__(self, otro):
        r'''
        - otro: Nodo = el `otro` Nodo será igual si tienen el mismo nombre
        '''
        if isinstance(otro, Nodo):
            return self.nombre == otro.nombre
        return False

    def __hash__(self):
        return hash(self.nombre)

    def __str__(self):
        return self.nombre
    
    def __repr__(self):
        return self.nombre

class Camino():
    r'''
    ### Camino
    **Clase para almacenar los datos de distancia y feromonas asociados al mismo**
    '''
    def __init__(self, origen: Nodo, destino: Nodo, distancia: float, feromona: float = 0.1):
        r'''
        - origen: Nodo = Nodo desde el que se asocia el camino
        - destino: Nodo = Nodo hacia donde se asocia el camino
        - distancia: float = Valor numérico flotante que describe la distancia entre los dos nodos
        - feromona: float = Valor numérico flotante que describe la feromona asociada al camino
        '''
        self.__A__: float = 1.5
        self.__B__: float = 0.8
        self.origen: Nodo = origen
        self.destino: Nodo = destino
        self.distancia: float = distancia
        self.feromona: float = feromona
        self.visibilidad: float = 1/distancia
        self._producto_: float = (self.visibilidad ** self.__B__) * (self.feromona ** self.__A__)
    def __eq__(self, otro):
        r'''
        - otro: Camino = el `otro` será igual si los nodos asociados son los mismos, no necesariamente asociados igualmente como origen o destino
        '''
        if isinstance(otro, Camino):
            return (self.origen == otro.origen and self.destino == otro.destino) or (self.origen == otro.destino and self.destino == otro.origen)
        return False

    def __hash__(self):
        return hash((min(self.origen, self.destino, key=str), max(self.origen, self.destino, key=str)))


    def __str__(self, opcion: str = 'short'):
        r'''
        **Opciones**
        - 'full': todos los detalles del camino
        - 'short': solo los nodos del camino
        '''
        if opcion == 'full':
            return f"{self.origen} -- {self.destino}. Distancia:{self.distancia:.2f}. Feromona:{self.feromona:.2f}. Visibilidad:{self.visibilidad:.2f}"
        else:
            return f"{self.origen} -- {self.destino}"
    
    def __repr__(self, opcion: str = 'short'):
        r'''
        **Opciones**
        - 'full': todos los detalles del camino
        - 'short': solo los nodos del camino
        '''
        if opcion == 'full':
            return f"{self.origen} - {self.destino}. Distancia:{self.distancia:.2f}. Feromona:{self.feromona:.2f}. Visibilidad:{self.visibilidad:.2f}"
        else:
            return f"{self.origen} - {self.destino}"

class Mapa():
    r'''
    ### Mapa
    **Clase que almacena los caminos que tendrá el mapa**
    '''
    def __init__(self, caminos: list[list[str]]):
        r'''
        - caminos: list[list[str]]: Lista de lista de cadenas que siguen el formato ["Origen", "Distancia", "Destino"], donde Origen y Destino son el nombre de los nodos correspondientes y Distancia la cadena con el número que describe el atributo
        
        '''
        self.contenido: set[Camino] = set()
        for camino in caminos:
            origen = Nodo(camino[0])
            destino = Nodo(camino[2])
            nuevo_camino = Camino(origen, destino, float(camino[1]))
            self.contenido.add(nuevo_camino)
        self.nodos: set[Nodo] = set()
        for camino in self.contenido:
            if camino.origen not in self.nodos:
                self.nodos.add(camino.origen)
            if camino.destino not in self.nodos:
                self.nodos.add(camino.destino)
        
    def __str__(self, opcion: str = 'd'):
        r'''
        Todas las impresiones implican mostrar el nombre de los nodos. Las opciones indican qué propiedad mostrar entre los nodos.
        **Opciones**:
        - d: Para imprimir distancias
        - v: Para imprimir el inverso de la visibilidad
        - f: Para imprimir las feromonas
        '''
        resultado: str = ""
        if opcion == 'f':
            for camino in self.contenido:
                resultado+= f"{camino.origen} --- {camino.feromona:.2f} --- {camino.destino}\n"
        elif opcion == 'v':
            for camino in self.contenido:
                resultado+= f"{camino.origen} --- {camino.visibilidad:.2f} --- {camino.destino}\n"
        else:
            for camino in self.contenido:
                resultado+= f"{camino.origen} --- {camino.distancia:.2f} --- {camino.destino}\n"
        return resultado

    def caminos_disponibles(self, origen: str) -> list[Camino]:
        r'''
        ### caminos_disponibles
        Con el nombre del nodo indicado, regresa la lista de caminos que lo conectan
        '''
        resultado: list[Camino] = []
        for caminos in self.contenido:
            if origen == caminos.origen.nombre or origen == caminos.destino.nombre:
                resultado.append(caminos)
        return resultado
    
class Hormiga():
    r'''
    ### Hormiga
    **Clase que abstrae a la hormiga, su recorrido y calcula individualmente a qué nodo moverse**
    '''
    def __init__(self, posicion: str, recorrido: list[Camino]):
        r'''
        - posicion: str = Almacena el nombre de la posición en la que está
        - recorrido: list[str] = Lista que almacena el nombre de los nodos que ha visitado
        '''
        self.posicion: str = posicion
        self.recorrido: list[Camino] = []
        self.origen: str = posicion
        self.nodos_visitados: list[Nodo] = []
        self.costo_camino: float = 0

    def __str__(self):
        return f"{self.posicion}. Recorrido:\n {self.recorrido}\n Costo del recorrido: {self.costo_camino}"
    
    def agregar_nodo_recorrido(self, camino: Camino) -> None:
        r'''
        ### agregar_nodo_recorrido
        Recibe un `Camino` y maneja la adición correcta de los nodos del camino a los ya visitados por la hormiga, así como la actualización del costo del movimiento
        '''
        if camino.origen not in self.nodos_visitados:
            self.nodos_visitados.append(camino.origen)
        if camino.destino not in self.nodos_visitados:
            self.nodos_visitados.append(camino.destino)
        self.recorrido.append(camino)
        self.costo_camino+=camino.distancia
        
class Colmena():
    r'''
    **Clase que almacena la población de hormigas y el grafo o mapa donde se desempeñan**
    '''
    def __init__(self, mapa: Mapa):
        r'''
        - mapa: Mapa = Grafo sobre el que se desempeña colmena
        '''
        self.__Q__:int = 1
        self.__RHO__:float = 0.5
        self.mapa: Mapa = mapa
        self.poblacion: list[Hormiga] = []
        for nodo in self.mapa.nodos:
            nueva_hormiga: Hormiga = Hormiga(nodo.nombre, [])
            nueva_hormiga.nodos_visitados.append(nodo)
            self.poblacion.append(nueva_hormiga)
    
    def __str__(self):
        resultado: str = ""
        for i, hormiga in enumerate(self.poblacion):
            resultado += f"Hormiga {i}: {hormiga.posicion} \n"
        return resultado
    
    def avanzar(self) -> None:
        r'''
        ### avanzar
        Hace "caminar" a las hormigas por todo el mapa en un camino cerrado desde su origen sin repetir nodos visitados
        '''
        #print(" Nueva generación ".center(70, '-'))
        for hormiga in self.poblacion:
            # Encuentra los caminos disponibles
            caminos_disponibles: list[Camino] = self.mapa.caminos_disponibles(hormiga.posicion)
            #print(caminos_disponibles)
            # Encuentra los caminos que NO ha visitado ya
            caminos_siguientes: list[Camino] = []
            for camino in caminos_disponibles:
                #print(camino not in hormiga.recorrido)
                if camino.origen not in hormiga.nodos_visitados or camino.destino not in hormiga.nodos_visitados:
                    caminos_siguientes.append(camino)
                    #print(caminos_siguientes, "\n\n")
            #print(f"Hormiga: {hormiga}\n")
            #for camino in caminos_siguientes:
            #    print(camino)
            if caminos_siguientes:
                # Calcula la suma de todos los productos de visibilidad por las feromonas correspondientes
                suma: float = 0
                for camino in caminos_siguientes:
                    suma += camino._producto_
                #print(suma, "\n")
                # Calcula la probabilidad de elegir cada nodo
                probabilidades: list[float] = []
                for camino in caminos_siguientes:
                    probabilidades.append(camino._producto_ / suma)
                #print(probabilidades, "\n")
                # RULETA
                # Modifica la lista de probabilidades para que corresponda a las frecuencias acumuladas
                for i in range(len(probabilidades)-1):
                    probabilidades[i+1] = probabilidades[i+1] + probabilidades[i]
                #print(probabilidades, "\n")
                # Obtiene un número al azar entre 0 y 1
                random: float = np.random.random()
                #print(f"Random = {random}")
                camino_elegido: Camino = caminos_siguientes[0]
                for i, probabilidad in enumerate(probabilidades):
                    if random < probabilidad:
                        break
                    else:
                        camino_elegido = caminos_siguientes[i+1]
                #print(camino_elegido, "\n")
                # Agrega el siguiente nodo al recorrido
                hormiga.agregar_nodo_recorrido(camino_elegido)
                if hormiga.posicion == camino_elegido.origen.nombre:
                    hormiga.posicion = camino_elegido.destino.nombre
                elif hormiga.posicion == camino_elegido.destino.nombre:
                    hormiga.posicion = camino_elegido.origen.nombre
                #print(hormiga.nodos_visitados)
                #print(hormiga)
            else:
                # Movemos a la hormiga al punto inicial
                siguiente_camino: Camino
                for camino in self.mapa.caminos_disponibles(hormiga.posicion):
                    if hormiga.nodos_visitados[0] in [camino.origen, camino.destino]:
                        siguiente_camino = camino
                        break
                hormiga.agregar_nodo_recorrido(siguiente_camino)
                hormiga.posicion = hormiga.nodos_visitados[0].nombre
                print(hormiga)
            #print('*'*70)

    def actualizar_feromonas(self) -> None:
        r'''
        ### actualizar_feromonas
        Maneja la actualización de las feromonas en el mapa
        '''
        #print(self.mapa.__str__("f"))
        # ACTUALIZACIÓN DE FEROMÓNAS
        for caminos in self.mapa.contenido:
            # Actualizamos todos los caminos con la erosión inicial
            caminos.feromona *= (1 - self.__RHO__)
            for hormiga in self.poblacion:
                for paso in hormiga.recorrido:
                    if paso == caminos:
                        caminos.feromona += self.__Q__/hormiga.costo_camino
        print(self.mapa.__str__("f"))

    def iteracion(self) -> None:
        r'''
        ### iteracion
        Maneja el algortimo completo de cada iteración y el reinicio de los datos guardados en las hormigas sobre el recorrido realizado
        '''
        for i, hormiga in enumerate(self.poblacion):
            print(f"Hormiga: {i}\nPosición actual: {hormiga}")
        for _ in range(len(self.mapa.nodos)):
            self.avanzar()
        self.actualizar_feromonas()
        for hormiga in self.poblacion:
            hormiga.recorrido = []
            hormiga.costo_camino = 0
            hormiga.nodos_visitados = hormiga.nodos_visitados[:1]
            #print(hormiga)
            #print(hormiga.nodos_visitados)
        
if __name__ == '__main__':
    transiciones: list[list[str]] = [
        ["2", "6", "1"],
        ["3", "9", "1"],
        ["4", "17", "1"],
        ["5", "13", "1"],
        ["6", "21", "1"],
        ["3", "19", "2"],
        ["4", "21", "2"],
        ["5", "12", "2"],
        ["6", "18", "2"],
        ["4", "20", "3"],
        ["5", "23", "3"],
        ["6", "11", "3"],
        ["5", "15", "4"],
        ["6", "10", "4"],
        ["6", "21", "5"],
        ["6", "21", "5"]
    ]
    grafo: Mapa = Mapa(transiciones)
    principal: Colmena = Colmena(grafo)
    for i in range(50):
        print("\n\n", f" Generación {i} ".center(70, '-'))
        principal.iteracion() 