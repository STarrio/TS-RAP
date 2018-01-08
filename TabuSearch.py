import random
import numpy as np
from scipy.stats import binom

class TabuSearch():
    """
    TabuSearch es la clase que engloba el código necesario para la resolución del problema
    
    s: número de componentes del sistema
    A_0: reliability mínimo requerido
    X_max: vector indicando el máximo Xi permitido (lista)
    J_max: vector indicando el máximo Ji permitido (lista)
    A: vector de disponibilidad de cada versión de los componente (lista de listas)
    C: vector de costes de cada versión de los componentes (lista de listas)
    W: vector de rendimientos nominales de cada versión de los componentes (lista de listas)    
    W_T: [(w_k,t_k),...] rendimiento w_k requerido durante t_k unidades de tiempo (lista de tuplas)
    q: penalización
    mnli: número máximo de iteraciones locales después de encontrar la mejor solución 
    mngi: número máximo de iteraciones globales después de encontrar la mejor solución
    discount_func: función de descuentos (función)
    """
    def __init__(self, s, A_0, X_max, J_max, A, C, W, W_T, q, mnli, mngi,discount_func=lambda c, x: 1):
        self.s = s
        self.A_0 = A_0
        self.X_max = X_max
        self.J_max = J_max
        self.A = A
        self.C = C
        self.W = W
        self.discount_func = discount_func
        self.W_T = W_T
        self.q = q
        self.mnli = mnli
        self.mngi = mngi
    
    def ts(self):
        stop_cond = self.mngi
        s0 = self.generate_s0()
        s_best = s0
        while stop_cond:
            stop_cond-=1
            s_cand = self.random_modification(s_best)
            s_sub_cand = self.ts_sub(s_cand)
            if(self.score(s_sub_cand)>self.score(s_best)):
                s_best=s_sub_cand
                stop_cond = self.mngi 
            
        return s_best
    
    def ts_sub(self, s0):
        s_best = s0
        stop_cond = self.mnli
        tabu_list = []
        tabu_list.append(s0)
        while stop_cond:
            stop_cond-=1  
            s_neighbors = self.neighborhood(s0, tabu_list)
            if(len(s_neighbors) == 0):
                continue
            s_cand = s_neighbors.pop()
            for neighbor in s_neighbors:
                if (not neighbor in tabu_list and self.score(neighbor)>self.score(s_cand)):
                    s_cand = neighbor
            if(self.score(s_cand)>self.score(s_best)):
                s_best=s_cand
                stop_cond = self.mnli
            tabu_list.append(s_cand)
            if(len(tabu_list) > self.mnli ):
                del tabu_list[0]
                          
        return s_best
    
    
    def neighborhood(self, solution, tabu_list):
        # Devuelve la lista de vecinos de solution
        def is_valid_neighbor(n, i1, i2):
            ''' Checkea si la modificación de Xi y/o de Ji de un vecino dado es válida (si estos valores se sitúan 
            entre 1 y X_max/J_max) y comprueba si no se encuentra ya en la lista tabu'''
            return (n[i1] in range(1, (self.X_max + self.J_max)[i1]+1) and
               n[i2] in range(1, (self.X_max + self.J_max)[i2]+1) and n not in tabu_list)

        def add_subtract_one_at_indexes(l,i1,i2):
            ''' Dados dos índices (i1,i2), se aplican las modificaciones (+1,-1) y (-1,+1) en los elementos
             correspondientes a las posiciones de dichos índices en la solución dada. Además, se checkea si
             dicha modificación es válida'''
            n1,n2 = l.copy(), l.copy()        
            n1[i1],n1[i2] = n1[i1]+1,n1[i2]-1
            n2[i1],n2[i2] = n2[i1]-1,n2[i2]+1            
            return [ n for n in [n1, n2] if is_valid_neighbor(n, i1, i2)]

        return [ n for i1 in range(len(solution)-1) 
                     for i2 in range(i1+1, len(solution)) 
                        for n in add_subtract_one_at_indexes(solution,i1,i2)]
    
    
    def score(self, solution):
        # Devuelve el score de solution de acuerdo con la heuristica del articulo        
        def extract_possible_m():
            def rec(index):
                if(index == self.s-1):                   
                    return list(range(1, solution[index]+1))
                else:
                    return [ [x, c] if( type(c) is not list) else [x]+c for x in range(1, solution[index]+1) for c in rec(index+1) ]
            return rec(0)        
        
        def m_performance(k_list):
            return min(k_list[i]*ij_performance(solution[self.s:][i],i) for i in range(len(k_list)))
        
        def ij_performance(version,component):            
            return self.W[component][version-1]        

        def delta_summ(k_interval,k_lists):
            m_sort = sorted(k_lists,key=lambda k: m_performance(k))
            return self.W_T[k_interval][1]*sum(delta(m) for m in m_sort if m_performance(m)>=self.W_T[k_interval][0])

        def delta(m):
            alpha = lambda k,x_i,a_ij: binom.pmf(k,x_i,a_ij)
            return np.prod([alpha(m[c],solution[:self.s][c],self.A[c][solution[self.s:][c]-1]) for c in range(len(m))])                
        
        def availability():
            possible_m = extract_possible_m()
            return ( (1/sum([ t_k for w_k, t_k in self.W_T ]))
                *sum([
                    delta_summ(k, possible_m)
                    for k, _ in enumerate(self.W_T)
                ]) )
        def cost():            
            return sum(self.discount_func(c, solution[c])*solution[c] * self.C[c][j-1] for c, j in enumerate(solution[self.s:]))
        
        return cost() + self.q*max(0, self.A_0 - availability())
    
    def generate_s0(self):
        s0 = [1,1]*self.s
        return s0
    
    def address(self, solution):
        return sum(solution)    
    
    def random_modification(self, solution):
        return solution
