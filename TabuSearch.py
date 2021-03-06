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
        s_cand = s0
        while stop_cond > 0:            
            s_cand = self.random_modification(s_cand)
            if(s_cand == []):
                break            
            s_sub_cand = self.ts_sub(s_cand)
            if(self.score(s_sub_cand)<self.score(s_best)):
                s_best=s_sub_cand
                print(s_best,self.address(s_best), self.score(s_best))
                stop_cond = self.mngi 
            else:
                stop_cond -= 1
        return s_best, self.score(s_best)
    
    def ts_sub(self, s0):
        s_best = s0
        s_best_score = self.score(s_best)
        #print(s_best)
        stop_cond = self.mnli
        tabu_list = []
        tabu_list.append(s0)
        s_cand = s0
        while stop_cond:
            stop_cond-=1  
            s_neighbors = self.neighborhood(s_cand, tabu_list)
            if(s_best != self.generate_s0()):
                s_neighbors = list(filter(lambda neighbor: np.mean([abs(x-y) for (x,y) in zip(neighbor,s_best)])<=1, s_neighbors)) 
                #print(len(s_neighbors))
            if(len(s_neighbors) == 0):
                continue
            s_cand = s_neighbors.pop()
            s_cand_score = self.score(s_cand)            
            for neighbor in s_neighbors:
                neighbor_score = self.score(neighbor)
                if (not neighbor in tabu_list and neighbor_score<s_cand_score):
                    s_cand = neighbor
                    s_cand_score = neighbor_score
            if(s_cand_score<s_best_score):
                s_best=s_cand
                s_best_score = s_cand_score
                #print(s_best)
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
        def alpha(c, w_k):            
            w_c = self.W[c][solution[self.s:][c]-1] #w_i_j
            x = solution[c]
            a = self.A[c][solution[self.s:][c]-1]
            if w_k <= w_c:
                return 1-(1-a)**x
            else:
                return next(( 1 - binom.pmf(k-1, x, a) for k in range(2, x+1)
                              if (k-1)*w_c < w_k <= k*w_c), 0)
        def delta_summ(k, w_k):
            return np.prod([ alpha(c, w_k) for c in range(self.s)])
        
        def availability():
            return ( (1/sum([ t_k for w_k, t_k in self.W_T ]))
                *sum([
                    self.W_T[k][1]*delta_summ(k, w_k[0])
                    for k, w_k in enumerate(self.W_T)
                        ]))
        
        def cost():            
            return sum(self.discount_func(c, solution[c])*solution[c] * self.C[c][j-1] for c, j in enumerate(solution[self.s:]))
        
        return cost() + self.q*max(0, self.A_0 - availability())
    
    def generate_s0(self):
        s0 = [1,1]*self.s
        return s0
    
    def address(self, solution):
        return sum(solution)    
    
    def random_modification3(self,solution):
        def is_valid_swap(n, i1, i2):
            return (n[i1] in range(1, (self.X_max + self.J_max)[i2]+1)) and (n[i2] in range(1, (self.X_max + self.J_max)[i1]+1)) 
                
        def is_valid(n, i):
            return (n[i] in range(1, (self.X_max + self.J_max)[i]+1))
        
        if self.address(solution) == self.address(self.X_max+self.J_max)-1:
            return self.random_modification2(solution)
        
        I_n = list(range(2*(self.s)))
        I_n2 = list(range(2*(self.s)))
        while len(I_n)>0:
            s_n = solution.copy()
            i_n = random.choice(I_n) 
            I_n.remove(i_n)
            s_n[i_n] += 1
            if(is_valid(s_n,i_n)):
                swap = next( ( i2 for i2 in I_n2 if i_n != i2 and is_valid_swap(s_n, i_n, i2)), None )
                if swap != None:
                    s_n[i_n], s_n[swap] = s_n[swap], s_n[i_n]
                    return s_n
        return []
            
    
    def random_modification2(self,solution):
        def is_valid(n, i):
            return (n[i] in range(1, (self.X_max + self.J_max)[i]+1))
        
        I_n = list(range(2*(self.s)))
        while len(I_n)>0:
            s_n = solution.copy()
            i_n = random.choice(I_n)
            I_n.remove(i_n)
            s_n[i_n]+=1
            
            if(is_valid(s_n,i_n)):
                return s_n
        return []
        
    def random_modification(self, solution):
        return [ random.randint(1, (self.X_max+self.J_max)[c]) for c in range(self.s*2) ]
