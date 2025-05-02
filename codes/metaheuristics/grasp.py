import numpy as np  # Biblioteca principal para manipulação de arrays e cálculos
from numpy.random import randint  # Função para escolher números inteiros aleatórios
from numpy import int32 as DTYPE, inf as INF  # Importação de tipos específicos para garantir a compatibilidade e eficiência
from copy import deepcopy  # Usado para criar cópias profundas de objetos
from time import perf_counter  # Usado para medir o tempo de execução da função

def GRASP(n_voos, m_runways, r, c, p, t, local_search_heuristic, movements, local_descent_heuristic, max_iter=500, alpha=0.2):
    """
    GRASP (Greedy Randomized Adaptive Search Procedure).

    Parâmetros:
    ----------
    n_voos : int
        Número de voos.
    m_runways : int
        Número de pistas.
    r : np.ndarray
        Vetor com os tempos de liberação dos voos.
    c : np.ndarray
        Vetor com os tempos de ocupação das pistas.
    p : np.ndarray
        Penalidades por minuto de atraso dos voos.
    t : np.ndarray
        Matriz de separação entre dois voos consecutivos.
    local_search_heuristic : callable
        Função de busca local (por exemplo, VND, ILS, etc.).
    movements : iterable(callable)
        Funções que definem os movimentos de vizinhança.
    local_descent_heuristic : callable
        Função de busca local utilizada durante o GRASP.
    max_iter : int, opcional
        Número máximo de iterações do GRASP (padrão é 500).
    alpha : float, opcional
        Parâmetro que controla a restrição da Lista de Candidatos Restritos (LCR).

    Retorna:
    -------
    best_solution : list(np.ndarray)
        Melhor solução encontrada.
    best_cost : float
        Custo da melhor solução encontrada.
    max_iter : int
        Número máximo de iterações.
    elapsed_time : float
        Tempo de execução total da função (em segundos).
    """
    dt = perf_counter()  # Inicia o cronômetro para medir o tempo de execução da função
    
    # Função para calcular o custo total de uma solução
    def compute_total_cost(solution, r, c, p, t):
        # Função auxiliar para calcular os horários de decolagem e chegada dos voos
        def _runway_schedule(runway, r, c, t):
            n = runway.size  # Número de voos na pista
            start = np.empty(n, dtype=r.dtype)  # Array para armazenar os horários de início dos voos
            start[0] = r[runway[0]]  # O primeiro voo começa no seu horário de liberação
            for i in range(1, n):
                prev = runway[i-1]  # Voo anterior
                cur  = runway[i]    # Voo atual
                # O horário de início do voo atual depende do voo anterior, do tempo de ocupação e da separação
                start[i] = max(r[cur], start[i-1] + c[prev] + t[prev, cur])
            return start

        # Função principal para calcular o custo total da solução
        def total_cost(solution, r, c, p, t):
            total = 0  # Inicializa o custo total
            for runway in solution:
                if(runway.size == 0):
                    continue  # Se a pista estiver vazia, não há custo
                start = _runway_schedule(runway, r, c, t)  # Calcula os horários de início dos voos na pista
                for idx in range(runway.size):
                    delay = start[idx] - r[runway[idx]]  # Atraso de cada voo em relação ao seu horário de liberação
                    if(delay > 0):
                        total += delay * p[runway[idx]]  # Penalidade por atraso
            return total
        
        return total_cost(solution, r, c, p, t)

    # Função para gerar uma solução inicial usando o GRASP (Construção da solução)
    def construction(m_runways, r, c, p, t, alpha):
        sorted_idx = np.argsort(r)  # Ordena os voos por horário de liberação
        solution = [np.array([], dtype=DTYPE) for _ in range(m_runways)]  # Pistas começam vazias
        
        # Criar a Lista de Candidatos Restritos (LCR)
        remaining = list(sorted_idx)  # Lista dos voos restantes
        while(len(remaining)):
            # Cálculo dos benefícios de cada voo
            benefits = []
            for i in remaining:
                costs = []
                for j in range(m_runways):
                    # Avalia o custo da inserção do voo i na pista j
                    new_solution = [np.copy(solution[k]) if k != j else np.concatenate([solution[k], np.array([i], dtype=DTYPE)]) for k in range(m_runways)]
                    cost = compute_total_cost(new_solution, r, c, p, t)
                    costs.append(cost)
                min_cost = min(costs)
                benefits.append(min_cost)
            
            # Definir a Lista de Candidatos Restritos (RCL) com base no parâmetro alpha
            g_min = min(benefits)
            g_max = max(benefits)
            RCL = [remaining[i] for i in range(len(benefits)) if benefits[i] <= g_min + alpha * (g_max - g_min)]
            
            # Seleção aleatória de um voo da RCL
            selected_i = randint(len(RCL))
            selected = RCL[selected_i]
            
            # Remove o voo selecionado da lista remaining
            remaining.remove(selected)
            
            # Aloca o voo em uma pista (minimiza custo)
            best_runway = min(range(m_runways), key=lambda j: compute_total_cost([np.copy(solution[k]) if k != j else np.concatenate([solution[k], np.array([selected])]) for k in range(m_runways)], r, c, p, t))
            
            # Atualiza a solução e o custo da pista
            solution[best_runway] = np.concatenate([solution[best_runway], np.array([selected])])

        return solution  # Retorna a solução construída

    # GRASP - Inicia a busca com um número máximo de iterações
    best_solution = None
    best_cost = INF  # Inicializa o melhor custo com infinito

    # Laço para executar o GRASP por um número máximo de iterações
    for _ in range(max_iter):
        # Construir uma solução inicial com GRASP
        solution = construction(m_runways, r, c, p, t, alpha)
        
        # Aplica a busca local na solução construída
        local_solution, local_cost, *_ = local_search_heuristic(solution, r, c, p, t, compute_total_cost, movements, local_descent_heuristic)
        
        # Se a nova solução for melhor, atualiza a melhor solução
        if(local_cost < best_cost):
            best_solution = deepcopy(local_solution)
            best_cost = local_cost
    
    # Retorna a melhor solução, o melhor custo, o número de iterações e o tempo de execução
    return best_solution, best_cost, max_iter, perf_counter()-dt