import numpy as np  # Biblioteca principal para manipulação de arrays e cálculos
from numpy.random import choice  # Usado para selecionar elementos aleatoriamente
from copy import deepcopy  # Usado para fazer uma cópia profunda de objetos
from time import perf_counter  # Usado para medir o tempo de execução da função

def ILS(initial_solution, r, c, p, t, cost_function, movements, local_descent_heuristic, max_no_improve=500, perturb_strength=2):
    """
    Iterated Local Search (ILS).
    1. Parte de uma solução inicial
    2. Aplica pequenas perturbações (shake)
    3. Desce a solução com uma busca local (VND ou outra heurística)
    4. Guarda a melhor solução e repete até max_no_improve iterações sem melhora consecutiva.

    Parâmetros:
    ----------
    initial_solution : list(np.ndarray)
        Sequência de voos por pista.
    r, c, p : np.ndarray
        Vetores de release‑time, tempo de ocupação de pista e penalidade por minuto de atraso, respectivamente.
    t : np.ndarray
        Matriz de separação entre voos consecutivos.
    max_no_improve : int
        Número máximo de iterações sem melhoria para parar a busca.
    perturb_strength : int
        Quantidade de pequenos movimentos randomizados aplicados no "shake".

    Retorna:
    -------
    best_solution : list(np.ndarray)
        Melhor solução encontrada.
    best_cost : float
        Custo da melhor solução encontrada.
    """
    dt = perf_counter()  # Inicia o cronômetro para medir o tempo de execução da função

    # Inicialização
    best_sol = deepcopy(initial_solution)  # Faz uma cópia profunda da solução inicial para garantir que não seja alterada
    best_cost = cost_function(best_sol, r, c, p, t)  # Calcula o custo da solução inicial
    current_sol = deepcopy(best_sol)  # Cria uma cópia da solução inicial para ser manipulada durante as iterações
    no_improve = 0  # Inicializa o contador de iterações sem melhoria

    # Variável para contar o número de iterações realizadas
    j = 0
    
    # Laço principal do ILS: tenta melhorar a solução até o número máximo de iterações sem melhoria
    while(no_improve < max_no_improve):
        j += 1  # Incrementa o contador de iterações

        # ---------- 1) Perturbação (Shake) ----------
        shaken = deepcopy(current_sol)  # Faz uma cópia da solução atual para ser perturbada

        for _ in range(perturb_strength):
            shaken = choice(movements)(shaken)  # Aplica uma perturbação aleatória, selecionando um movimento da lista

        # ---------- 2) Descida Local (VND) ----------
        # Aplica uma busca local (pode ser VND ou outra heurística)
        local_sol, local_cost, *_ = local_descent_heuristic(shaken, r, c, p, t, cost_function, movements)

        # ---------- 3) Aceitação ---------------
        # Se a solução local for melhor que a melhor encontrada até agora, atualiza a melhor solução
        if(local_cost < best_cost):
            best_sol = deepcopy(local_sol)  # Atualiza a melhor solução encontrada
            best_cost = local_cost  # Atualiza o custo da melhor solução
            current_sol = deepcopy(local_sol)  # A solução atual é a melhor solução encontrada
            no_improve = 0  # Zera o contador de iterações sem melhoria
        else:
            # Caso contrário, continua explorando a solução atual
            current_sol = deepcopy(local_sol)  # Atualiza a solução atual com a solução local
            no_improve += 1  # Incrementa o contador de iterações sem melhoria

    return best_sol, best_cost, j, perf_counter()-dt  # Retorna a melhor solução, o melhor custo, o número de iterações e o tempo de execução