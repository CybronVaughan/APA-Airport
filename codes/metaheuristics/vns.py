from copy import deepcopy  # Usado para criar uma cópia profunda de objetos
from time import perf_counter  # Usado para medir o tempo de execução da função

def VNS(initial_solution, r, c, p, t, cost_function, neighborhood_functions, max_no_improve=500, k_max=3):
    """
    Variable Neighborhood Search (VNS).

    Parâmetros:
    ----------
    initial_solution : list(np.ndarray)
        Sequência de voos por pista.
    r, c, p : np.ndarray
        Vetores de release‑time, tempo de ocupação de pista e penalidade por minuto de atraso, respectivamente.
    t : np.ndarray
        Separações obrigatórias entre voos.
    max_no_improve : int, opcional
        Número máximo de iterações sem melhoria para parar a busca (padrão é 500).
    k_max : int, opcional
        Número máximo de vizinhanças a serem exploradas (padrão é 3).

    Retorna:
    -------
    best_solution : list(np.ndarray)
        Melhor solução encontrada.
    best_cost : float
        Custo da melhor solução encontrada.
    j : int
        Número de iterações realizadas.
    elapsed_time : float
        O tempo total de execução da função (em segundos).
    """
    dt = perf_counter()  # Inicia o cronômetro para medir o tempo de execução da função
    
    # Inicializa a solução e o custo da melhor solução
    current_solution = deepcopy(initial_solution)
    best_solution = deepcopy(initial_solution)
    best_cost = cost_function(best_solution, r, c, p, t)  # Calcula o custo da solução inicial
    no_improve = 0  # Contador de iterações sem melhoria

    j = 0  # Contador de iterações realizadas
    while(no_improve < max_no_improve):
        j += 1  # Incrementa o contador de iterações

        # Laço sobre as vizinhanças, tentando melhorar a solução
        for k in range(k_max):
            candidate = neighborhood_functions[k](current_solution)  # Aplica a vizinhança k
            candidate_cost = cost_function(candidate, r, c, p, t)  # Calcula o custo da solução gerada

            # Se a nova solução for melhor, atualiza a melhor solução encontrada
            if(candidate_cost < best_cost):
                best_solution = deepcopy(candidate)
                best_cost = candidate_cost  # Atualiza o melhor custo
                no_improve = 0  # Reinicia o contador de iterações sem melhoria
                break  # Sai do laço de vizinhanças, já encontrou uma solução melhor
        else:
            no_improve += 1  # Caso não tenha encontrado melhoria, incrementa o contador de iterações sem melhoria

        # Atualiza a solução atual com a melhor solução encontrada até o momento
        current_solution = deepcopy(best_solution)

    # Retorna a melhor solução, o custo da melhor solução, o número de iterações realizadas e o tempo de execução
    return best_solution, best_cost, j, perf_counter()-dt