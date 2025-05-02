from copy import deepcopy  # Usado para fazer uma cópia profunda de objetos
from time import perf_counter  # Usado para medir o tempo de execução da função

def VND(initial_solution, r, c, p, t, cost_function, neighborhood_functions, max_iter=500):
    """
    Variable‑Neighbourhood Descent (VND).

    Parâmetros:
    ----------
    initial_solution : list(np.ndarray)
        Sequência de voos por pista.
    r, c, p : np.ndarray
        Vetores de release‑time, tempo de ocupação da pista e penalidade por minuto, respectivamente.
    t : np.ndarray
        Matriz de separação entre dois voos consecutivos.
    neighborhood_functions : iterable(callable)
        Operadores de vizinhança. Devem receber e retornar uma solução no mesmo formato.
    max_iter : int, opcional
        Número máximo de iterações sem encontrar uma melhoria (padrão é 500).

    Retorna:
    -------
    best_solution : list(np.ndarray)
        A melhor solução encontrada.
    best_cost : float
        O custo da melhor solução encontrada.
    j : int
        Número de iterações realizadas.
    elapsed_time : float
        O tempo total de execução da função (em segundos).
    """
    dt = perf_counter()  # Inicia o cronômetro para medir o tempo de execução da função
    
    current = deepcopy(initial_solution)  # Cria uma cópia da solução inicial para não modificar a original
    best_cost = cost_function(current, r, c, p, t)  # Calcula o custo da solução inicial

    j = 0  # Contador de iterações
    k = 0  # Índice da vizinhança atual (inicia com a primeira vizinhança)

    # Loop principal do VND: tenta melhorar a solução enquanto houver vizinhanças e iterações
    while(k < len(neighborhood_functions) and j <= max_iter):
        candidate = neighborhood_functions[k](current)  # Aplica o operador de vizinhança k na solução atual
        cand_cost = cost_function(candidate, r, c, p, t)  # Calcula o custo da solução gerada pela vizinhança

        # Se encontrou uma solução melhor (custo menor), atualiza a solução e reseta o índice da vizinhança
        if(cand_cost < best_cost):
            current = deepcopy(candidate)  # Atualiza a solução com a nova melhor solução
            best_cost = cand_cost  # Atualiza o melhor custo
            k = 0  # Volta para a primeira vizinhança
        else:
            k += 1  # Caso contrário, tenta a próxima vizinhança
        
        j += 1  # Incrementa o contador de iterações

    elapsed_time = perf_counter() - dt  # Calcula o tempo total de execução da função
    return current, best_cost, j, elapsed_time  # Retorna a melhor solução, o melhor custo, o número de iterações e o tempo de execução