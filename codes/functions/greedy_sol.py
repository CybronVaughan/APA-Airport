from numba import njit  # Importa o decorator 'njit' do Numba para otimizar funções com compilação Just-In-Time
from numba.typed import List  # Permite o uso de listas mutáveis no Numba
import numpy as np  # Biblioteca principal para manipulação de arrays e cálculos
from numpy import int32 as DTYPE  # Define o tipo de dado 'int32' para uso nos arrays
from time import perf_counter  # Usado para medir o tempo de execução das funções

# Função otimizada para gerar a solução inicial de forma gulosa (sem mutabilidade fora do JIT)
@njit  # 'njit' do Numba otimiza esta função para maior desempenho
def greedy_initial_solution_nb(m_runways, r, t):
    order = np.argsort(r)  # Ordena os voos por seus tempos de liberação
    runways = List()  # Lista mutável de pistas (reconhecida pelo JIT do Numba)
    
    # Inicializa as pistas, cada uma começa vazia
    for _ in range(m_runways):
        runways.append(List.empty_list(np.int32))  # Cada pista começa com uma lista vazia

    # Inicializa variáveis para controlar o último voo em cada pista
    last_finish = np.zeros(m_runways, dtype=r.dtype)  # Guarda o tempo de término do último voo de cada pista
    last_flight = np.full(m_runways, -1, dtype=DTYPE)  # Indica o último voo alocado em cada pista

    # Itera sobre os voos ordenados por tempo de liberação
    for v in order:
        r_v = r[v]  # Tempo de liberação do voo atual
        sep = np.where(last_flight >= 0, t[last_flight, v], 0)  # Tempo de separação entre o voo atual e o último alocado
        cand = np.maximum(r_v, last_finish + sep)  # Calcula o melhor tempo de início possível para o voo atual

        best = cand.argmin()  # Encontra a pista com o menor tempo de início possível
        bests = cand[best]  # O melhor tempo de início (minimizado)

        # Aloca o voo à pista selecionada
        runways[best].append(v)
        last_finish[best] = bests  # Atualiza o tempo de término da pista
        last_flight[best] = v  # Atualiza o último voo alocado na pista

    return runways  # Retorna as pistas com os voos alocados

# Função principal que chama a função otimizada e converte a solução para o formato adequado
def greedy_initial_solution(n_voos, m_runways, r, t):
    dt = perf_counter()  # Começa a contagem do tempo de execução
    
    # Valida o tamanho de 'r' para garantir que corresponda ao número de voos
    assert len(r) == n_voos, 'Tamanho de r incoerente com n_voos'
    
    # Chama a função otimizada para gerar a solução inicial
    raw = greedy_initial_solution_nb(m_runways, r, t)
    
    # Converte as listas mutáveis em arrays NumPy, para melhor manipulação fora do JIT
    return [np.asarray(raw[i], dtype=DTYPE) for i in range(m_runways)], perf_counter()-dt  # Retorna a solução e o tempo de execução