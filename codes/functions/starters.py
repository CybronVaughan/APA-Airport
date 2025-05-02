import numpy as np
from numpy import int32 as DTYPE  # Definindo o tipo de dado (inteiro de 32 bits) para garantir compatibilidade
from numba import njit  # Importando o decorator njit do Numba para otimização de desempenho
from time import perf_counter  # Para medir o tempo de execução

# Função para ler uma instância do problema a partir de um arquivo
def read_instance(filename):
    # Lê as linhas do arquivo e remove espaços em branco extras
    lines = [line.strip() for line in open(filename, 'r') if line.strip()]

    try:
        # Lê o número de voos e o número de pistas (convertendo para inteiro)
        n_voos = int(lines[0])
        m_runways = int(lines[1])
    except ValueError as e:
        raise ValueError('Erro ao converter número de voos ou número de pistas para inteiro.') from e

    try:
        # Lê os arrays 'r', 'c' e 'p' a partir das linhas do arquivo, convertendo-os para arrays NumPy de inteiros
        r = np.array([int(v) for v in lines[2].split()], dtype=DTYPE)
        c = np.array([int(v) for v in lines[3].split()], dtype=DTYPE)
        p = np.array([int(v) for v in lines[4].split()], dtype=DTYPE)
    except ValueError as e:
        raise ValueError("Erro ao converter os arrays 'r', 'c' ou 'p' para inteiros.") from e

    # Lê a matriz de separação entre voos 't'
    t = np.array([[int(v) for v in l.split() if v] for l in lines[5:]], dtype=DTYPE)
    
    del lines  # Limpa a variável 'lines' da memória
    return n_voos, m_runways, r, c, p, t  # Retorna os valores lidos do arquivo

# Função para calcular os horários de decolagem e chegada para cada pista
@njit(inline='always')  # 'njit' do Numba para otimização do desempenho (compilação Just-In-Time)
def _runway_schedule(runway, r, c, t):
    n = runway.size  # Tamanho da pista (número de voos alocados nela)
    start = np.empty(n, dtype=r.dtype)  # Vetor para armazenar os horários de início

    # O primeiro voo começa no seu horário de liberação
    start[0] = r[runway[0]]
    
    # Calcula o horário de início de cada voo em sequência, levando em conta o tempo de ocupação e separação
    for i in range(1, n):
        prev = runway[i-1]  # Índice do voo anterior
        cur  = runway[i]    # Índice do voo atual
        # O horário de início do voo atual depende do horário do voo anterior, do tempo de ocupação e do tempo de separação
        start[i] = max(r[cur], start[i-1] + c[prev] + t[prev, cur])
    
    return start

# Função para calcular o custo total de uma solução (penalidades por atraso)
@njit  # Aplicando 'njit' para otimizar a função
def compute_total_cost(solution, r, c, p, t):
    total = 0  # Inicializa o custo total como zero
    
    # Para cada pista (onde há uma lista de voos alocados)
    for runway in solution:
        if(runway.size == 0):  # Se a pista está vazia, não há custo
            continue
        
        # Calcula os horários de início dos voos na pista
        start = _runway_schedule(runway, r, c, t)
        
        # Calcula o atraso de cada voo, se houver
        for idx in range(runway.size):
            delay = start[idx] - r[runway[idx]]  # Atraso em relação ao tempo de liberação do voo
            if(delay > 0):  # Se houver atraso, adiciona a penalidade ao custo total
                total += delay * p[runway[idx]]  # Penalidade por atraso = atraso * penalidade do voo

    return total  # Retorna o custo total (penalidade por atraso)

# Função para gerar uma solução inicial balanceada (distribui os voos nas pistas de forma balanceada)
def initial_solution(n_voos, m_runways, r):
    dt = perf_counter()  # Inicia o cronômetro para medir o tempo de execução da função
    
    # Verifica se o número de voos e o tamanho do vetor 'r' são consistentes
    assert len(r) == n_voos, 'Tamanho de r incoerente com n_voos'

    # Ordena os voos com base no tempo de liberação (r) e divide os voos balanceadamente entre as pistas
    sorted_idx = np.argsort(r)  # Ordena os índices dos voos pelo tempo de liberação
    solution = [np.array(sorted_idx[i::m_runways].tolist()) for i in range(m_runways)]  # Distribui os voos entre as pistas
    
    # Retorna a solução e o tempo de execução da função
    return solution, perf_counter()-dt