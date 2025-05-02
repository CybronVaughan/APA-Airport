import numpy as np  # Biblioteca principal para manipulação de arrays e cálculos
from numpy.random import choice, randint  # Funções para seleção aleatória e escolha de índices

# Função para realizar a troca de dois voos aleatoriamente dentro da mesma pista
def neighbor_swap_in_runway(sol):
    # Seleciona pistas com 2 ou mais voos
    cand = [i for i, rw in enumerate(sol) if len(rw) >= 2]
    if(not cand):
        return sol  # Se não houver pistas com 2 ou mais voos, retorna a solução original

    k = choice(cand)  # Escolhe aleatoriamente uma pista que tenha 2 ou mais voos
    rw_in = sol[k]  # Pista selecionada

    # Escolhe aleatoriamente dois voos dentro da pista (sem reposição)
    i, j = choice(len(rw_in), 2, replace=False)  # Escolhe 2 índices aleatórios

    # Troca os voos entre os índices i e j na pista selecionada
    rw_out = rw_in.copy()  # Cria uma cópia da pista para realizar a troca
    rw_out[i], rw_out[j] = rw_out[j], rw_out[i]  # Realiza a troca de voos

    # Atualiza a lista de soluções com a pista modificada
    sol_out = sol[:]
    sol_out[k] = rw_out  # Modifica a pista selecionada na solução

    return sol_out  # Retorna a solução com a pista modificada

# Função para mover um voo de uma pista para outra
def neighbor_move_between_runways(sol):
    # Seleciona pistas com ao menos um voo
    src_candidates = [i for i, rw in enumerate(sol) if len(rw) > 0]
    if(not src_candidates):
        return sol  # Se não houver pistas com voos, retorna a solução original

    # Escolhe aleatoriamente uma pista de origem e uma de destino diferentes
    src = choice(src_candidates)  # Pista de origem
    dst = choice([i for i in range(len(sol)) if i != src])  # Pista de destino diferente da origem

    rw_src = sol[src]  # Pista de origem
    rw_dst = sol[dst]  # Pista de destino

    # Retira um voo aleatório da pista de origem
    idx_out = randint(len(rw_src))  # Escolhe um voo aleatório da pista de origem
    flight = rw_src[idx_out]  # Voo que será movido
    rw_src_new = np.delete(rw_src, idx_out)  # Remove o voo da pista de origem

    # Insere o voo na pista de destino em uma posição aleatória
    idx_in = randint(len(rw_dst) + 1)  # Escolhe um índice aleatório na pista de destino
    rw_dst_new = np.insert(rw_dst, idx_in, flight)  # Insere o voo na pista de destino

    # Atualiza as pistas diretamente na lista de soluções
    sol_out = sol[:]
    sol_out[src] = rw_src_new  # Modifica a pista de origem
    sol_out[dst] = rw_dst_new  # Modifica a pista de destino

    return sol_out  # Retorna a solução com a pista modificada

# Função para reinserir um voo dentro da mesma pista em uma nova posição
def neighbor_reinsert_in_runway(sol):
    # Seleciona pistas com ao menos 2 voos
    cand = [i for i, rw in enumerate(sol) if len(rw) >= 2]
    if(not cand):
        return sol  # Se não houver pistas com 2 ou mais voos, retorna a solução original

    k = choice(cand)  # Escolhe aleatoriamente uma pista com ao menos 2 voos
    rw_in = sol[k]  # Pista selecionada

    # Remove um voo aleatório da pista
    idx_out = randint(len(rw_in))  # Escolhe um voo aleatório da pista
    flight = rw_in[idx_out]  # Voo que será reinserido
    intermed = np.delete(rw_in, idx_out)  # Remove o voo da pista

    # Insere o voo em uma posição aleatória dentro da mesma pista
    idx_in = randint(len(intermed) + 1)  # Escolhe uma nova posição na pista
    rw_out = np.insert(intermed, idx_in, flight)  # Insere o voo na nova posição

    # Atualiza a pista diretamente na lista de soluções
    sol_out = sol[:]
    sol_out[k] = rw_out  # Modifica a pista com o voo reinserido

    return sol_out  # Retorna a solução com a pista modificada