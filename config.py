from networkx.generators.random_graphs import barabasi_albert_graph, erdos_renyi_graph, watts_strogatz_graph
from networkx.generators.community import stochastic_block_model
from networkx.generators.trees import random_tree
from networkx.generators.classic import star_graph
from networkx.generators.geometric import random_geometric_graph

def set_params(g_type, **kwargs):
    N = kwargs['n'] if 'n' in kwargs else 100
    params = {'n': N}
    if g_type == 'RG':
        # RG graph parameters 
        rad = kwargs['rad'] if 'rad' in kwargs else 0.3
        params['radius'] = round(rad *(100/N)**.5, 3) # adjust for larger graphs
        gen_func = random_geometric_graph
    elif g_type == 'BA':
        m = kwargs['m'] if 'm' in kwargs else 3
        params['m'] = m # BA graph parameters (edges_per_node)
        gen_func = barabasi_albert_graph
    elif g_type == 'ER':
        p = kwargs['p'] if 'p' in kwargs else 0.06
        params['p'] = p # ER graph parameters (edge_prob)
        gen_func = erdos_renyi_graph
    elif g_type == 'SW':
        # SW graph parameters (knn, rewire_prob)
        k = kwargs['k'] if 'k' in kwargs else 6
        p = kwargs['p'] if 'p' in kwargs else 5*1e-3 # 0, 1e-3, [5*1e-3], 5*1e-2, 1e-1, 0.2
        params['k'] = k
        params['p'] = p
        gen_func = watts_strogatz_graph
    elif g_type == 'SBM':
        # SBM graph parameters
        SBM_sizes = [30, 30, 40]
        P = 0.17
        Q = 0.005
        SBM_probs = \
        [[P, Q, Q],
         [Q, P, Q],
         [Q, Q, P]]    
        params['sizes'] = SBM_sizes
        params['p'] = SBM_probs
        params.pop('n')
        gen_func = stochastic_block_model
    elif g_type == 'TR':
        gen_func = random_tree
    elif g_type == 'ST':
        params['n'] -= 1
        gen_func = star_graph
    else:
        assert False, 'Bad topology argument !!! Exiting...'    
    return params, gen_func
