import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pickle 
import time
import sys
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data, DataLoader, DataListLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse 
import torch.nn.functional as F
from torch.nn import DataParallel as DataParallel
from torch_geometric.nn import DataParallel as geoDataParallel
from torch.distributions.multivariate_normal import MultivariateNormal

def get_infected_neighbors(G, node, node_values):

    result = []
    neighbors_dict = G.neighbors(node)
    
    for neighbor in neighbors_dict:
        if node_values[neighbor] == 1:
            result.append(neighbor)

    return result 


# Returns the number of infected neighbors of a given node
def count_infected_neighbors(G, node, node_values):

    counter = 0
    neighbors_dict = G.neighbors(node)
    
    for neighbor in neighbors_dict:
        if node_values[neighbor] == 1:
            counter += 1
    
    return counter


def load_graph_from_dataset(filepath, node_number):
    
    # Existing graph:
    N_train = 1
    N_val = 0
    batch_size = 1
    num_nodes = node_number
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader, _ = load_dataset(filepath, N_train, N_val, batch_size, device)
    edge_index = loader.dataset[0].edge_index
    pos = loader.dataset[0].pos

    A = COO_to_matrix(edge_index, num_nodes).cpu().numpy()
    graph = nx.from_numpy_matrix(A)
    nx.set_node_attributes(graph, pos, "pos")

    return graph



def draw_one_day(A_coo, pos, num_nodes, ground_truth, prediction, fig_path):
    """
    If you want to draw only one day, without a subplot, pass it as 
    ground_truth, and pass None for prediction
    """

    fig_name = os.path.basename(fig_path)
    
    cmap = sns.cubehelix_palette(as_cmap=True)

    # num_nodes = len(pos)
    A = COO_to_matrix(A_coo, num_nodes).cpu().numpy()
    graph = nx.from_numpy_matrix(A)

    if pos is None or len(pos) == 0:
        pos = nx.kamada_kawai_layout(graph)
    
    ground_truth = ground_truth.cpu().numpy()[ : num_nodes]

    gt_labels = {}
    for i, val in enumerate(ground_truth):
        gt_labels.update({i : np.round(val, 2)[0]})
    
    gt_color_map = []
    for i, _ in enumerate(graph):
        gt_color_map.append(cmap(float(ground_truth[i])))

    nx.set_node_attributes(graph, pos, "pos")

    if prediction is not None:
        prediction = prediction.cpu().numpy()[ : num_nodes]

        pr_labels = {}
        for i, val in enumerate(prediction):
            pr_labels.update({i : np.round(val, 2)[0]})

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(30,10), dpi=100)
        
        ax[0].set_title('Ground truth', fontsize=30)
        ax[1].set_title('Prediction', fontsize=30)

        pr_color_map = []
        for i, _ in enumerate(graph):
            pr_color_map.append(cmap(float(prediction[i])))

        nx.draw(
            graph, pos=pos, node_color=gt_color_map, node_size=500, width=0.1, ax=ax[0],
            with_labels=True, labels=gt_labels, font_color='white', font_size=10
            )
        
        nx.draw(
            graph, pos=pos, node_color=pr_color_map, node_size=500, width=0.1, ax=ax[1], 
            with_labels=True, labels=pr_labels, font_color='white', font_size=10
            )

    else:
        fig = plt.figure(figsize=(15, 10))

        nx.draw(
            graph, pos=pos, node_color=gt_color_map, node_size=500, width=0.1, 
            with_labels=True, labels=gt_labels, font_color='white', font_size=10
            )

    fig.suptitle(fig_name, fontsize=30)
    plt.savefig(fig_path)
    plt.close(fig)
    plt.show()
    

def draw_full_evolution(A_coo, pos, ground_truth, prediction, figures_dir):
    """
    If you want to draw only one day, without a subplot, pass it as 
    ground_truth, and pass None for prediction
    """

    if len(pos) == 0:
        pos = None

    num_nodes = ground_truth.shape[0] 
    num_days = ground_truth.shape[1]

    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    for day in range(num_days):

        fig_name = os.path.join(figures_dir, f"day_{day:03d}.png")
        curr_day_gt = ground_truth[ : num_nodes, day].reshape((-1, 1))

        if prediction is not None:
            curr_day_pr = prediction[ : num_nodes, day].reshape((-1, 1))
        else:
            curr_day_pr = None

        draw_one_day(A_coo, pos, num_nodes, curr_day_gt, curr_day_pr, fig_name)



# Function that draws a graph so that infected nodes are drawn red and susceptible nodes are drawn blue
# The colormap is chosen so that it maps: 0 -> Blue, 1 -> Red
 
def draw_epidemic_network(G, colors_dict, with_labels = True, labels = None, figsize = (6.4, 4.8), layout = "shell", plt_show = True, axis = None):

    colors = []
    for node in G.nodes():
        if (colors_dict.get(node, 0) == 0):
            colors.append('blue')
        elif (colors_dict.get(node, 0) == 0.9):
            colors.append('orangered')
        elif (colors_dict.get(node, 0) == 1):
            colors.append('red')

    # colors = [colors_dict.get(node, 0) for node in G.nodes()]

    plt.figure(figsize=figsize)

    if (layout == "shell"):
        pos = nx.shell_layout(G)
    elif (layout == "kamada_kawai"):
        pos = nx.kamada_kawai_layout(G)
    elif (layout == "geometric"):
        pos = nx.get_node_attributes(G, "pos")

    if (axis == None):
        if (with_labels == False and labels == None):
            nx.draw(G, pos=pos, cmap=plt.get_cmap('rainbow'), node_color=colors, node_size=node_size, with_labels=with_labels, font_color='white')
        else:
            nx.draw(G, pos=pos, cmap=plt.get_cmap('rainbow'), node_color=colors, node_size=600, with_labels=with_labels, labels = labels, 
                    font_color='white', font_size=10)
    else:
        if (with_labels == False and labels == None):
            nx.draw(G, pos=pos, cmap=plt.get_cmap('rainbow'), node_color=colors, node_size=node_size, with_labels=with_labels, font_color='white', ax = axis)
        else:
            nx.draw(G, pos=pos, cmap=plt.get_cmap('rainbow'), node_color=colors, node_size=600, with_labels=with_labels, labels = labels, 
                    font_color='white', font_size=10, ax = axis)

    if (plt_show == True):
        plt.show()

    


"""
Next few functions are used for certain sparse-dense matrix conversions. 
Most of them are not used at the moment since I found the corresponding 
built-in functions, which are being called from my update_x_W_A function.

"""

def COO_to_matrix(edge_index, num_nodes):
    "Converts 2xM to num_nodes x num_nodes"
    
    try:
        A_matrix = torch.zeros((num_nodes, num_nodes), dtype = torch.float).to(edge_index.device)
    except:
        A_matrix = torch.zeros((num_nodes, num_nodes), dtype = torch.float)
    
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    N = source_nodes.shape[0]
        
    for iter in range(0, N):    
        i = source_nodes[iter]
        j = target_nodes[iter]
        
        A_matrix[i][j] = 1
    
    return A_matrix


def W_matrix_to_A_matrix(W_matrix, device=None):

    dim1, dim2 = W_matrix.shape
    if not W_matrix.is_cuda: #(device.type == 'cpu'):
        A_matrix = torch.FloatTensor(dim1, dim2).fill_(1)
    else: 
        A_matrix = torch.cuda.FloatTensor(dim1, dim2).fill_(1)
    A_mask = (W_matrix > 0).type(torch.LongTensor)#.to(device)

    A_matrix = A_matrix * A_mask

    # A_matrix = (W_matrix > 0).type(torch.FloatTensor).to(device)
    return A_matrix
    

def matrix_to_COO(matrix):

    source_nodes = torch.tensor([], dtype=torch.long)
    target_nodes = torch.tensor([], dtype=torch.long)

    N, M = matrix.shape[0], matrix.shape[1]

    for i in range(0, N):
        for j in range(0, M):

            if(matrix[i][j] > 0):
                source_nodes = torch.cat((source_nodes, torch.tensor([i], dtype=torch.long)), 0 )
                target_nodes = torch.cat((target_nodes, torch.tensor([j], dtype=torch.long)), 0 )
    
    source_nodes = torch.reshape(source_nodes, (1, source_nodes.shape[0]))
    target_nodes = torch.reshape(target_nodes, (1, target_nodes.shape[0]))

    coo = torch.cat((source_nodes, target_nodes), 0)
    
    return coo


def weight_matrix_to_COO(W_matrix, A_coo):

    source_nodes = A_coo[0]
    target_nodes = A_coo[1]

    N = source_nodes.shape[0]

    weight_coo = torch.zeros(N, dtype=torch.float)
    
    for iter in range(0, N):
        i = source_nodes[iter]
        j = target_nodes[iter]

        weight_coo[iter] = W_matrix[i][j]
    
    return weight_coo


def update_x_W_A(x, W_matrix, device=None):

     # reshape x and W((1, dim1, dim2) to (dim1, dim2))
    _, dim1, dim2 = x.shape
    x = torch.reshape(x, (dim1, dim2))

    _, dim1, dim2 = W_matrix.shape 
    W_matrix = torch.reshape(W_matrix, (dim1, dim2))

    A_matrix = W_matrix_to_A_matrix(W_matrix)#, device)

    # A_coo = matrix_to_COO(A_matrix)
    A_coo = dense_to_sparse(A_matrix)[0]

    # W_coo = weight_matrix_to_COO(W_matrix, A_coo)
    W_coo = dense_to_sparse(W_matrix)[1]

    return x, W_matrix, W_coo, A_matrix, A_coo


"""==== for KL divergence loss computation (in a DataParallel manner) ===="""     
    
from torch.distributions.multivariate_normal import _batch_mahalanobis
from torch.distributions.kl import _batch_trace_XXT

# class _KLD_MVNx2(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, mu_q, sigma_q, mu_p, sigma_p):
#     # Adapted from 
#     # https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#_kl_multivariatenormal_multivariatenormal
#     # https://github.com/pytorch/pytorch/blob/master/torch/distributions/multivariate_normal.py
    
#         if mu_p.shape[0] != mu_q.shape[0]:
#             raise ValueError("KL-divergence between two Multivariate Normals with\
#                               different event shapes cannot be computed")
            
#         q_covmat = torch.diag_embed(torch.exp(sigma_q))
#         p_covmat = torch.diag_embed(torch.exp(sigma_p))

#         q_unbroadcasted_scale_tril = torch.linalg.cholesky(q_covmat)
#         p_unbroadcasted_scale_tril = torch.linalg.cholesky(p_covmat)

#         half_term1 = (q_unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) -
#                       p_unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1))
#         combined_batch_shape = torch._C._infer_size(q_unbroadcasted_scale_tril.shape[:-2],
#                                                     p_unbroadcasted_scale_tril.shape[:-2])
#         n = mu_q.shape[0]
#         q_scale_tril = q_unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
#         p_scale_tril = p_unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
#         term2 = _batch_trace_XXT(torch.triangular_solve(p_scale_tril, q_scale_tril, upper=False)[0])

#         term3 = _batch_mahalanobis(q_unbroadcasted_scale_tril, (mu_q - mu_p))
#         kl = half_term1 + 0.5 * (term2 + term3 - n)        
#         return kl
class _KLD_MVNx2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mu_q, sigma_q, mu_p, sigma_p):
    # Adapted from 
    # https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#_kl_multivariatenormal_multivariatenormal
    # https://github.com/pytorch/pytorch/blob/master/torch/distributions/multivariate_normal.py
    
        if mu_p.shape[0] != mu_q.shape[0]:
            raise ValueError("KL-divergence between two Multivariate Normals with\
                              different event shapes cannot be computed")
        
        q_covmat = torch.diag_embed(torch.exp(sigma_q))
        p_covmat = torch.diag_embed(torch.exp(sigma_p))
                
        q = MultivariateNormal(mu_q, q_covmat)
        p = MultivariateNormal(mu_p, p_covmat)
        
        kl = torch.distributions.kl.kl_divergence(p, q)
        return kl
    
class KLDLoss(torch.nn.Module):
    def __init__(self, reduction="sum", device_ids=None, output_device=None):
        super().__init__()
        self.module = _KLD_MVNx2()
        self.reduce = reduction

        if not device_ids or len(device_ids)<2:
            self.parallel = False
        else:
            self.parallel = True
            if output_device is None:
                self.device_ids = device_ids
                self.output_device = device_ids[0]  
        self.parallel = False  ## TODO: KLD loss parallelism
            
    def forward(self, mu_q, sigma_q, mu_p, sigma_p):
        if not self.parallel:
            kl = self.module(mu_q, sigma_q, mu_p, sigma_p)
        else:
            # Adapted from https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/data_parallel.py
            input = (mu_q, sigma_q, mu_p, sigma_p)
            replicas = torch.nn.parallel.replicate(self.module, self.device_ids)
            inputs = torch.nn.parallel.scatter(input, self.device_ids)
            replicas = replicas[:len(inputs)]
            outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
            kl = torch.nn.parallel.gather(outputs, self.output_device)
            
        if self.reduce == 'sum':
            return torch.sum(kl) 
        else:
            raise ValueError
            
def parallalize_model(mdl, device_ids):
    if device_ids:
        m_class = '_'.join(type(mdl).__name__.split('_')[:-1])
                            
        # ex: dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        if 'CVAE' in m_class.upper():
            print('graph parallel!')
            return geoDataParallel(mdl, device_ids)
        else:
            print('regular parallel!')
            return DataParallel(mdl, device_ids)    
    else:
        return mdl            
         
"""========================= (end) ============================"""     


class NILReg(torch.nn.Module):
    def __init__(self):
        super(NILReg, self).__init__()
        
    def forward(self, y_hat, A_coo, **kwargs):
        return _neighbor_infecting_rule(y_hat, A_coo, **kwargs)

class NILWithLogitsReg(torch.nn.Module):
    def __init__(self):
        super(NILWithLogitsReg, self).__init__()
        
    def forward(self, logits, A_coo, **kwargs):
        y_hat = torch.sigmoid(logits)
        return _neighbor_infecting_rule(y_hat, A_coo, **kwargs)

def _neighbor_infecting_rule(y_hat, A_coo, **kwargs):
    m = kwargs['num_nodes'] # max nodes
    bs = y_hat.shape[0]//m # batch size
    A_coo = A_coo.reshape((bs, 2, -1))
    A_sparse = []
    for b in range(bs):
        A_sparse.append(COO_to_matrix(A_coo[b]%m, m) + torch.eye(m).to(y_hat.device))
    A = torch.block_diag(*A_sparse)        
    reg = torch.sum(F.relu(y_hat[...,1:] - torch.matmul(A,y_hat[...,:-1])))
    return reg


def load_dataset(filepath, N_train, N_val, batch_size, 
                 T=None, device=None, shuffle_data=False, aggreg=None, parallel=False, **kwargs):
    """
    Load dataset with specified timesteps, shuffling strategy, permutation, and aggregation options.
    
    Inputs:
        - filepath (str)       : dataset file path
        - N_train (int)        : number of training samples (first N_train samples in the dataset will be trained)
        - N_val (int)          : number of validation samples (next N_val samples in the dataset will be validated)
        - batch_size (int)
        - T (int)              : how many days of epidemics to return; default is all T' days in the original dataset. 
                                 If set a value T<=T', then return the first T days values.
        - device (torch.device): move data to certain gpu/cpu device
        - shuffle_data (bool)  : whether or not to shuffle the training samples
        - aggreg (str)         : which aggregation method to use, including: 
                                 1) avg = average of all chosen days; 
                                 2) acc = accumulated average (first infection day) of all chosen days
                                 3) last = binary states of the last chosen day
                                 4) OPT_1+...+OPT_k = combinations of the above options
        - parallel (bool)      : pytorch data parallelism
        - kwargs:
            + shift (int)      : number of days to shift when getting training data
            + miss_rate ([float, float]): rate of missing nodes and days
            + miss_seed (int)  : random seed for simulating missingness
    """
    with open(filepath, 'rb') as file:
        dataset = pickle.load(file)
    
    A_mat = dataset['adj']
    A_coo = matrix_to_COO(torch.from_numpy(A_mat))
    node_attributes = dataset['attr']
    node_class = dataset['cls'] if 'cls' in dataset else {}
    graph = dataset['graph'] if 'graph' in dataset else None
        
    data = dataset['data']
        
    n_nodes, timesteps = data[0][0].shape # get from y's shape
    if T is not None: 
        timesteps = T

    # process kwargs
    if 'shift' in kwargs and kwargs['shift']:
        shift = kwargs['shift']
        assert shift+N_train+N_val<= len(data)
    else:
        shift = 0 
    
    if 'miss_rate' in kwargs and kwargs['miss_rate']:
        mflag = True
        mrate_nodes, mrate_days = kwargs['miss_rate']
        mseed = kwargs['miss_seed'] if 'miss_seed' in kwargs else None
    else:
        mflag = False
        
    train_list, val_list = [], []
    # GUHA's notation!!! Y - uncollapsed; X - collapsed
    for i in range(shift, shift+N_train+N_val):
        curr_y = torch.from_numpy(data[i][0].astype(float)).type(torch.float32)#[...,:timesteps]
        if 'spread' in filepath:
            assert aggreg
            curr_y = torch.from_numpy(data[i][1].astype(float)).type(torch.float32)[...,:timesteps]
        else:
            curr_y = torch.from_numpy(data[i][0].astype(float)).type(torch.float32)[...,:timesteps]

        if not aggreg:
            curr_x = torch.from_numpy(data[i][1].astype(float)).type(torch.float32)[...]
        else:
            curr_x = []
            for opt in aggreg.split('+'):
                opt_str = opt.lower()
                if opt_str=='avg':
                    curr_x.append(curr_y.mean(-1))
                elif opt_str.startswith('acc'):
                    x = np.argmax(curr_y,-1).float()/timesteps
                    if rr:=opt_str[3:]:
                        n_x = int(len(x)*(1-int(rr)/100))
                        n_idx = np.random.permutation(len(x))[:n_x]
                        x[n_idx] = 0
                    curr_x.append(x)
                elif opt_str=='last':
                    curr_x.append(curr_y[:,-1])
                else:
                    raise
            curr_x = torch.stack(curr_x, -1)
        if curr_x.ndim ==1:
            curr_x = curr_x.reshape((-1,1))
            
        # missingness
        if mflag:
            pass # TODO        
            
        if i < N_train+shift:
            train_list.append(Data(x=curr_x, edge_index=A_coo, pos=node_attributes, cls=node_class, y=curr_y, graph=graph))
        else:
            val_list.append(Data(x=curr_x, edge_index=A_coo, pos=node_attributes, cls=node_class, y=curr_y, graph=graph))
                        
    DL = DataListLoader if parallel else DataLoader
    train_loader = DL(train_list, batch_size=batch_size, shuffle=shuffle_data)
#     validation_loader = DL(val_list, batch_size=min(max(batch_size, N_val), np.inf), shuffle=False)
    validation_loader = DL(val_list, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader

load_dataset_withrandmiss = load_dataset


def spread_score(full_evolution, graph, normalize=None):
    [N, T] = full_evolution.shape
    # Initialize all weights to zero
    spreading_weights = np.zeros(N)
    A = np.squeeze(np.asarray(
        nx.adjacency_matrix(graph, nodelist=graph.nodes()).todense()
    ))
    daily_new_infections = full_evolution[:, 1:] - full_evolution[:, :-1]
    for day in range(T-1):
        new_infections = daily_new_infections[:, day] > 0.
        infecting_neighbors = A[new_infections, :]
        expand_evol = np.tile(full_evolution[:, day], (infecting_neighbors.shape[0],1))
        infecting_weights = expand_evol * infecting_neighbors    
        infecting_weights /= infecting_weights.sum(1)[:,None]
        spreading_weights += np.array(infecting_weights).sum(0)
    if normalize is not None:
        spreading_weights = normalize(spreading_weights)
    return spreading_weights

def compute_r2_weighted(y_true, y_pred, weight):
    sse = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    tse = (weight * (y_true - np.average(
        y_true, axis=0, weights=weight)) ** 2).sum(axis=0, dtype=np.float64)
    r2_score = 1 - (sse / tse)
    return r2_score, sse, tse    

def compute_r2(y_true, y_predicted):
    sse = np.sum((y_true - y_predicted)**2)
    tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
    r2_score = 1 - (sse / tse)
    return r2_score, sse, tse

def find_optimal_cutoff(target, predicted):
    # https://stackoverflow.com/a/56204455
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']) 


def demix_predict(model, data_loader, T=None, device=None):
    if T is None:
        T = data_loader.dataset[0].y.shape[1]
    if device is None:
        device = next(model.parameters()).device
        
    max_nodes = data_loader.dataset[0].x.shape[0]
    unroll_shape = (-1, max_nodes, T)

    running_metric = 0.
    y_hat_, y_true_, A_coo_ = [],[],[]
    
    t_clock = 0
    
    model.eval()     
    with torch.no_grad():
        for data in data_loader:
            # load data and labels            
            if isinstance(data, list):
                [d.to(device) for d in data]
                for dd in range(len(data)): # update x steps
                    data[dd].y = data[dd].y[...,:T]
                x_true = torch.cat([d.x for d in data],0)
                y_true = torch.cat([d.y for d in data],0)
                A_coo = torch.cat([d.edge_index for d in data],0)
            else:
                data.to(device)
                x_true = data.x
                y_true = data.y[...,:T]
                A_coo = data.edge_index
  
            t1 = time.time()
            model_output = model(data, device)
            t2 = time.time()
            t_clock += (t2-t1)
            

            if (len(model_output) == 2):
                y_hat, (mu_q, sigma_q, mu_p, sigma_p) = model_output
            else:
                y_hat = model_output #(400, 20) = 4x100x20

            y_hat = y_hat.reshape(y_true.shape)
            y_hat_.append(y_hat.cpu().numpy().reshape(*unroll_shape))
            y_true_.append(y_true.cpu().numpy().reshape(*unroll_shape))
            A_coo_.append(A_coo.cpu().numpy())
    
    y_true_ = np.concatenate(y_true_).reshape(*unroll_shape)
    y_hat_ = np.concatenate(y_hat_).reshape(*unroll_shape)
    
    print(model.__class__, 'Time:', t_clock)
    
    return y_true_, y_hat_, A_coo_


def calculate_metric(model, data_loader, **kwargs):
    return_pred = kwargs['return_pred'] if 'return_pred' in kwargs else False
    metric = kwargs['metric'] if 'metric' in kwargs else 'MSE'
    device = kwargs['device'] if 'device' in kwargs else None
    thresh = kwargs['threshold'] if 'threshold' in kwargs else .5
    rpath  = kwargs['path'] if 'path' in kwargs else None
    
    max_nodes, max_steps = data_loader.dataset[0].y.shape
    T = kwargs['T'] if 'T' in kwargs.keys() else max_steps

    if rpath and os.path.exists(rpath):
        results = pickle.load(open(rpath, 'rb'))
        y_true = results['y_true']
        y_hat =  results['y_hat']
        A_coo =  results['A_coo']
    else:    
        y_true, y_hat, A_coo = demix_predict(model, data_loader, T, device)
        if rpath:
            results = {
                'y_true': y_true,
                'y_hat': y_hat,
                'A_coo': A_coo,
            }
            os.makedirs(os.path.dirname(rpath), exist_ok=True)
            pickle.dump(results, open(rpath,'wb') )
        
    y_pred = y_hat >= thresh

    y_true_1d = y_true.reshape(-1)
    y_hat_1d  = y_hat.reshape(-1)
    y_pred_1d = y_pred.reshape(-1)

    y_true_2d = y_true.reshape(-1, T*max_nodes)
    y_hat_2d = y_hat.reshape(-1, T*max_nodes)
    

    if metric == 'MSE':
        running_metric = np.mean((y_true_1d-y_hat_1d)**2)
    elif metric == 'RMSE':
        running_metric = np.mean(
            [mean_squared_error(y_hat_2d[ii], y_true_2d[ii], squared=False) for ii in range(y_true_2d.shape[0])]
        )        
    elif metric == 'AUC':
        running_metric = roc_auc_score(y_true=y_true_1d, y_score=y_hat_1d)
    elif metric == 'F1':
        running_metric = f1_score(y_true=y_true_1d, y_pred=y_pred_1d)
    elif metric == 'ACC':
        running_metric = accuracy_score(y_true=y_true_1d, y_pred=y_pred_1d)
    elif metric == 'FCS':
        yt_daily_mean = y_true.mean(0).mean(0)
        yp_daily_mean = y_pred.mean(0).mean(0)
#         running_metric = np.corrcoef(yt_daily_mean, yp_daily_mean)[0,1]
        running_metric = cosine_similarity([yt_daily_mean], [yp_daily_mean])[0,0]
    else:
        raise

    if not return_pred:
        return running_metric
    else:
        return running_metric, (y_true, y_hat, A_coo)


    
    
from matplotlib.gridspec import SubplotSpec
from colorsys import hls_to_rgb

def get_distinct_colors(n):
    colors = []
    for i in np.arange(0., 360., 360. / n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))
    return colors

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', y=-0.15,pad=-10)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, **kwargs):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """
    
    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            #bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
            bar = ax.barh(x + x_offset, y, height=bar_width * single_width, color=colors[i % len(colors)], edgecolor='w')

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])
        
    if 'xticks' in kwargs:
        if kwargs['xticks']:
            ax.set(xticks=range(len(kwargs['xticks'])), xticklabels=kwargs['xticks'])
        else:
            ax.set_xticks([])
            ax.set_xticklabels([], minor=True)
            
    if 'yticks' in kwargs:
        if kwargs['yticks']:
            ax.set(yticks=range(len(kwargs['yticks'])), yticklabels=kwargs['yticks'])
        else:
            ax.set_yticks([])
            ax.set_yticklabels([], minor=True)

    # Draw legend if we need
    if legend:
        return bars
        #ax.legend(bars, data.keys(), loc=7)

