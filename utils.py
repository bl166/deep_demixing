import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle 
from torch_geometric.data import Data, DataLoader, DataListLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse 
import torch.nn.functional as F

from torch.nn import DataParallel as DataParallel
from torch_geometric.nn import DataParallel as geoDataParallel


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


"""==== for kl divergence loss computation in a DataParallel manner ===="""     
    
from torch.distributions.multivariate_normal import _batch_mahalanobis, MultivariateNormal
from torch.distributions.kl import _batch_trace_XXT

# @register_kl(MultivariateNormal, MultivariateNormal)
def _kl_divergence_mvn_mvn(mu_q, sigma_q, mu_p, sigma_p):
    # Adapted from https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#_kl_multivariatenormal_multivariatenormal
    # https://github.com/pytorch/pytorch/blob/master/torch/distributions/multivariate_normal.py
    
    if mu_p.shape[0] != mu_q.shape[0]:
        raise ValueError("KL-divergence between two Multivariate Normals with\
                          different event shapes cannot be computed")
        
#     q = MultivariateNormal(mu_q, torch.diag_embed(torch.exp(sigma_q)))
#     p = MultivariateNormal(mu_p, torch.diag_embed(torch.exp(sigma_p)))            
        
    q_covmat = torch.diag_embed(torch.exp(sigma_q))
    p_covmat = torch.diag_embed(torch.exp(sigma_p))
    
    q_unbroadcasted_scale_tril = torch.cholesky(q_covmat)
    p_unbroadcasted_scale_tril = torch.cholesky(p_covmat)
    
    half_term1 = (q_unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) -
                  p_unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    combined_batch_shape = torch._C._infer_size(q_unbroadcasted_scale_tril.shape[:-2],
                                                p_unbroadcasted_scale_tril.shape[:-2])
    n = mu_q.shape[0]
    q_scale_tril = q_unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    p_scale_tril = p_unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    term2 = _batch_trace_XXT(torch.triangular_solve(p_scale_tril, q_scale_tril, upper=False)[0])
    
    term3 = _batch_mahalanobis(q_unbroadcasted_scale_tril, (mu_q - mu_p))
    return half_term1 + 0.5 * (term2 + term3 - n)


# Adapted from https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/data_parallel.py
def _data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = torch.nn.parallel.replicate(module, device_ids)
    inputs = torch.nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    res = torch.nn.parallel.gather(outputs, output_device)
    return res

    
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

        q_unbroadcasted_scale_tril = torch.cholesky(q_covmat)
        p_unbroadcasted_scale_tril = torch.cholesky(p_covmat)

        half_term1 = (q_unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) -
                      p_unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1))
        combined_batch_shape = torch._C._infer_size(q_unbroadcasted_scale_tril.shape[:-2],
                                                    p_unbroadcasted_scale_tril.shape[:-2])
        n = mu_q.shape[0]
        q_scale_tril = q_unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
        p_scale_tril = p_unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
        term2 = _batch_trace_XXT(torch.triangular_solve(p_scale_tril, q_scale_tril, upper=False)[0])

        term3 = _batch_mahalanobis(q_unbroadcasted_scale_tril, (mu_q - mu_p))
        kl = half_term1 + 0.5 * (term2 + term3 - n)        
        return kl

    
class KLDLoss(torch.nn.Module):
    def __init__(self, reduction="sum", device_ids=None, output_device=None):
        super().__init__()
        self.module = _KLD_MVNx2()
        self.reduce = reduction

        if 1:#not device_ids or len(device_ids)<2:
            self.parallel= False
        else:
            self.parallel= True
            if output_device is None:
                self.device_ids = device_ids
                self.output_device = device_ids[0]  
            
    def forward(self, mu_q, sigma_q, mu_p, sigma_p):
        if not self.parallel:
            kl = self.module(mu_q, sigma_q, mu_p, sigma_p)
        else:
            kl = _data_parallel(self.module, (mu_q, sigma_q, mu_p, sigma_p), self.device_ids, self.output_device)
            
        if self.reduce == 'sum':
            return torch.sum(kl) 
        else:
            raise

            
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
     
#     assert torch.max(A_coo)<=m
   
#     A = COO_to_matrix(A_coo, m) + torch.eye(m).to(y_hat.device)
#     A = torch.block_diag(*[A]*bs)

    A_coo = A_coo.reshape((bs, 2, -1))
    A_sparse = []
    for b in range(bs):
        A_sparse.append(COO_to_matrix(A_coo[b]%m, m) + torch.eye(m).to(y_hat.device))
    A = torch.block_diag(*A_sparse)        

    reg = torch.sum(F.relu(y_hat[...,1:] - torch.matmul(A,y_hat[...,:-1])))
    
    return reg


def load_dataset(filepath, N_train, N_val, batch_size, device=None, shuffle_data = False, parallel = False, **kwargs):
    
    shuffle_data = shuffle_data and N_train

    file = open(filepath, 'rb')
    dataset = pickle.load(file)
    file.close()
    
    A_mat = dataset['adj']
    node_attributes = dataset['attr']
    data = dataset['data']

    if 'shift' in kwargs.keys() and kwargs['shift']:
        shift =   kwargs['shift']
        assert shift+N_train + N_val<= len(data)
    else:
        shift = 0    
        
    A_mat_torch = torch.from_numpy(A_mat)
    A_coo = matrix_to_COO(A_mat_torch)
    
    train_list = []
    val_list = []

    # GUHA's notation!!! Y - uncollapsed; X - collapsed
    for i in range(shift, N_train+shift):
        curr_y = torch.from_numpy(data[i][0].astype(float)).type(torch.float32)
        curr_x = torch.from_numpy(data[i][1].astype(float)).type(torch.float32)

        if curr_x.ndim ==1:
            curr_x = curr_x.reshape((-1,1))
            
        train_list.append(Data(x=curr_x, edge_index=A_coo, pos=node_attributes, y=curr_y))

    for i in range(shift+N_train, shift+N_train + N_val):
#         print(data[i][0])
        curr_y = torch.from_numpy(data[i][0].astype(float)).type(torch.float32)
        curr_x = torch.from_numpy(data[i][1].astype(float)).type(torch.float32)
        
        # assert x shape = [num nodes, 1], y shape = [num nodes, num steps]
        if curr_x.ndim ==1:
            curr_x = curr_x.reshape((-1,1))
            
        val_list.append(Data(x=curr_x, edge_index=A_coo, pos=node_attributes, y=curr_y))

    if parallel:
        train_loader = DataListLoader(train_list, batch_size=batch_size, shuffle=shuffle_data)
        validation_loader = DataListLoader(val_list, batch_size=min(batch_size, N_val or np.inf))
    else:
        train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=shuffle_data)
        validation_loader = DataLoader(val_list, batch_size=min(batch_size, N_val or np.inf))
        
    return train_loader, validation_loader



def calculate_metric(model, data_loader, **kwargs):

    return_pred = 'return_pred' in kwargs.keys() and kwargs['return_pred']
   
    if 'device' not in kwargs.keys() or kwargs['device'] is None:
        device = next(model.parameters()).device
    else:
        device = kwargs['device']

    model.eval()        
    max_nodes = data_loader.dataset[0].x.shape[0]
    
    if 'T' in kwargs.keys() and kwargs['T']:
        T = kwargs['T']
    else:
        T = data_loader.dataset[0].y.shape[1]
        
    i = 0
    running_metric = 0.0
            
    y_hat_, y_true_, A_coo_ = [],[],[]
    
    with torch.no_grad():
    
        for data in data_loader:

            i += 1

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
                data.y = data.y
                x_true = data.x
                y_true = data.y[...,:T]
                A_coo = data.edge_index
  
            model_output = model(data)#, device)

            if (len(model_output) == 2):
                y_hat, (mu_q, sigma_q, mu_p, sigma_p) = model_output
            else:
                y_hat = model_output #(400, 20) = 4x100x20

            y_hat = y_hat.reshape(y_true.shape)
            y_hat_.append(y_hat.cpu().numpy().reshape((-1, max_nodes, T)))
            y_true_.append(y_true.cpu().numpy().reshape((-1, max_nodes, T)))
            A_coo_.append(A_coo.cpu().numpy())
            
            # mse
            SE = (y_hat - torch.reshape(y_true, (-1, ))) ** 2
            sum_SE = torch.sum(SE)
            MSE = sum_SE / (max_nodes * T)
            running_metric += float(MSE)

    y_hat_ = np.concatenate(y_hat_)
    y_true_ = np.concatenate(y_true_)
    running_metric = np.round(running_metric / (i*data_loader.batch_size) , 4)
    
    if not return_pred:
        return running_metric
    else:
        return running_metric, (y_true_, y_hat_, A_coo_)
