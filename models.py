import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Dropout, ReLU, LeakyReLU,  Linear, Conv1d, ConvTranspose1d
from torch_geometric.nn import GCNConv, GraphUNet 
import numpy as np


def proc_input(data):
    if isinstance(data, torch.Tensor):
        x = data
    else:
        x = data.x  
    return x

class rnn(nn.Module):
    def __init__(self, T, max_nodes, feat_in, depth, hidden_size=None, cell_type=None, bidirectional=False):
        super(rnn, self).__init__()
        
        self.N = max_nodes
        self.in_size = self.N * feat_in
        self.hidden_size = self.in_size *2 if hidden_size is None else hidden_size
        self.T = T
        D = 2**bidirectional
        
        self.cell = cell_type(
            input_size=self.in_size, 
            hidden_size=self.hidden_size, 
            num_layers=depth, 
            bidirectional=bidirectional,
            dropout=0.5)
        self.out = nn.Linear(self.hidden_size * D, self.in_size)
        self.final = nn.Linear(self.in_size, self.N)
        self.relu = ReLU()
        
        if 'LSTM' in str(cell_type):
            self.init_hidden = self._init_hc
        elif 'GRU' in str(cell_type):
            self.init_hidden = self._init_h
        else:
            raise NotImplemented
            
    def _init_h(self, shp, dev):
        return torch.zeros(*shp, device=dev)
    
    def _init_hc(self, shp, dev):
        return (torch.zeros(*shp, device=dev), torch.zeros(*shp, device=dev))
    
    def forward(self, data, device=None):
        x = proc_input(data) 
        xshp = [1, -1, self.in_size] # input: tensor of shape (L, N, Hin)
        x = x.view(*xshp)
        D = 2**self.cell.bidirectional # h_0: tensor of shape  ( D * num_layers , N , Hout ) 
        h = self.init_hidden([D * self.cell.num_layers, x.shape[1], self.hidden_size], x.device)
        output = []
        for _ in range(self.T):
            dout, h = self.cell(x, h) # output, (h_n, c_n) -- output: tensor of shape  ( L , N , D * Hout ) 
            dout = self.out(dout[0])
            output.append(dout)
            x = dout.view(*xshp)
        x = torch.cat(output, 0)
        x = self.final(x)
        x = torch.sigmoid(x)
        #x = torch.flip(x, [1])
        return x
        
class LSTM_benchmark(rnn):
    def __init__(self, T, max_nodes, feat_in=1, depth=1, hidden_size=None, cell_type=nn.LSTM, bidirectional=False):
        super(LSTM_benchmark, self).__init__(T, max_nodes, feat_in, depth, hidden_size, cell_type, bidirectional)

class LSTM_bi_benchmark(rnn):
    def __init__(self, T, max_nodes, feat_in=1, depth=1, hidden_size=None, cell_type=nn.LSTM, bidirectional=True):
        super(LSTM_bi_benchmark, self).__init__(T, max_nodes, feat_in, depth, hidden_size, cell_type, bidirectional)
        
class GRU_benchmark(rnn):
    def __init__(self, T, max_nodes, feat_in=1, depth=1, hidden_size=None, cell_type=nn.GRU, bidirectional=False):
        super(GRU_benchmark, self).__init__(T, max_nodes, feat_in, depth, hidden_size, cell_type, bidirectional)
    
    
class MLP_benchmark(nn.Module):
    
    def __init__(self, T, max_nodes, feat_in=1, **kwargs):
        super(MLP_benchmark, self).__init__()
        
        self.T = T
        self.M = max_nodes
        hidden_size = self.M * int(np.ceil(T/4))
        self.cin = self.M * feat_in
        
        self.linears = nn.Sequential(
            Linear(self.cin, hidden_size),
            Dropout(0.5),
            ReLU(),
            Linear(hidden_size, hidden_size),
            Dropout(0.5),
            ReLU(),
            Linear(hidden_size, self.M * T),
            Dropout(0.5),
            ReLU(),
            Linear(self.M * T, self.M * T)
        )
            
    def forward(self, data, device=None):
        x = proc_input(data)
        x = x.view((-1, self.cin))
        x = self.linears(x)
        x = torch.sigmoid(x)
        x = x.view((-1, self.T))
        return x
    
    
class CNN_time_benchmark(nn.Module):

    def __init__(self, T, max_nodes, feat_in=1, **kwargs):
        super(CNN_time_benchmark, self).__init__()
        
        self.M = max_nodes
        self.cin = feat_in

        # Layer 0: Nx1x1 -> Nx1x1 => kernel_size = 1
        self.conv_0 = Conv1d(in_channels = self.cin, out_channels = 1, kernel_size = 1)
        self.relu_0 = ReLU()

        # Layer 1: Nx1x1 -> Nx1x (T/4) => kernel_size = (T/4)
        self.conv_1 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(np.ceil(T/4)))
        self.relu_1 = ReLU()

        # Layer 2: Nx1x (T/4) -> Nx1x (T/4) => kernel_size = 1
        self.conv_2 = Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1)
        self.relu_2 = ReLU()

        # Layer 3: Nx1x (T/4) -> Nx1x (T/2) => kernel_size = (T/2) - (T/4) + 1
        self.conv_3 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(np.ceil(T/2)) - int(np.ceil(T/4)) + 1)
        self.relu_3 = ReLU()

        # Layer 4: Nx1x (T/2) -> Nx1x (T/2) => kernel_size = 1
        self.conv_4 = Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1)
        self.relu_4 = ReLU()

        # Layer 5: Nx1x (T/2) -> Nx1x (T) => kernel_size = (T) - (T/2) + 1
        self.conv_5 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(T) - int(np.ceil(T/2)) + 1)
        self.relu_5 = ReLU()

        # Layer 6: Nx1x (T) -> Nx1x (T) => kernel_size = 1
        self.conv_6 = Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1)
        self.relu_6 = ReLU()
            
            
    def forward(self, data, device=None):
        x = proc_input(data)
        x = torch.reshape(x, (-1, self.cin, 1))
        
        # Layer 0:
        x_0 = self.conv_0(x)
        x_0 = self.relu_0(x_0)

        # Layer 1:
        x_1 = self.conv_1(x_0)
        x_1 = self.relu_1(x_1)

        # Layer 2:
        x_2 = self.conv_2(x_1)
        x_2 = self.relu_2(x_2)

        # Layer 3:
        x_3 = self.conv_3(x_2)
        x_3 = self.relu_3(x_3)

        # Layer 4:
        x_4 = self.conv_4(x_3)
        x_4 = self.relu_4(x_4)

        # Layer 5:
        x_5 = self.conv_5(x_4)
        x_5 = self.relu_5(x_5)

        # Layer 6:
        x_6 = self.conv_6(x_5)
        x_6 = torch.sigmoid(x_6)

        out = x_6
        out = torch.reshape(out, (out.shape[0], out.shape[2]))

        return out

    
class CNN_time2_benchmark(nn.Module):

    def __init__(self, T, max_nodes, feat_in=1, **kwargs):
        super(CNN_time2_benchmark, self).__init__()
        
        self.M = max_nodes
        self.cin = feat_in

        # Layer 0: Nx1x1 -> Nx1x1 => kernel_size = 1
        self.conv_0 = Conv1d(in_channels = self.cin, out_channels = 1,  kernel_size = 3, padding = 1)
        self.relu_0 = LeakyReLU()

        # Layer 1: Nx1x1 -> Nx1x (T/4) => kernel_size = (T/4)
        self.conv_1 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(np.ceil(T/4)))
        self.relu_1 = ReLU()

        # Layer 2: Nx1x (T/4) -> Nx1x (T/4) => kernel_size = 1
        self.conv_2 = Conv1d(in_channels = 1, out_channels = 1,  kernel_size = 3, padding = 1)
        self.relu_2 = ReLU()

        # Layer 3: Nx1x (T/4) -> Nx1x (T/2) => kernel_size = (T/2) - (T/4) + 1
        self.conv_3 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(np.ceil(T/2)) - int(np.ceil(T/4)) + 1)
        self.relu_3 = ReLU()

        # Layer 4: Nx1x (T/2) -> Nx1x (T/2) => kernel_size = 1
        self.conv_4 = Conv1d(in_channels = 1, out_channels = 1,  kernel_size = 3, padding = 1)
        self.relu_4 = ReLU()

        # Layer 5: Nx1x (T/2) -> Nx1x (T) => kernel_size = (T) - (T/2) + 1
        self.conv_5 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(T) - int(np.ceil(T/2)) + 1)
        self.relu_5 = ReLU()

        # Layer 6: Nx1x (T) -> Nx1x (T) => kernel_size = 1
        self.conv_6 = Conv1d(in_channels = 1, out_channels = 1,  kernel_size = 3, padding = 1)
        self.relu_6 = ReLU()
            
            
    def forward(self, data, device=None):
        x = proc_input(data)
        x = torch.reshape(x, (-1, self.cin, 1))
        
        # Layer 0:
        x_0 = self.conv_0(x)
        x_0 = self.relu_0(x_0)

        # Layer 1:
        x_1 = self.conv_1(x_0)
        x_1 = self.relu_1(x_1)

        # Layer 2:
        x_2 = self.conv_2(x_1)
        x_2 = self.relu_2(x_2)

        # Layer 3:
        x_3 = self.conv_3(x_2)
        x_3 = self.relu_3(x_3)

        # Layer 4:
        x_4 = self.conv_4(x_3)
        x_4 = self.relu_4(x_4)

        # Layer 5:
        x_5 = self.conv_5(x_4)
        x_5 = self.relu_5(x_5)

        # Layer 6:
        x_6 = self.conv_6(x_5)
        x_6 = torch.sigmoid(x_6)

        out = x_6
        out = torch.reshape(out, (out.shape[0], out.shape[2]))

        return out
    
    
class CNN_time3_benchmark(nn.Module):

    def __init__(self, T, max_nodes, feat_in=1, **kwargs):
        super(CNN_time3_benchmark, self).__init__()
        
        self.M = max_nodes
        self.cin = feat_in

        # Layer 0: Nx1x1 -> Nx1x1 => kernel_size = 1
        self.conv_0 = Conv1d(in_channels = self.cin, out_channels = 1,  kernel_size = 3, padding = 1)
        self.relu_0 = ReLU()

        # Layer 1: Nx1x1 -> Nx1x (T/4) => kernel_size = (T/4)
        self.conv_1 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(np.ceil(T/4)))
        self.relu_1 = ReLU()

        # Layer 2: Nx1x (T/4) -> Nx1x (T/4) => kernel_size = 1
        self.conv_2 = Conv1d(in_channels = 1, out_channels = 1,  kernel_size = 1)
        self.relu_2 = ReLU()

        # Layer 3: Nx1x (T/4) -> Nx1x (T/2) => kernel_size = (T/2) - (T/4) + 1
        self.conv_3 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(np.ceil(T/2)) - int(np.ceil(T/4)) + 1)
        self.relu_3 = ReLU()

        # Layer 4: Nx1x (T/2) -> Nx1x (T/2) => kernel_size = 1
        self.conv_4 = Conv1d(in_channels = 1, out_channels = 1,  kernel_size = 1)
        self.relu_4 = ReLU()

        # Layer 5: Nx1x (T/2) -> Nx1x (T) => kernel_size = (T) - (T/2) + 1
        self.conv_5 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(T) - int(np.ceil(T/2)) + 1)
        self.relu_5 = ReLU()

        # Layer 6: Nx1x (T) -> Nx1x (T) => kernel_size = 1
        self.conv_6 = Conv1d(in_channels = 1, out_channels = 1,  kernel_size = 1)
        self.relu_6 = ReLU()    
    
    def forward(self, data, device=None):
        x = proc_input(data)
        x = torch.reshape(x, (-1, self.cin, 1))
        
        # Layer 0:
        x_0 = self.conv_0(x)
        x_0 = self.relu_0(x_0)

        # Layer 1:
        x_1 = self.conv_1(x_0)
        x_1 = self.relu_1(x_1)

        # Layer 2:
        x_2 = self.conv_2(x_1)
        x_2 = self.relu_2(x_2)

        # Layer 3:
        x_3 = self.conv_3(x_2)
        x_3 = self.relu_3(x_3)

        # Layer 4:
        x_4 = self.conv_4(x_3)
        x_4 = self.relu_4(x_4)

        # Layer 5:
        x_5 = self.conv_5(x_4)
        x_5 = self.relu_5(x_5)

        # Layer 6:
        x_6 = self.conv_6(x_5)
        x_6 = torch.sigmoid(x_6)

        out = x_6
        out = torch.reshape(out, (out.shape[0], out.shape[2]))

        return out
    
    
    
class CNN_nodes_benchmark(nn.Module):

    def __init__(self, T, max_nodes, feat_in=1, **kwargs):
        super(CNN_nodes_benchmark, self).__init__()

        self.T = T
        self.M = max_nodes
        self.cin = feat_in
         
        # Layer 0: 1xCinxN -> 1x1xN
        self.conv_0 = Conv1d(in_channels = self.cin, out_channels = 1, kernel_size = 3, padding = 1)
        self.batch_norm_0 = BatchNorm1d(1, track_running_stats = False)
        self.relu_0 = ReLU()

        # Layer 1: 1x1xN -> 1x(T/4)xN 
        self.conv_1 = Conv1d(in_channels = 1, out_channels = int(np.ceil(T/4)), kernel_size = 3, padding = 1)
        self.batch_norm_1 = BatchNorm1d(int(np.ceil(T/4)), track_running_stats = False)
        self.relu_1 = ReLU()

        # Layer 2: 1x(T/4)xN -> 1x(T/4)xN
        self.conv_2 = Conv1d(in_channels = int(np.ceil(T/4)), out_channels = int(np.ceil(T/4)), kernel_size = 3, padding = 1)
        self.batch_norm_2 = BatchNorm1d(int(np.ceil(T/4)), track_running_stats = False)
        self.relu_2 = ReLU()

        # Layer 3: 1x(T/4)xN -> 1x(T/2)xN 
        self.conv_3 = Conv1d(in_channels = int(np.ceil(T/4)), out_channels = int(np.ceil(T/2)), kernel_size = 3, padding = 1)
        self.batch_norm_3 = BatchNorm1d(int(np.ceil(T/2)), track_running_stats = False)
        self.relu_3 = ReLU()

        # Layer 4: 1x(T/2)xN -> 1x(T/2)xN 
        self.conv_4 = Conv1d(in_channels = int(np.ceil(T/2)), out_channels = int(np.ceil(T/2)), kernel_size = 3, padding = 1)
        self.batch_norm_4 = BatchNorm1d(int(np.ceil(T/2)), track_running_stats = False)
        self.relu_4 = ReLU()

        # Layer 5: 1x(T/2)xN -> 1x(T)xN
        self.conv_5 = Conv1d(in_channels = int(np.ceil(T/2)), out_channels = int(T), kernel_size = 3, padding = 1)
        self.batch_norm_5 = BatchNorm1d(int(T), track_running_stats = False)
        self.relu_5 = ReLU()

        # Layer 6: 1x(T)xN -> 1x(T)xN
        self.conv_6 = Conv1d(in_channels = int(T), out_channels = int(T), kernel_size = 3, padding = 1)
        self.batch_norm_6 = BatchNorm1d(int(T), track_running_stats = False)
        self.relu_6 = ReLU()
        
    
    def forward(self, data, device=None):
        x = proc_input(data)
        x = torch.reshape(x, (-1, self.cin, self.M))

        # Layer 0:
        x_0 = self.conv_0(x)
        x_0 = self.batch_norm_0(x_0)
        x_0 = self.relu_0(x_0)

        # Layer 1:
        x_1 = self.conv_1(x_0)
        x_1 = self.batch_norm_1(x_1)
        x_1 = self.relu_1(x_1)

        # Layer 2:
        x_2 = self.conv_2(x_1)
        x_2 = self.batch_norm_2(x_2)
        x_2 = self.relu_2(x_2)

        # Layer 3:
        x_3 = self.conv_3(x_2)
        x_3 = self.batch_norm_3(x_3)
        x_3 = self.relu_3(x_3)

        # Layer 4:
        x_4 = self.conv_4(x_3)
        x_4 = self.batch_norm_4(x_4)
        x_4 = self.relu_4(x_4)

        # Layer 5:
        x_5 = self.conv_5(x_4)
        x_5 = self.batch_norm_5(x_5)
        x_5 = self.relu_5(x_5)

        # Layer 6:
        x_6 = self.conv_6(x_5)
        x_6 = torch.sigmoid(x_6) # final layer should be between 0,1 to use bce

        out = x_6
        out = torch.cat([out[i] for i in range(len(out))], -1)
        out = torch.transpose(out, 0, 1)
        return out
    

    
class VAE_UNET_Batch(nn.Module):

    def __init__(
        self, encoder_feat, prior_feat, T, max_nodes, depth=1, 
        encoder_feat_in=1, prior_feat_in=None, out_feat=None
        ):
        super(VAE_UNET_Batch, self).__init__()
        
        # hyperparams
        gd = depth # depth of the graph unet
        self.M = max_nodes
        self.T = T
        out_feat = T if out_feat is None else out_feat        
        
        self.encoder = GraphUNet(in_channels=encoder_feat_in, hidden_channels=encoder_feat, out_channels=encoder_feat, depth=gd, pool_ratios=0.5)
        self.prior = GraphUNet(in_channels=prior_feat_in, hidden_channels=prior_feat, out_channels=prior_feat, depth=gd, pool_ratios=0.5)

        # mu and sigma for prior net
        self.conv_mu_pr = GCNConv(in_channels=prior_feat, out_channels=prior_feat, normalize=True)
        self.conv_sigma_pr = GCNConv(in_channels=prior_feat, out_channels=prior_feat, normalize=True)

        self.batch_norm_mu_pr = BatchNorm1d(prior_feat, momentum=0.9, track_running_stats=False)
        self.batch_norm_sigma_pr = BatchNorm1d(prior_feat, momentum=0.9, track_running_stats=False)        

        # decoder
        self.decoder = GraphUNet(
            in_channels = encoder_feat + prior_feat, 
            hidden_channels = encoder_feat, 
            out_channels = out_feat, depth = gd, pool_ratios = 0.5
        )
               
               
    def forward(self, data, device=None):
        
        A_coo, y, x = data.edge_index, data.y, data.x
        
        batch = torch.repeat_interleave(input=torch.arange(x.shape[0]//self.M, device=x.device), repeats=self.M)
#         batch=None
 
        x_encoded = self.encoder(x, A_coo, batch=batch)

        # prior (p)
        y_prior = self.prior(x, A_coo)

        mu_pr = self.conv_mu_pr(y_prior, A_coo)
        mu_pr = torch.flatten(self.batch_norm_mu_pr(mu_pr))

        logstd_pr = 0.1 * self.conv_sigma_pr(y_prior, A_coo)
        logstd_pr = torch.flatten(self.batch_norm_sigma_pr(logstd_pr))
                
        # reparameterization 
        p = mu_pr + torch.randn_like(logstd_pr) * torch.exp(logstd_pr)   
        
        z = torch.cat((x_encoded, p.reshape(x_encoded.shape)), 1)

        y_hat = self.decoder(z, A_coo, batch=batch)
        y_hat = torch.sigmoid(y_hat)
        
        return y_hat, (None, None, mu_pr, logstd_pr)
    
    
    
    
class CVAE_UNET_Pr(nn.Module):

    def __init__(
        self, encoder_feat, posterior_feat, T, max_nodes, depth=1, 
        encoder_feat_in=1, prior_feat_in=1, posterior_feat_in=None, out_feat=None
        ):
        super(CVAE_UNET_Pr, self).__init__()
        
        # hyperparams
        gd = depth # depth of the graph unet
        self.M = max_nodes
        self.T = T
        prior_feat = posterior_feat
        posterior_feat_in = T if posterior_feat_in is None else posterior_feat_in
        out_feat = T if out_feat is None else out_feat        
        
        self.encoder = GraphUNet(in_channels=encoder_feat_in, hidden_channels=encoder_feat, out_channels=encoder_feat, depth=gd, pool_ratios=0.5)
        self.posterior = GraphUNet(in_channels=posterior_feat_in, hidden_channels=posterior_feat, out_channels=posterior_feat, depth=gd, pool_ratios=0.5)
        self.prior = GraphUNet(in_channels=prior_feat_in, hidden_channels=prior_feat, out_channels=prior_feat, depth=gd, pool_ratios=0.5)

        # mu and sigma for posterior net
        self.conv_mu = GCNConv(in_channels = posterior_feat, out_channels = posterior_feat, normalize = True)
        self.conv_sigma = GCNConv(in_channels = posterior_feat, out_channels = posterior_feat, normalize = True)

        self.batch_norm_mu = BatchNorm1d(posterior_feat, momentum=0.9, track_running_stats=False)
        self.batch_norm_sigma = BatchNorm1d(posterior_feat, momentum=0.9, track_running_stats=False)
        
        # mu and sigma for prior net
        self.conv_mu_pr = GCNConv(in_channels=prior_feat, out_channels=prior_feat, normalize=True)
        self.conv_sigma_pr = GCNConv(in_channels=prior_feat, out_channels=prior_feat, normalize=True)

        self.batch_norm_mu_pr = BatchNorm1d(prior_feat, momentum=0.9, track_running_stats=False)
        self.batch_norm_sigma_pr = BatchNorm1d(prior_feat, momentum=0.9, track_running_stats=False)        

        # decoder
        self.decoder = GraphUNet(
            in_channels = encoder_feat + posterior_feat, 
            hidden_channels = encoder_feat, 
            out_channels = out_feat, depth = gd, pool_ratios = 0.5
        )
               
    
    def forward(self, data, device=None):
        
        A_coo, y, x = data.edge_index, data.y, data.x
        x_encoded = self.encoder(x, A_coo)

        # posterior (q)
        y_post = self.posterior(y, A_coo)

        mu = self.conv_mu(y_post, A_coo)
        mu = torch.flatten(self.batch_norm_mu(mu))

        logstd = 0.1 * self.conv_sigma(y_post, A_coo)
        logstd = torch.flatten(self.batch_norm_sigma(logstd))
                
        # reparameterization  
        q = mu + torch.randn_like(logstd) * torch.exp(logstd)       
        
        # prior (p)
        y_prior = self.prior(x, A_coo)

        mu_pr = self.conv_mu_pr(y_prior, A_coo)
        mu_pr = torch.flatten(self.batch_norm_mu_pr(mu_pr))

        logstd_pr = 0.1 * self.conv_sigma_pr(y_prior, A_coo)
        logstd_pr = torch.flatten(self.batch_norm_sigma_pr(logstd_pr))
        
        # reparameterization 
        p = mu_pr + torch.randn_like(logstd_pr) * torch.exp(logstd_pr)     
                        
        if self.training:
            z = torch.cat((x_encoded, q.reshape(x_encoded.shape)), 1)
        else:
            z = torch.cat((x_encoded, p.reshape(x_encoded.shape)), 1)

        y_hat = self.decoder(z, A_coo)
        y_hat = torch.sigmoid(y_hat)
        
        return y_hat, (mu, logstd, mu_pr, logstd_pr)
    
    

class CVAE_UNET_Batch(CVAE_UNET_Pr):
    
    def forward(self, data, device=None):
        
        A_coo, y, x = data.edge_index, data.y, data.x
        
        batch = torch.repeat_interleave(input=torch.arange(x.shape[0]//self.M, device=x.device), repeats=self.M)
#         batch=None
 
        x_encoded = self.encoder(x, A_coo, batch=batch)

        # posterior (q)
        y_post = self.posterior(y, A_coo, batch=batch)

        mu = self.conv_mu(y_post, A_coo)
        mu = torch.flatten(self.batch_norm_mu(mu))

        logstd = 0.1 * self.conv_sigma(y_post, A_coo)
        logstd = torch.flatten(self.batch_norm_sigma(logstd))
                
        # reparameterization 
        q = mu + torch.randn_like(logstd) * torch.exp(logstd)
        
        # prior (p)
        y_prior = self.prior(x, A_coo, batch=batch)

        mu_pr = self.conv_mu_pr(y_prior, A_coo)
        mu_pr = torch.flatten(self.batch_norm_mu_pr(mu_pr))

        logstd_pr = 0.1 * self.conv_sigma_pr(y_prior, A_coo)
        logstd_pr = torch.flatten(self.batch_norm_sigma_pr(logstd_pr))
        
        # reparameterization 
        p = mu_pr + torch.randn_like(logstd_pr) * torch.exp(logstd_pr)
                        
        if self.training:
            z = torch.cat((x_encoded, q.reshape(x_encoded.shape)), 1)
        else:
            z = torch.cat((x_encoded, p.reshape(x_encoded.shape)), 1)

        y_hat = self.decoder(z, A_coo, batch=batch)
        y_hat = torch.sigmoid(y_hat)
        
        return y_hat, (mu, logstd, mu_pr, logstd_pr)
    
    
    

    