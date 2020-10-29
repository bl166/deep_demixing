import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Dropout, ReLU, Linear, Conv1d, ConvTranspose1d
from torch_geometric.nn import GCNConv, GraphUNet 
import numpy as np


class MLP_benchmark(nn.Module):
    
    def __init__(self, T, max_nodes):
        super(MLP_benchmark, self).__init__()

        self.N = max_nodes
        hidden_size = self.N * int(np.ceil(T/4))
        
        self.linears = nn.Sequential(
            Linear(self.N, hidden_size),
            Dropout(0.5),
            ReLU(),
            Linear(hidden_size, hidden_size),
            Dropout(0.5),
            ReLU(),
            Linear(hidden_size, self.N * T),
            Dropout(0.5),
            ReLU(),
            Linear(self.N * T, self.N * T)
        )
        
            
    def forward(self, data, device=None):
        if isinstance(data, torch.Tensor):
            x = data
        else:
            x = data.x
        x = x.reshape((-1, self.N))
        
        x = self.linears(x)
        x = torch.sigmoid(x)
        
        return x

    
class CNN_time_benchmark(nn.Module):

    def __init__(self, T, max_nodes):
        super(CNN_time_benchmark, self).__init__()

        # Layer 0: Nx1x1 -> Nx1x1 => kernel_size = 1
        self.conv_0 = Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1)
        self.batch_norm_0 = BatchNorm1d(1, track_running_stats = False)
        self.relu_0 = ReLU()

        # Layer 1: Nx1x1 -> Nx1x (T/4) => kernel_size = (T/4)
        self.conv_1 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(np.ceil(T/4)))
        self.batch_norm_1 = BatchNorm1d(1, track_running_stats = False)
        self.relu_1 = ReLU()

        # Layer 2: Nx1x (T/4) -> Nx1x (T/4) => kernel_size = 1
        self.conv_2 = Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1)
        self.batch_norm_2 = BatchNorm1d(1, track_running_stats = False)
        self.relu_2 = ReLU()

        # Layer 3: Nx1x (T/4) -> Nx1x (T/2) => kernel_size = (T/2) - (T/4) + 1
        self.conv_3 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(np.ceil(T/2)) - int(np.ceil(T/4)) + 1)
        self.batch_norm_3 = BatchNorm1d(1, track_running_stats = False)
        self.relu_3 = ReLU()

        # Layer 4: Nx1x (T/2) -> Nx1x (T/2) => kernel_size = 1
        self.conv_4 = Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1)
        self.batch_norm_4 = BatchNorm1d(1, track_running_stats = False)
        self.relu_4 = ReLU()

        # Layer 4_1: Nx1x (T/2) -> Nx1x (T/2) => kernel_size = 1
        self.conv_4_1 = Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1)
        self.batch_norm_4_1 = BatchNorm1d(1, track_running_stats = False)
        self.relu_4_1 = ReLU()

        # Layer 5: Nx1x (T/2) -> Nx1x (T) => kernel_size = (T) - (T/2) + 1
        self.conv_5 = ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = int(T) - int(np.ceil(T/2)) + 1)
        self.batch_norm_5 = BatchNorm1d(1, track_running_stats = False)
        self.relu_5 = ReLU()

        # Layer 6: Nx1x (T) -> Nx1x (T) => kernel_size = 1
        self.conv_6 = Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1)
        self.batch_norm_6 = BatchNorm1d(1, track_running_stats = False)
        self.relu_6 = ReLU()
            
            
    def forward(self, data, device=None):
        
        if isinstance(data, torch.Tensor):
            x = data
        else:
            x = data.x
        x = torch.reshape(x, (x.shape[0], 1, 1))
        
        # Layer 0:
        x_0 = self.conv_0(x)
        x_0 = self.batch_norm_0(x_0)
        x_0 = self.relu_0(x_0)

        # Layer 1:
        x_1 = self.conv_1(x)
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

        # Layer 4:
        x_4_1 = self.conv_4_1(x_4)
        x_4_1 = self.batch_norm_4_1(x_4_1)
        x_4_1 = self.relu_4_1(x_4_1)

        # Layer 5:
        x_5 = self.conv_5(x_4_1)
        x_5 = self.batch_norm_5(x_5)
        x_5 = self.relu_5(x_5)

        # Layer 6:
        x_6 = self.conv_6(x_5)
        x_6 = self.batch_norm_6(x_6)
        x_6 = torch.sigmoid(x_6)

        out = x_6
        out = torch.reshape(out, (out.shape[0], out.shape[2]))

        return out


class CNN_nodes_benchmark(nn.Module):

    def __init__(self, T, max_nodes):
        super(CNN_nodes_benchmark, self).__init__()


        # Layer 0: 1x1xN -> 1x1xN
        self.conv_0 = Conv1d(in_channels = 1, out_channels = 1, kernel_size = 3, padding = 1)
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

        # Layer 4_1: 1x(T/2)xN -> 1x(T/2)xN
        self.conv_4_1 = Conv1d(in_channels = int(np.ceil(T/2)), out_channels = int(np.ceil(T/2)), kernel_size = 3, padding = 1)
        self.batch_norm_4_1 = BatchNorm1d(int(np.ceil(T/2)), track_running_stats = False)
        self.relu_4_1 = ReLU()

        # Layer 5: 1x(T/2)xN -> 1x(T)xN
        self.conv_5 = Conv1d(in_channels = int(np.ceil(T/2)), out_channels = int(T), kernel_size = 3, padding = 1)
        self.batch_norm_5 = BatchNorm1d(int(T), track_running_stats = False)
        self.relu_5 = ReLU()

        # Layer 6: 1x(T)xN -> 1x(T)xN
        self.conv_6 = Conv1d(in_channels = int(T), out_channels = int(T), kernel_size = 3, padding = 1)
        self.batch_norm_6 = BatchNorm1d(int(T), track_running_stats = False)
        self.relu_6 = ReLU()
        
    
    def forward(self, data, device=None):

        if isinstance(data, torch.Tensor):
            x = data
        else:
            x = data.x
        x = torch.reshape(x, (1, 1, x.shape[0]))

        # Layer 0:
        x_0 = self.conv_0(x)
        x_0 = self.batch_norm_0(x_0)
        x_0 = self.relu_0(x_0)

        # Layer 1:
        x_1 = self.conv_1(x)
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

        # Layer 4:
        x_4_1 = self.conv_4_1(x_4)
        x_4_1 = self.batch_norm_4_1(x_4_1)
        x_4_1 = self.relu_4_1(x_4_1)

        # Layer 5:
        x_5 = self.conv_5(x_4_1)
        x_5 = self.batch_norm_5(x_5)
        x_5 = self.relu_5(x_5)

        # Layer 6:
        x_6 = self.conv_6(x_5)
        x_6 = self.batch_norm_6(x_6)
#         x_6 = self.final_activ(x_6)
        x_6 = torch.sigmoid(x_6) # final layer should be between 0,1 to use bce

        out = x_6
        out = torch.reshape(out, (out.shape[1], out.shape[2]))
        out = torch.transpose(out, 0, 1)

        return out
    

class CVAE_UNET_Pr(nn.Module):

    def __init__(self, encoder_feat, posterior_feat, T, max_nodes):
        super(CVAE_UNET_Pr, self).__init__()
        
        # hyperparams
        gd = 1 # depth of the graph unet
        self.M = max_nodes
        self.T = T
        prior_feat = posterior_feat

        self.encoder = GraphUNet(in_channels = 1, hidden_channels = encoder_feat, out_channels = encoder_feat, depth = gd, pool_ratios = 0.5)

        self.posterior = GraphUNet(in_channels = T, hidden_channels = posterior_feat, out_channels = posterior_feat, depth = gd, pool_ratios = 0.5)
        self.prior = GraphUNet(in_channels = 1, hidden_channels = prior_feat, out_channels = prior_feat, depth = gd, pool_ratios = 0.5)

        # mu and sigma for posterior net
        self.conv_mu = GCNConv(in_channels = posterior_feat, out_channels = posterior_feat, normalize = True)
        self.conv_sigma = GCNConv(in_channels = posterior_feat, out_channels = posterior_feat, normalize = True)

        self.batch_norm_mu = BatchNorm1d(posterior_feat, momentum=0.9, track_running_stats=False)
        self.batch_norm_sigma = BatchNorm1d(posterior_feat, momentum=0.9, track_running_stats=False)
        
        # mu and sigma for prior net
        self.conv_mu_pr = GCNConv(in_channels = prior_feat, out_channels = prior_feat, normalize = True)
        self.conv_sigma_pr = GCNConv(in_channels = prior_feat, out_channels = prior_feat, normalize = True)

        self.batch_norm_mu_pr = BatchNorm1d(prior_feat, momentum=0.9, track_running_stats=False)
        self.batch_norm_sigma_pr = BatchNorm1d(prior_feat, momentum=0.9, track_running_stats=False)        

        self.decoder = GraphUNet(in_channels = 2 * T, hidden_channels = T, out_channels = T, depth = gd, pool_ratios = 0.5)
               
    
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
    
    