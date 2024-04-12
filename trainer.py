import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
# PyTorch logger tutorial: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from tqdm.auto import tqdm


import builtins as __builtin__
def print(*args, **kwargs):
    # My custom print() function: Overload print function to get time logged
    __builtin__.print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), end = ' | ')
    return __builtin__.print(*args, **kwargs)


# Save model with epoch number and validation error
def save_network(model, epoch, err, save_path):
    # deal with multi-gpu parallelism
    pa = type(model)
    if pa.__name__ == 'DataParallel':
        net_save = model.module
    else:
        net_save = model
    # format to save
    state = {
        'net': net_save,
        'err': err, # error rate or negative loss (something to minimize)
        'epoch': epoch
    }
    torch.save(state, save_path)


# Load model with epoch number and validation error
def load_network(network, net_path):

    if network: # if the given template network is not none

        check_pt = torch.load(net_path)

        if 'net' in check_pt:
            checkpoint = check_pt['net']
            epoch = check_pt['epoch']
            err = check_pt['err']
        else:
            checkpoint = check_pt
            epoch = -1
            err = np.nan

        # assert checkpoint should be a nn.module or parallel nn.module
        pa_0 = type(checkpoint)
        if pa_0.__name__ == 'DataParallel':
            checkpoint = checkpoint.module
        else:
            pass

        new_dict = checkpoint.state_dict()

        # same to the network template
        pa_1 = type(network)
        if pa_1.__name__ == 'DataParallel':
            network = network.module

        old_dict = network.state_dict()

        # make sure new and old state_dict match each other
        new_keys = set(new_dict.keys())
        old_keys = set(old_dict.keys())

        if old_keys == new_keys:
            network.load_state_dict(new_dict)

        else:
            if new_keys-old_keys:
                warnings.warn("Ignoring keys in new dict: {}".format(new_keys-old_keys))
            if old_keys-new_keys:
                warnings.warn("Missing keys in old dict: {}".format(old_keys-new_keys))

            # filter out unnecessary keys
            new_dict = {k: v for k, v in new_dict.items() if k in old_dict}
            # overwrite entries in the existing state dict
            old_dict.update(new_dict)
            # load the new state dict
            network.load_state_dict(old_dict)

        # if network used to be DataParallel, now it's time to convert it back to DP
        if pa_0.__name__ == 'DataParallel':
            network = pa_0(network)
        elif pa_1.__name__ == 'DataParallel':
            network = pa_1(network)

        return network, epoch, err

    else: # if not given any network template

        checkpoint = torch.load(net_path)

        if 'net' in checkpoint:
            network = checkpoint['net']
            epoch = checkpoint['epoch']
            err = checkpoint['err']
        else:
            network = checkpoint
            epoch = -1
            err = np.nan

        return network, epoch, err


# ----------------------------------
# -------- Trainer Wrapper ---------
# ----------------------------------

class _trainer(object):
    def __init__(self, model, check_path, optimizer, resume=True, **kwargs):
        super().__init__()

        # training configurations
        self.bs = None # placeholder for batch_size
        self.ds = kwargs['display_step'] if 'display_step' in kwargs.keys() else 10
        self.gc = kwargs['gradient_clipping'] if 'gradient_clipping' in kwargs.keys() else None
        self.cw = kwargs['cweights'] if 'cweights' in kwargs.keys() else None  # criterion weights
        self.cp = check_path

        self.optimizer = optimizer
        self.global_step = 0

        if model:
            self.model, self.epoch, self.track_minimize = model, 0, np.inf
        else:
            self.model, self.epoch, self.track_minimize = load_network(model, self.cp)

        print(self.model)
        self.device = next(self.model.parameters()).device

        # learning rate scheduler
        if 'decay_step' in kwargs.keys() and kwargs['decay_step']:
            self.lr_scheduler = MultiStepLR(self.optimizer,
                                            milestones=kwargs['decay_step'][0],
                                            gamma=kwargs['decay_step'][1])
        else:
            self.lr_scheduler = None

        # l2 regularization
        if 'l2' in kwargs.keys() and kwargs['l2']:
            self.l2 = kwargs['l2']
        else:
            self.l2 = 0
            
        # num steps to recover "T"
        if 'max_steps' in kwargs.keys() and kwargs['max_steps']:
            self.max_steps = kwargs['max_steps']
        else:
            self.max_steps = np.inf

        # tensorboard
        self.logger = SummaryWriter(self.cp+'-logs') if self.cp else None

        # autostop when loss doesnt go down for 10 epochs
        if 'auto_stop' in kwargs.keys() and kwargs['auto_stop']:
            self.autostop = kwargs['auto_stop']
        else:
            self.autostop = 10
        self.trackstop = 0
        self.stop = False

        if resume:
            self._resume()


    def _save_latest(self, desc=None):
        if not desc:
            desc = "-latest"
        else:
            desc = "-ep%d"%self.epoch
        # save latest checkpoint upon exit
        # http://effbot.org/zone/stupid-exceptions-keyboardinterrupt.htm
        save_network(self.model, self.epoch, self.track_minimize, self.cp+desc)
        print("Saving to", self.cp+desc, "...")

    # alias of _save_latest
    def save_checkpoint(self, desc=None):
        return self._save_latest(desc)

    # Save & restore model
    def save(self, epoch):
        # normal save
        save_network(self.model, epoch, self.track_minimize, self.cp)
        print("Saving to", self.cp, "...")

    # Externally: restore the best epoch
    def restore(self):
        assert self.model is not None, "Must initialize a model!"
        if os.path.exists(self.cp):
            self.model, self.epoch, self.track_minimize = load_network(self.model, self.cp)
            print("Loading from", self.cp, "at epoch", self.epoch,"...")

    # Internally: resume from the latest checkpoint
    def _resume(self):
        res_path = self.cp+'-latest'
        if os.path.exists(res_path):
            self.model, self.epoch, self.track_minimize = load_network(self.model, res_path)
            print("Resuming training from", res_path, "at epoch", self.epoch,"...")
        else:
            print("Starting fresh at epoch 0 ...")


    def train(self, epoch, loader):
        raise NotImplemented

    def predict(self):
        raise NotImplemented

    # Evaluate performance
    def calculate_metric(self, loader, metric='mse'):
        y_true, y_hat, A_coo = self.predict(self.epoch, criterion=None, loader=loader, save=False)

        if metric == 'mse':
            m = np.mean((y_true- y_hat)**2)
        elif metric == 'auc':
            from sklearn import metrics
            m = metrics.roc_auc_score(y_true.flatten(), y_hat.flatten())
        else:
            raise NotImplemented

        return m, (y_true, y_hat, A_coo)


    # tensorboard logging
    def logging(self, info, phase):
        universal_step = self.global_step*self.bs
        if phase == 'train':
            # Log scalar values (scalar summary)
            for k,v in info.items():
                self.logger.add_scalar('%s/train'%k, v, universal_step)

            # Log values and gradients of the parameters (histogram summary)
            for tag, value in self.model.named_parameters():
                if 1:#'enc' in tag.lower() or 'lstm' in tag.lower():
                    try:
                        tag = '.'.join(tag.split('.')[:-1])+'/'+tag.split('.')[-1]
                        self.logger.add_histogram(tag, value.data.cpu().numpy(), universal_step)
                        self.logger.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), universal_step)
                    except:
                        continue
        else:
            ep = info['epoch'] if 'epoch' in info else self.epoch
            for k,v in info.items():
                if k != 'epoch':
                    self.logger.add_scalar('%s/%s'%(k,phase), v, ep)

    # load data and labels
    def data_handler(self, data):
        if isinstance(data, list):
            [d.to(self.device) for d in data]
            for dd in range(len(data)): # update x steps
                data[dd].y = data[dd].y[...,:self.T]
            x_true = torch.cat([d.x for d in data],0)
            y_true = torch.cat([d.y for d in data],0)
            A_coo = torch.cat([d.edge_index for d in data],0)
        else:
            data.to(self.device)
            data.y = data.y
            x_true = data.x
            y_true = data.y[...,:self.T]
            A_coo = data.edge_index

        # if 'CVAE', considering both parallel/nonparallel model cases
        try:
            assert 'VAE' in self.model.__class__.__name__ or 'VAE' in self.model.module.__class__.__name__
            return data, y_true, A_coo
        except:
            return x_true, y_true, A_coo

    # returns the numbers of nodes and steps.
    def calc_numbers(self, loader):
        max_nodes = loader.dataset[0].x.shape[0]
        self.T = min(loader.dataset[0].y.shape[1], self.max_steps)
        self.bs = loader.batch_size
        n = max_nodes*self.bs*self.T
        return max_nodes, n


# ----------------------------------------------
# ---- Graph CVAE Trainer (classification) ----
# ----------------------------------------------

class GCVAE_Trainer(_trainer):

    # Train & test functions for a single epoch

    def train(self, epoch, criterion, loader):
        torch.cuda.empty_cache()
        
        #set default cweights
        if self.cw is None:
            self.cw = {k:1 for k in criterion.keys()}

        # get nodes and steps
        max_nodes , n = self.calc_numbers(loader)

        # training logs
        trn_log = {'loss_%s'%k:[] for k in criterion.keys()} # kld/bce/mse loss
        trn_log['loss'] = [] # entire loss

        self.epoch = epoch
        self.global_step = epoch*len(loader) if not self.global_step else self.global_step

        # set to train mode
        self.model.train()

        with tqdm(loader, desc='Epoch#%d:'%epoch) as pbar:

            count_b = 0

            for data in pbar:
                if len(data) <= 1:
                    continue # for batchnorm the batchsize has to be greater than 1

                self.optimizer.zero_grad()
                
                data, y_true, A_coo = self.data_handler(data)

                # p (prior) and q (post) are multi-var gaussian distributions
                y_hat, (mu_q, sigma_q, mu_p, sigma_p)= self.model(data)

                losses = {}
                if 'kld' in criterion:
                    losses['kld'] = criterion['kld'](mu_q, sigma_q, mu_p, sigma_p)/n                  

                losses['bce']  = criterion['bce'](y_hat, y_true)/n
                losses['rule'] = criterion['rule'](y_hat, A_coo, num_nodes=max_nodes)/n
                
                loss = 0
                for k,w in self.cw.items():
                    loss += w*losses[k]

                if self.l2: # l2 regularization
                    l2_reg = torch.tensor(0.).to(self.device)
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param)
                    loss += self.l2 * l2_reg

                loss.backward()

                if self.gc is not None: # gradient clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gc)
                self.optimizer.step()

                trn_log['loss'] += [loss.item()]
                for k,l in losses.items(): 
                    trn_log[f'loss_{k}'] += [l.item()]

                count_b += 1

                if not (self.global_step) % self.ds:
                    # record the last step
                    info = {k:v[-1] for k,v in trn_log.items() if v}
                    self.logging(info, 'train')

                    pbar.write('Train step {} at epoch {}: '.format(self.global_step, self.epoch) +
                               ', '.join(['{}: {:.6f}'.format(k, np.mean(v[-count_b:])) for k,v in trn_log.items()]))

                    count_b = 0

                # update global step
                self.global_step += 1

        # if increment, reset loss etc. information
        print('====> Epoch %d: Average training '%self.epoch +
              ', '.join(['{}: {:.6f}'.format(k, np.sum(v)/len(loader)) for k,v in trn_log.items()]))
        
    def predict(self, epoch, criterion=None, loader=None, **kwargs):
        
        if self.cw is None:
            self.cw = {k:1 for k in criterion.keys()}

        # has to be a valid dataloader
        assert loader is not None

        # calculate n from dataloader
        max_nodes , n = self.calc_numbers(loader)

        lflag = 'log' not in kwargs or kwargs['log'] # whether to log to tensorboard
        sflag = 'save' in kwargs and kwargs['save']  # whether to save the best model
        if 'phase' in kwargs:
            phase = kwargs['phase'] 
        else:
            phase = 'val' if sflag else 'test'

        self.model.eval()

        # prediction
        y_pr, y_gt, A_eidx = [],[],[]

        # track the loss
        if criterion:
            trn_log = {'loss_%s'%k:0 for k in criterion.keys()} # kld/bce/mse loss
            trn_log['loss'] = 0

        with torch.no_grad():
            with tqdm(loader, disable=1==len(loader)) as pbar:
                for data in pbar:
                    data, y_true, A_coo = self.data_handler(data)

                    # p (prior) and q (post) are multi-var gaussian distributions
                    y_hat, (mu_q, sigma_q, mu_p, sigma_p)= self.model(data)

                    # append true and predicted labels to lists
                    y_pr.append(y_hat.view((-1, max_nodes, self.T)))
                    y_gt.append(y_true.view((-1, max_nodes, self.T)))
                    #A_eidx.append(A_coo)

                    if criterion: # if given loss functions

                        kld = 0#criterion['kld'](mu_q, sigma_q, mu_p, sigma_p)/n
                        bce = criterion['bce'](y_hat, y_true)/n
                        rule = 0#criterion['rule'](y_hat, A_coo, num_nodes=max_nodes)/n

                        loss = self.cw['bce']*bce #+ self.cw['kld']*kld + self.cw['rule']*rule

                        trn_log['loss'] += loss.item()
                        trn_log['loss_bce'] += bce.item()


            # if criterion is given, we can track loss and save if best and needed;
            # if criterion is not given, just return the true and predicted values
            if criterion:

                info = {k:v/len(loader) for k,v in trn_log.items()  if v}

                print('====> %s set '%phase + ', '.join(['{}: {:.6}'.format(k,v) for k,v in info.items()]) )

                if lflag: # if validation, save the model if it results in a new lowest error/loss
                    info['epoch'] = epoch
                    self.logging(info, phase)
                    
                    if sflag:
                        track = info['loss'] #test_err
                        if track < self.track_minimize:
                            # update min err track
                            self.trackstop = 0
                            self.track_minimize = track

                            self.save(epoch) # save model
                        else:
                            self.trackstop += 1
                            if self.trackstop > self.autostop:
                                self.stop = True
                else: # if test, do not save
                    pass

                
        y_pr = torch.cat(y_pr).cpu().numpy()
        y_gt = torch.cat(y_gt).cpu().numpy()

        return y_gt, y_pr, None#A_eidx
        
        
# ----------------------------------------------
# ---- Benchmark Trainer (classification) ----
# ----------------------------------------------

class Benchmark_Trainer(_trainer):

    # Train & test functions for a single epoch.
    # Generally follows CVAE_Pr_Trainer excpet for not considering the variational loss and adjacency matrix

    def train(self, epoch, criterion=None, loader=None):
        torch.cuda.empty_cache()

        # calculate n
        max_nodes , n = self.calc_numbers(loader)

        # training logs
        trn_log = {'loss_%s'%k:[] for k in criterion.keys()}
        trn_log['loss'] = [] # entire loss

        self.epoch = epoch
        self.global_step = epoch*len(loader) if not self.global_step else self.global_step

        # set mode to train
        self.model.train()

        with tqdm(loader, desc='Epoch#%d:'%epoch) as pbar:

            count_b = 0

            for data in pbar:
                if len(data) <= 1:
                    continue # for batchnorm the batchsize has to be greater than 1

                self.optimizer.zero_grad()

                # load data and labels
                x_true, y_true, _ = self.data_handler(data)
                
                y_hat = self.model(x_true).reshape(y_true.shape)
                
                #print(y_true, y_hat)

                bce  = criterion['bce'](y_hat, y_true)/n
                loss = bce

                if self.l2: # l2 regularization
                    l2_reg = torch.tensor(0.).to(self.device)
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param)
                    loss += self.l2 * l2_reg

                loss.backward()

                if self.gc is not None: # gradient clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gc)
                self.optimizer.step()

                trn_log['loss'] += [loss.item()]
                trn_log['loss_bce'] += [bce.item()]

                count_b += 1

                if not (self.global_step) % self.ds:
                    # record the last step
                    info = {k:v[-1] for k,v in trn_log.items() if len(v)}
                    self.logging(info, 'train')

                    pbar.write('Train step {} at epoch {}: '.format(self.global_step, self.epoch) +
                               ', '.join(['{}: {:.6f}'.format(k, np.mean(v[-count_b:])) for k,v in trn_log.items()]))
                    count_b = 0

                # update global step
                self.global_step += 1

        # if increment, reset loss etc. information
        print('====> Epoch %d: Average training '%self.epoch +
              ', '.join(['{}: {:.6f}'.format(k, np.sum(v)/len(loader)) for k,v in trn_log.items()]))


    def predict(self, epoch, criterion=None, loader=None, **kwargs):

        # has to be a valid dataloader
        assert loader is not None

        # calculate n from dataloader
        max_nodes , n = self.calc_numbers(loader)

        lflag = 'log' not in kwargs or kwargs['log']
        sflag = 'save' in kwargs and kwargs['save']
        if 'phase' in kwargs:
            phase = kwargs['phase']
        else:
            phase = 'val' if sflag else 'test'

        self.model.eval()

        # prediction
        y_pr, y_gt, A_eidx = [],[],[]

        # track the loss
        if criterion:
            trn_log = {'loss_%s'%k:0 for k in criterion.keys()} # kld/bce/mse loss
            trn_log['loss'] = 0

        with torch.no_grad():
            with tqdm(loader, disable=1==len(loader)) as pbar:
                for data in pbar:

                    # load data and labels
                    x_true, y_true, _ = self.data_handler(data)

                    # predict
                    y_hat = self.model(x_true).reshape(y_true.shape)

                    # append true and predicted labels to lists
                    y_pr.append(y_hat.cpu().numpy().reshape((-1, max_nodes, self.T)))
                    y_gt.append(y_true.cpu().numpy().reshape((-1, max_nodes, self.T)))

                    if criterion: # if given loss functions

                        bce = criterion['bce'](y_hat, y_true)/n
                        loss = bce

                        if self.l2: # l2 regularization
                            l2_reg = torch.tensor(0.).to(self.device)
                            for param in self.model.parameters():
                                l2_reg += torch.norm(param)
                            loss += self.l2 * l2_reg

                        trn_log['loss'] += loss.item()
                        trn_log['loss_bce'] += bce.item()

            y_pr = np.concatenate(y_pr)
            y_gt = np.concatenate(y_gt)

            # if criterion is given, we can track loss and save if best and needed
            # if criterion is not given, just return the true and predicted values
            if criterion:
                info = {k:v/len(loader) for k,v in trn_log.items() if v}
                print('====> %s set '%phase + ', '.join(['{}: {:.6}'.format(k,v) for k,v in info.items()]) )

                if lflag: # if validation, save the model if resulting in a new lowest error/loss
                    info['epoch'] = epoch
                    self.logging(info, phase)
                    
                    if sflag:
                        track = info['loss'] #test_err
                        if track < self.track_minimize:
                            # update min err track
                            self.trackstop = 0
                            self.track_minimize = track

                            self.save(epoch) # save model
                        else:
                            self.trackstop += 1
                            if self.trackstop > self.autostop:
                                self.stop = True

                else: # if test, do not save
                    pass

            return y_gt, y_pr, A_eidx
