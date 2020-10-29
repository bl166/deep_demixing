import torch
torch.backends.cudnn.benchnark=True

import torch.optim as optim
import numpy as np
from torch.nn import MSELoss, BCELoss
import os

# SET YOUR PROJECT ROOT DIR HERE
PROJ_RT = "/root/Deep_demixing/"
import sys; sys.path.append(PROJ_RT)

from utils import load_dataset, calculate_metric
from trainer import GCVAE_Trainer, Benchmark_Trainer
from models import *
from utils import KLDLoss, NILReg, NILWithLogitsReg, parallalize_model

# ------ CONFIGURATIONS -------

epochs = 50           # total epochs to run
save_freq = 5        # save model every x epochs
batch_size = 4        # training mini-batch size
l2 = 1e-6             # l2 normalization coefficient
learning_rate = .01   # learning rate

exp = 10               # experiment number 
Ts = [20]                # steps
max_nodes = 136        # training nodes

# if more than 2 gpu, do parallel
n_gpu = torch.cuda.device_count() 
device = torch.device('cuda' if n_gpu else 'cpu')
pa = False#n_gpu > 1
 

for T in Ts:
    encoder_feat = T      # hidden feature size, equal to steps
    posterior_feat = T    
    
    # ------ LOAD DATASETS -------

    filepath = PROJ_RT+"datasets/exp_%d/%d_steps/REAL_wCLS_A_Excl15T_5edgethresh_%dnodes_20steps_1000sims.obj"%(exp,T,max_nodes)
    
    N_train = 600 
    N_val = 100

    train_loader, validation_loader = load_dataset(
        filepath, N_train, N_val, batch_size, device=None, shuffle_data=True, parallel=pa, shift=0)
    

    # lookup dict of criterion to use
    criterion = {
        'kld': KLDLoss(reduction='sum', device_ids=list(range(n_gpu)), output_device=None),
        'bce': BCELoss(reduction='sum'),
        'rule': NILReg()
    }

    # ------ MODEL AND TRAIN -------
    
    # initialize model and decide whether to do parallel training
    models = [
              CNN_time_benchmark(T, max_nodes),
              CNN_nodes_benchmark(T, max_nodes),
              CVAE_UNET_Batch(encoder_feat, posterior_feat, T, max_nodes)
             ]
    
    trainers = [Benchmark_Trainer]*2 + [GCVAE_Trainer]

    
    for MODEL, TRAINER in zip(models, trainers):
        m_str = '_'.join(type(MODEL).__name__.split('_')[:-1])
        
        print('\n\n', m_str, TRAINER.__name__, '\n\n')

        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = parallalize_model(MODEL, device_ids=(pa and list(range(n_gpu))))

        model.to(device)
        
        # optimization method
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # set model save path
        save_path = PROJ_RT+"models/exp_%d/%d_steps/"%(exp, T)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        check_path = save_path+"%s_REAL_%dsims_%dbatch_%.2elr_%.2el2/model.pt"%(
            m_str, (N_train+N_val), batch_size, learning_rate, l2)

        # construct the trainer
        trainer = TRAINER(model, check_path, optimizer, resume=True, l2=l2, max_steps=T, display_step=50)

        # start training, and save the best model (lowest loss)
        try:
            for epoch in range(trainer.epoch, epochs):
                # train(self, epoch, criterion, loader)
                trainer.train(epoch=epoch, criterion=criterion, loader=train_loader)

                # predict(self, epoch, criterion, loader, save)
                trainer.predict(epoch=epoch, criterion=criterion, loader=validation_loader, save=True)

                if not (epoch+1)%save_freq:
                    trainer._save_latest(True)

                # autostop
                if hasattr(trainer, 'stop') and trainer.stop:
                    print('Terninating because the loss hasn\'t been reducing for a while ...')
                    break

#             # test
#             trainer.predict(epoch=epoch, criterion=criterion, loader=test_loader, save=False)

        except (KeyboardInterrupt, SystemExit):
            trainer._save_latest()

        trainer._save_latest()


