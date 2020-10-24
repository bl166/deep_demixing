import torch
torch.backends.cudnn.benchnark=True

import torch.optim as optim
import numpy as np
from torch.nn import MSELoss, BCELoss
import os


# PROJ_RT = "/root/Deep_demixing/"
PROJ_RT = "/Users/boningli/Desktop/Deep_demixing/"

import sys; sys.path.append(PROJ_RT)

from utils import load_dataset, calculate_metric
from trainer import GCVAE_Trainer, Benchmark_Trainer
from models import *#CVAE_UNET_Pr
from utils import KLDLoss, NILReg, NILWithLogitsReg, parallalize_model



# ------ CONFIGURATIONS -------

epochs = 50           # total epochs to run
save_freq = 1        # save model every x epochs
batch_size = 4        # training mini-batch size
l2 = 1e-6             # l2 normalization coefficient
learning_rate = .01   # learning rate
auto_stop = 5         # terminate training if val loss stops going down for this many epochs

exp = 1               # experiment number
Ts = [5,10,15,20]     # steps
max_nodes = 100       # training nodes

# if more than 2 gpu, do parallel
n_gpu = torch.cuda.device_count()
device = torch.device('cuda:0' if n_gpu else 'cpu')
pa = n_gpu > 1



for T in Ts:
    encoder_feat = T      # hidden feature size, equal to steps
    posterior_feat = T

    # ------ LOAD DATASETS -------

    # this should point to your test set
    filepath = PROJ_RT+"datasets/exp_%d/%d_steps/RG_%dnodes_%dsteps_1000sims.obj"%(exp, T, max_nodes, T)
    N_train = 1000 # this is actually "N_test"
    N_val = 0      # this doesnt matter

    test_loader, _ = load_dataset(
        filepath, N_train, N_val, batch_size, device=None, shuffle_data=False, parallel=pa)

    # this should point to your train (and validation) sets
    filepath = PROJ_RT+"datasets/exp_%d/%d_steps/RG_%dnodes_%dsteps_5000sims.obj"%(exp, T, max_nodes, T)
    N_train = 4500
    N_val = 500

    train_loader, validation_loader = load_dataset(
        filepath, N_train, N_val, batch_size, device=None, shuffle_data=True, parallel=pa)


    # lookup dict of criterion to use
    criterion = {
        'kld': KLDLoss(reduction='sum', device_ids=list(range(n_gpu)), output_device=None),
        'bce': BCELoss(reduction='sum'),
        'rule': NILReg()
    }

    # ------ MODEL AND TRAIN -------

    # initialize model and decide whether to do parallel training
    models = [CVAE_UNET_Batch(encoder_feat, posterior_feat, T, max_nodes),
              MLP_benchmark(T, max_nodes),
              CNN_time_benchmark(T, max_nodes),
              CNN_nodes_benchmark(T, max_nodes)
             ]

    trainers = [GCVAE_Trainer] + [Benchmark_Trainer]*3



    for MODEL, TRAINER in zip(models, trainers):
        m_str = '_'.join(type(MODEL).__name__.split('_')[:-1])

        print('\n\n', m_str, TRAINER.__name__, '\n\n')
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = parallalize_model(MODEL, device_ids=(pa and list(range(n_gpu))))
        model.to(device)

        # optimization method
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # set model save path
        save_path = PROJ_RT + "models/exp_%d/%d_steps/"%(exp, T)+\
                    "%s_RG_%dsims_%dtimesteps_%dbatch_%.2elr_%.2el2/"%(
                        m_str, (N_train+N_val), T, batch_size, learning_rate, l2)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        check_path = save_path+"model.pt"

        # construct the trainer
        trainer = TRAINER(model, check_path, optimizer, resume=True, l2=l2, display_step=10)

        # start training, and save the best model (lowest loss)
        try:
            for epoch in range(trainer.epoch, epochs):
                trainer.train(epoch=epoch, criterion=criterion, loader=train_loader)
                trainer.predict(epoch=epoch, criterion=criterion, loader=validation_loader, save=True)

                if not (epoch+1)%save_freq:
                    trainer._save_latest(True)

                # autostop
                if hasattr(trainer, 'stop') and trainer.stop:
                    print('Terninating because the loss hasn\'t been reducing for a while ...')
                    break

            # test
            trainer.predict(epoch=epoch, criterion=criterion, loader=test_loader, save=False)

        except (KeyboardInterrupt, SystemExit):
            trainer._save_latest()

        trainer._save_latest()
