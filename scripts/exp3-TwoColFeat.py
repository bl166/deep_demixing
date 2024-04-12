import os
import shutil
import argparse
import numpy as np
import torch
#torch.backends.cudnn.benchnark=True
import torch.optim as optim
from torch.nn import BCELoss

PROJ_RT = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # SET YOUR PROJECT ROOT DIR HERE
DATA_PATH = os.path.join(PROJ_RT, "datasets") # CHANGE THIS TO YOUR DATASET LOCATION
import sys; sys.path.append(PROJ_RT)

# my func
from config import set_params
from utils import load_dataset, KLDLoss, NILReg, parallalize_model
from trainer import GCVAE_Trainer, Benchmark_Trainer
from models import CVAE_UNET_Batch, VAE_UNET_Batch, MLP_benchmark, CNN_time_benchmark, CNN_nodes_benchmark, GRU_benchmark, LSTM_benchmark, LSTM_bi_benchmark, CNN_time2_benchmark, CNN_time3_benchmark

# cpu or gpu
n_gpu = torch.cuda.device_count()
device = torch.device('cuda:0' if n_gpu else 'cpu')
pa = n_gpu > 1


parser = argparse.ArgumentParser()

parser.add_argument('--run',         type=int, default=None)
parser.add_argument('--seed',         type=int, default=None)
parser.add_argument('--aggregation', type=str, default='avg+acc')
parser.add_argument('--nodes',       type=int, default=100)
parser.add_argument('--ntrain',      type=int, default=1000)
parser.add_argument('--timesteps',   type=int, default=10)
parser.add_argument('--topology',    type=str, default='RG')
parser.add_argument('--observation', type=str, default='SIRS')

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--bs', type=int, default=64)


if __name__ == '__main__':

    args = parser.parse_args()

    # ------ 0. CONFIGURATIONS -------

    epochs        = args.epochs    # total epochs to run
    save_freq     = 5     # save model every xx epochs
    batch_size    = args.bs     # training mini-batch size
    l2            = 5e-4  # l2 normalization coefficient
    learning_rate = 1e-3  # learning rate (old values: 1e-2 for CVAE, 1e-4 for RNN, 1e-3 for other benchmarks)
    auto_stop     = 5    # terminate training if val loss stops going down for this many epochs

    rand = args.seed       # which random dataset
    run  = args.run        # number of run
    T    = args.timesteps  # number of timesteps
    N    = args.nodes      # training nodes

    g_type  = args.topology      # graph topology
    ob_type = args.observation   # observation model type
    gt_type = args.aggregation   # ground truth type
    graph_param,_ = set_params(g_type, n=N)

    # hidden feature sizes for encoder/decoder and prior/posterior networks
    # here they are equal to steps, but can be changed to arbitrary numbers
    encoder_feat   = T
    posterior_feat = T


    # ------ 1. LOAD DATASETS -------

    # this should point to your test set
    filepath = DATA_PATH + f"/{ob_type:s}_average_{N:d}nodes_20steps_1000sims_" \
                         + str(graph_param) \
                         + f"gparam_{g_type:s}_{rand:d}seed.obj"
    N_train  = 1000   # this is actually "N_test"
    N_val    = 0      # this doesnt matter
    test_loader, _ = load_dataset(
        filepath, N_train, N_val, batch_size=N_train, 
        T=T, device=None, shuffle_data=False, aggreg=gt_type, parallel=pa
    )

    # this should point to your train (and validation) sets
    filepath = filepath.replace('1000sims', '5000sims')
    N_train  = args.ntrain 
    N_val    = 1000
    train_loader, validation_loader = load_dataset(
        filepath, N_train, N_val, batch_size, shift=run*N_train, 
        T=T, device=None, shuffle_data=True, aggreg=gt_type, parallel=pa
    )


    # ------ 2. DECLARE LOSS FUNCTIONS AND MODELS -------

    # lookup dict of criteria to use
    criterions_all = {
        'kld': KLDLoss(reduction='sum', device_ids=list(range(n_gpu)), output_device=None), # TODO: check its mem use
        'bce': BCELoss(reduction='sum'),
        'rule': NILReg()
    }

    # initialize model and decide whether to do parallel training
    infeats = len(gt_type.split('+'))
    models = [
#         CNN_time3_benchmark(T, N, feat_in=2),
#         CNN_nodes_benchmark(T, N, feat_in=2),
#         MLP_benchmark(T, N, feat_in=2),
#         LSTM_bi_benchmark(T, N, feat_in=2, depth=2),
        VAE_UNET_Batch(encoder_feat, posterior_feat, T, N, depth=2, encoder_feat_in=infeats, prior_feat_in=infeats),
#         CVAE_UNET_Batch(encoder_feat, posterior_feat, T, N, depth=2, encoder_feat_in=infeats, prior_feat_in=infeats),
    ]


    # ------ 3. START TRAINING EACH MODEL -------

    for MODEL in models: 

        # process model name string: e.g. CVAE_UNET_Batch --> CVAE_UNET
        m_str = '_'.join(type(MODEL).__name__.split('_')[:-1])

        # find the right trainer: GCVAE_Trainer or Benchmark_Trainer
        TRAINER = GCVAE_Trainer if "VAE" in m_str else Benchmark_Trainer 
        criterion = criterions_all if "CVAE" in m_str else {k:v for k,v in criterions_all.items() if 'kld'!=k}

        print(f"\nmodel={m_str}; trainer={TRAINER.__name__}\nLet's use {n_gpu} GPUs!")

        # instantiate the model (and parallalize it when required)
        model = parallalize_model(MODEL, device_ids=(pa and list(range(n_gpu))))
        model.to(device)

        # define optimization method
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay = l2
        )
        # set the folder path to save models, checkpoints, and training logs 
        save_path = PROJ_RT + "/models_j/seed_{rd}/run_{run}/".format(rd = rand, run = run) \
            + "{ms:s}_{topo:s}_{ob:s}_{gt:s}_".format(ms = m_str, topo = g_type, ob=ob_type, gt=gt_type) \
            + "{ns:d}sims_{T:d}timesteps_{ef:d}encfeat_{pf:d}postfeat_{bs:d}batch_{lr:.2e}lr_{l2:.2e}l2/".format(
                ns = N_train+N_val, T = T, ef = encoder_feat, pf = posterior_feat, 
                bs = batch_size, lr = learning_rate, l2 = l2) 
        os.makedirs(save_path, exist_ok=True)
        check_path = save_path+"model.pt"

        # save this script 
        shutil.copy2(__file__, save_path)

        # construct the trainer 
        trainer = TRAINER(
            model      = model, 
            check_path = check_path, 
            optimizer  = optimizer, 
            resume     = True, # whether or not to start from a latest checkpoint if found
#             l2         = l2,   # l2 regularization coefficient 
            gradient_clipping = 5,# if 'LSTM' in m_str else None,
            display_step = 100
        )

        # start training, and save the best model (lowest loss)
        try:
            if not trainer.epoch:
                # initial validation and test results 
                trainer.predict(epoch=-1, criterion=criterion, loader=validation_loader, save=False, log=True, phase='val')
                trainer.predict(epoch=-1, criterion=criterion, loader=test_loader, save=False, log=True)

            for epoch in range(trainer.epoch, epochs):
                trainer.train(epoch=epoch, criterion=criterion, loader=train_loader)
                trainer.predict(epoch=epoch, criterion=criterion, loader=validation_loader, save=True, log=True)
                trainer.predict(epoch=epoch, criterion=criterion, loader=test_loader, save=False, log=True)

                if not (epoch+1)%save_freq:
                    trainer._save_latest(True)

                # autostop
                if hasattr(trainer, 'stop') and trainer.stop:
                    print('Terminating because the loss hasn\'t been reducing for a while ...')
                    break

        except (KeyboardInterrupt, SystemExit):
            trainer._save_latest()

        trainer._save_latest()

