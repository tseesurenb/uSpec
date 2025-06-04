'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Universal Spectral model for CF")

    # parser.add_argument('--lr', type=float,default=0.001, help="the learning rate")
    # parser.add_argument('--decay', type=float,default=1e-2, help="the weight decay for l2 normalizaton")
    # parser.add_argument('--testbatch', type=int,default=500, help="the batch size of users for testing")
    # parser.add_argument('--dataset', type=str,default='gowalla', help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    # parser.add_argument('--topks', nargs='?',default="[20]", help="@k test list")
    # parser.add_argument('--i_K', type=int,default=400)
    # parser.add_argument('--u_K', type=int,default=400)
    # parser.add_argument('--sim_K', type=int,default=40)
    # parser.add_argument('--s_temp', type=float,default=1.0)
    # parser.add_argument('--epochs', type=int,default=50)
    # parser.add_argument('--n_eigen', type=int,default=500)
    # parser.add_argument('--u_batch', type=int,default=1000, help='the batch size of users for training, -1 for full dataset')
    # parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    # parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    # parser.add_argument('--seed', type=int, default=2025, help='random seed')
    # parser.add_argument('--model', type=str, default='uspec', help='rec-model, support [cf, uspec]')
    # parser.add_argument('--m_type', type=str, default='single', help='sinle or double, single for only positive similarity, double for both positive and negative similarity')
    # parser.add_argument('--filter', type=str, default='u', help='u, i, and ui')
    # parser.add_argument('--filter_order', type=int, default=6, help='Number of polynomial order for spectral filters')
    # parser.add_argument('--verbose', type=int, default=0, help='0 for silent, 1 for verbose')
    # parser.add_argument('--a_fold', type=int,default=100, help="the fold num used to split large adj matrix, like gowalla")
    # parser.add_argument('--n_epoch_eval', type=int, default=3, help='number of epochs to evaluate the model during training')
    # parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of training data to use for validation')


    parser.add_argument('--lr', type=float,default=0.001, help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-2, help="the weight decay for l2 normalizaton")
    parser.add_argument('--train_u_batch_size', type=int,default=1000, help='batch size for training users, -1 for full dataset')
    parser.add_argument('--eval_u_batch_size', type=int,default=500, help="batch size for evaluation users (memory management)")
    parser.add_argument('--dataset', type=str,default='gowalla', help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--topks', nargs='?',default="[20]", help="@k test list")
    parser.add_argument('--epochs', type=int,default=50)
    parser.add_argument('--n_eigen', type=int,default=500)
    parser.add_argument('--val_ratio', type=float,default=0.1, help='ratio of training data to use for validation (0.1 = 10%)')
    parser.add_argument('--patience', type=int,default=5, help='early stopping patience')
    parser.add_argument('--min_delta', type=float,default=1e-5, help='minimum improvement for early stopping')
    parser.add_argument('--n_epoch_eval', type=int, default=5, help='evaluate every N epochs')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--model', type=str, default='uspec', help='rec-model, support [uspec]')
    parser.add_argument('--m_type', type=str, default='single', help='single or double similarity')
    parser.add_argument('--filter', type=str, default='u', help='u, i, or ui')
    parser.add_argument('--filter_order', type=int, default=6, help='polynomial order for spectral filters')
    parser.add_argument('--verbose', type=int, default=1, help='0 for silent, 1 for verbose')
    
    return parser.parse_args()

    
    return parser.parse_args()
