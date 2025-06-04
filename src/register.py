'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''
import world
import dataloader
import model_single
from pprint import pprint

# Dataset loading with ML-100K support
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'ml-100k':
    dataset = dataloader.ML100K()
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

if world.config['verbose'] > 0:
    print('===========config================')
    pprint(world.config)
    print("Test Topks:", world.topks)
    print('===========end===================')

MODELS = {
    # 'mf': model.PureMF,
    # 'lgn': model.LightGCN,
    'uspec': model_single.UniversalSpectralCF
}