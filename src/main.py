'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import world
import utils
import procedure
import time
from register import dataset
import warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")

if world.config['m_type'] == 'single':
    from model_single import UniversalSpectralCF
else:
    from model_double import UniversalSpectralCF

utils.set_seed(world.seed)

print(f"Creating Universal Spectral CF model (seed: {world.seed}, device: {world.device})...")
model_start = time.time()
adj_mat = dataset.UserItemNet.tolil()
Recmodel = UniversalSpectralCF(adj_mat, world.config)
print(f"Model created in {time.time() - model_start:.2f}s")

print("Starting MSE-based training...")
training_start = time.time()
trained_model, final_results = procedure.train_and_evaluate(dataset, Recmodel, world.config)
total_time = time.time() - training_start

# Display final results
print("\n" + "="*70)
print(f"              FINAL RESULTS (SIMPLIFIED MSE TRAINING) - {world.config['dataset']}, filter - {world.config['filter']}, {world.config['m_type']}")
print("="*70)
print(f"\033[91mFinal Results: Recall@20={final_results['recall'][0]:.6f}, NDCG@20={final_results['ndcg'][0]:.6f}, Precision@20={final_results['precision'][0]:.6f}\033[0m")
print(f"Total experiment time: {total_time:.2f}s")
print("="*70)

# Show what the model learned
print(f"\n[FINAL FILTER ANALYSIS - {world.config['m_type'].upper()}]")
trained_model.debug_filter_learning()

print("\n🎉 Simplified Universal Spectral CF training completed!")
