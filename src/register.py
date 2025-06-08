'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Enhanced with model selection between basic and enhanced versions

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''
import world
import dataloader

# Dataset loading
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'ml-100k':
    dataset = dataloader.ML100K()
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

# Model selection based on configuration
model_type = world.config.get('model_type', 'enhanced')

if model_type == 'basic':
    import model
    MODELS = {'uspec': model.UniversalSpectralCF}
    print("🔧 Using Basic Universal Spectral CF (model.py)")
    print(f"   └─ Simple eigendecomposition with separate u_n_eigen/i_n_eigen")
    print(f"   └─ Fast training, minimal complexity")
    
elif model_type == 'enhanced':
    import model_enhanced
    MODELS = {'uspec': model_enhanced.UniversalSpectralCF}
    print("🚀 Using Enhanced Universal Spectral CF (model_enhanced.py)")
    print(f"   └─ DySimGCF-style similarity-aware Laplacian")
    print(f"   └─ Advanced filter designs and caching")
    print(f"   └─ Adaptive eigenvalue calculation")
    
else:
    raise ValueError(f"Unknown model_type: {model_type}. Choose 'basic' or 'enhanced'")

# Display configuration info
if world.config['verbose'] > 0:
    print(f"\n📊 Dataset Configuration:")
    print(f"   └─ Dataset: {world.dataset}")
    print(f"   └─ Users: {dataset.n_users:,}, Items: {dataset.m_items:,}")
    print(f"   └─ Training: {dataset.trainDataSize:,}, Validation: {dataset.valDataSize:,}")
    
    print(f"\n⚙️  Model Configuration:")
    print(f"   └─ Model Type: {model_type}")
    
    # Eigenvalue configuration
    u_n_eigen = world.config.get('u_n_eigen', 0)
    i_n_eigen = world.config.get('i_n_eigen', 0)
    n_eigen = world.config.get('n_eigen', 0)
    
    if u_n_eigen > 0 and i_n_eigen > 0:
        print(f"   └─ User Eigenvalues: {u_n_eigen}")
        print(f"   └─ Item Eigenvalues: {i_n_eigen}")
        print(f"   └─ Eigenvalue Ratio (i/u): {i_n_eigen/u_n_eigen:.2f}")
    elif n_eigen > 0:
        print(f"   └─ Eigenvalues (legacy): {n_eigen}")
    else:
        print(f"   └─ Eigenvalues: Auto-adaptive")
    
    # Model-specific configuration
    if model_type == 'enhanced':
        print(f"   └─ Filter Design: {world.config.get('filter_design', 'enhanced_basis')}")
        print(f"   └─ Similarity Type: {world.config.get('similarity_type', 'cosine')}")
        print(f"   └─ Similarity Threshold: {world.config.get('similarity_threshold', 0.01)}")
    
    print(f"   └─ Filter Type: {world.config['filter']}")
    print(f"   └─ Filter Order: {world.config['filter_order']}")
    print(f"   └─ Device: {world.device}")

# Legacy compatibility checks
if world.config.get('use_laplacian', False):
    print(f"⚠️  Note: use_laplacian flag detected. Enhanced model uses similarity-aware Laplacian by default.")

if world.config.get('use_similarity_norm', False):
    print(f"⚠️  Note: use_similarity_norm flag detected. Enhanced model uses advanced similarity processing by default.")