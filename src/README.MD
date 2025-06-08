# Usage Examples for Both Model Types

## Quick Start Examples

### Basic Model (Fast, Simple)
```bash
# Default configuration with separate eigenvalues
python main.py --model_type basic --dataset ml-100k --u_n_eigen 32 --i_n_eigen 48

# Legacy mode (same eigenvalues for both)
python main.py --model_type basic --dataset ml-100k --n_eigen 50

# Large dataset with basic model
python main.py --model_type basic --dataset gowalla --u_n_eigen 64 --i_n_eigen 96
```

### Enhanced Model (Advanced, Full Features)
```bash
# Auto-adaptive with enhanced features
python main.py --model_type enhanced --dataset ml-100k --filter_design enhanced_basis

# Manual eigenvalues with advanced filter
python main.py --model_type enhanced --dataset ml-100k --u_n_eigen 48 --i_n_eigen 64 --filter_design multiscale

# High-performance configuration
python main.py --model_type enhanced --dataset gowalla --u_n_eigen 128 --i_n_eigen 256 --filter_design ensemble
```

## Model Comparison

### When to Use Basic Model:
- **Fast prototyping**: Quick experiments and baseline comparisons
- **Limited resources**: Lower memory and computational requirements
- **Simple datasets**: Small to medium datasets (< 10K users/items)
- **Educational purposes**: Understanding core spectral filtering concepts
- **Debugging**: Simpler architecture for troubleshooting

### When to Use Enhanced Model:
- **Production systems**: Maximum performance requirements
- **Large datasets**: Complex, sparse datasets (> 10K users/items)
- **Research**: Advanced filter designs and similarity techniques
- **Domain-specific optimization**: Specialized similarity measures
- **Scalability**: Auto-adaptive features for varying dataset sizes

## Dataset-Specific Recommendations

### ML-100K (Small, Dense Dataset)
```bash
# Basic model - fast and sufficient
python main.py --model_type basic --dataset ml-100k \
    --u_n_eigen 32 --i_n_eigen 48 \
    --epochs 50 --lr 0.001

# Enhanced model - maximum performance
python main.py --model_type enhanced --dataset ml-100k \
    --u_n_eigen 48 --i_n_eigen 64 \
    --filter_design enhanced_basis \
    --similarity_type cosine \
    --similarity_threshold 0.01
```

### LastFM (Music Domain)
```bash
# Basic model
python main.py --model_type basic --dataset lastfm \
    --u_n_eigen 48 --i_n_eigen 64 \
    --epochs 40

# Enhanced model with music-optimized settings
python main.py --model_type enhanced --dataset lastfm \
    --u_n_eigen 64 --i_n_eigen 96 \
    --filter_design enhanced_basis \
    --similarity_type cosine \
    --similarity_threshold 0.005
```

### Gowalla (Location Data)
```bash
# Basic model - good baseline
python main.py --model_type basic --dataset gowalla \
    --u_n_eigen 64 --i_n_eigen 96 \
    --epochs 30

# Enhanced model - location-optimized
python main.py --model_type enhanced --dataset gowalla \
    --u_n_eigen 128 --i_n_eigen 256 \
    --filter_design multiscale \
    --similarity_type cosine \
    --similarity_threshold 0.001
```

### Yelp2018 (Large, Sparse)
```bash
# Basic model - memory efficient
python main.py --model_type basic --dataset yelp2018 \
    --u_n_eigen 96 --i_n_eigen 128 \
    --epochs 25

# Enhanced model - high performance
python main.py --model_type enhanced --dataset yelp2018 \
    --u_n_eigen 192 --i_n_eigen 384 \
    --filter_design ensemble \
    --similarity_type jaccard \
    --similarity_threshold 0.0005
```

## Performance vs Complexity Trade-offs

### Basic Model Characteristics:
- **Parameters**: ~100-500 (depending on eigenvalue counts)
- **Training Time**: Fast (2-5x faster than enhanced)
- **Memory Usage**: Low (50-70% of enhanced model)
- **Performance**: Good baseline (usually 85-95% of enhanced performance)
- **Suitable for**: Datasets < 50K users/items

### Enhanced Model Characteristics:
- **Parameters**: ~500-2000+ (depending on filter design)
- **Training Time**: Moderate to slow (comprehensive features)
- **Memory Usage**: Higher (caching, advanced processing)
- **Performance**: Maximum (state-of-the-art results)
- **Suitable for**: Any dataset, especially large/complex ones

## Eigenvalue Configuration Comparison

### Basic Model Eigenvalue Handling:
```bash
# Simple, direct eigenvalue specification
python main.py --model_type basic --u_n_eigen 32 --i_n_eigen 48

# Uses basic heuristics for eigenvalue selection
# No auto-adaptive features
# Fast eigendecomposition
```

### Enhanced Model Eigenvalue Handling:
```bash
# Auto-adaptive based on dataset characteristics
python main.py --model_type enhanced --dataset ml-100k

# Manual with advanced similarity processing
python main.py --model_type enhanced --u_n_eigen 48 --i_n_eigen 64

# Intelligent caching and optimization
# Similarity-aware eigendecomposition
# Advanced threshold management
```

## Migration Guide

### From Basic to Enhanced:
```bash
# Step 1: Test with same eigenvalue configuration
python main.py --model_type basic --dataset ml-100k --u_n_eigen 32 --i_n_eigen 48
python main.py --model_type enhanced --dataset ml-100k --u_n_eigen 32 --i_n_eigen 48 --filter_design original

# Step 2: Enable enhanced features gradually
python main.py --model_type enhanced --dataset ml-100k --u_n_eigen 32 --i_n_eigen 48 --filter_design basis

# Step 3: Use auto-adaptive features
python main.py --model_type enhanced --dataset ml-100k --filter_design enhanced_basis

# Step 4: Optimize for your dataset
python main.py --model_type enhanced --dataset ml-100k --u_n_eigen 48 --i_n_eigen 64 --filter_design multiscale
```

### From Enhanced to Basic (for speed):
```bash
# Find equivalent basic configuration
# Enhanced: auto-adaptive eigenvalues
python main.py --model_type enhanced --dataset ml-100k  # Check output for actual eigenvalue counts

# Basic: use those same counts
python main.py --model_type basic --dataset ml-100k --u_n_eigen <actual_u> --i_n_eigen <actual_i>
```

## Debugging and Development

### Development Workflow:
```bash
# 1. Start with basic model for quick iteration
python main.py --model_type basic --dataset ml-100k --u_n_eigen 24 --i_n_eigen 32 --epochs 20

# 2. Move to enhanced for full features
python main.py --model_type enhanced --dataset ml-100k --filter_design basis --epochs 30

# 3. Optimize with advanced features
python main.py --model_type enhanced --dataset ml-100k --filter_design enhanced_basis --epochs 50
```

### Troubleshooting:
```bash
# Memory issues? Use basic model
python main.py --model_type basic --dataset large_dataset --u_n_eigen 32 --i_n_eigen 48

# Need maximum performance? Use enhanced
python main.py --model_type enhanced --dataset any_dataset --filter_design ensemble

# Debugging filters? Use basic with verbose output
python main.py --model_type basic --dataset ml-100k --verbose 1
```

## Performance Benchmarks (Approximate)

### ML-100K Results:
```
Basic Model:    NDCG@20 ≈ 0.375-0.385 (fast training)
Enhanced Model: NDCG@20 ≈ 0.385-0.395 (with optimization)
```

### Training Speed Comparison:
```
Basic Model:    ~30-60 seconds (ML-100K)
Enhanced Model: ~60-120 seconds (ML-100K, depending on filter design)
```

### Memory Usage:
```
Basic Model:    ~100-200 MB (ML-100K)
Enhanced Model: ~200-400 MB (ML-100K, with caching)
```

## Advanced Usage Patterns

### Hyperparameter Search:
```bash
# Quick search with basic model
for u in 24 32 48; do
  for i in 32 48 64; do
    python main.py --model_type basic --dataset ml-100k --u_n_eigen $u --i_n_eigen $i --epochs 20
  done
done

# Detailed search with enhanced model
python hyperparam_search.py --model_type enhanced --dataset ml-100k --filter_design enhanced_basis
```

### Cross-Model Validation:
```bash
# Train on basic, validate approach
python main.py --model_type basic --dataset ml-100k --u_n_eigen 32 --i_n_eigen 48

# Scale up to enhanced for final results
python main.py --model_type enhanced --dataset ml-100k --u_n_eigen 48 --i_n_eigen 64 --filter_design enhanced_basis
```