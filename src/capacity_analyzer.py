#!/usr/bin/env python3
'''
Filter Capacity Analyzer and Hyperparameter Search Guide
Analyzes filter complexity, estimates parameters, and recommends search strategies

Usage:
python capacity_analyzer.py [--analyze_all] [--recommend FILTER] [--dataset DATASET]
'''

import argparse
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import json

class FilterCapacityAnalyzer:
    """Analyzes filter designs and provides hyperparameter search guidance"""
    
    def __init__(self):
        self.filter_specs = self._define_filter_specifications()
        self.dataset_characteristics = self._define_dataset_characteristics()
    
    def _define_filter_specifications(self) -> Dict[str, Dict]:
        """Define detailed specifications for each filter design"""
        return {
            'original': {
                'base_params': 14,
                'param_formula': lambda config: 14,
                'complexity': 'Very Low',
                'description': 'Basic FIR filter with fixed coefficients',
                'capacity_factors': ['filter_order'],
                'typical_range': '10-20 params',
                'overfitting_risk': 'Very Low',
                'dataset_suitability': ['ml-100k', 'ml-1m', 'lastfm'],
                'search_priority': ['lr', 'decay', 'filter_order']
            },
            
            'basis': {
                'base_params': 32,
                'param_formula': lambda config: 32 + config.get('filter_order', 6) * 2,
                'complexity': 'Low',
                'description': 'Learned basis functions with linear combination',
                'capacity_factors': ['filter_order', 'basis_functions'],
                'typical_range': '30-50 params',
                'overfitting_risk': 'Low',
                'dataset_suitability': ['ml-100k', 'ml-1m', 'lastfm'],
                'search_priority': ['lr', 'decay', 'filter_order', 'basis_expansion']
            },
            
            'enhanced_basis': {
                'base_params': 52,
                'param_formula': lambda config: 52 + config.get('basis_expansion', 3) * 8,
                'complexity': 'Medium-Low',
                'description': 'Enhanced basis with adaptive weights and regularization',
                'capacity_factors': ['basis_expansion', 'adaptive_weights', 'filter_order'],
                'typical_range': '50-100 params',
                'overfitting_risk': 'Low-Medium',
                'dataset_suitability': ['ml-100k', 'ml-1m', 'lastfm'],
                'search_priority': ['lr', 'decay', 'basis_expansion', 'regularization_strength']
            },
            
            'adaptive_golden': {
                'base_params': 28,
                'param_formula': lambda config: 28 + config.get('filter_order', 6) * 3,
                'complexity': 'Low',
                'description': 'Golden ratio based adaptive filter',
                'capacity_factors': ['filter_order', 'adaptation_rate'],
                'typical_range': '25-50 params',
                'overfitting_risk': 'Low',
                'dataset_suitability': ['ml-100k', 'ml-1m', 'lastfm'],
                'search_priority': ['lr', 'decay', 'filter_order', 'adaptation_rate']
            },
            
            'adaptive': {
                'base_params': 15,
                'param_formula': lambda config: 15 + config.get('filter_order', 6) * 2,
                'complexity': 'Low',
                'description': 'Basic adaptive filter with learned parameters',
                'capacity_factors': ['filter_order', 'adaptation_strength'],
                'typical_range': '15-35 params',
                'overfitting_risk': 'Very Low',
                'dataset_suitability': ['ml-100k', 'ml-1m', 'lastfm'],
                'search_priority': ['lr', 'decay', 'filter_order']
            },
            
            'neural': {
                'base_params': 100,
                'param_formula': lambda config: self._calculate_neural_params(config),
                'complexity': 'High',
                'description': 'Multi-layer perceptron with configurable architecture',
                'capacity_factors': ['hidden_dims', 'dropout_rate', 'batch_norm', 'activation'],
                'typical_range': '200-1000 params',
                'overfitting_risk': 'Medium-High',
                'dataset_suitability': ['ml-1m', 'lastfm'],  # Needs larger datasets
                'search_priority': ['hidden_dims', 'dropout_rate', 'lr', 'batch_norm', 'activation']
            },
            
            'deep': {
                'base_params': 200,
                'param_formula': lambda config: self._calculate_deep_params(config),
                'complexity': 'Very High',
                'description': 'Deep neural network with residual connections',
                'capacity_factors': ['hidden_dims', 'depth_scaling', 'residual_connections', 'layer_decay'],
                'typical_range': '500-2000+ params',
                'overfitting_risk': 'High',
                'dataset_suitability': ['ml-1m', 'lastfm'],  # Only large datasets
                'search_priority': ['hidden_dims', 'dropout_rate', 'residual_connections', 'layer_decay', 'lr']
            },
            
            'multiscale': {
                'base_params': 150,
                'param_formula': lambda config: self._calculate_multiscale_params(config),
                'complexity': 'High',
                'description': 'Multi-resolution processing with attention fusion',
                'capacity_factors': ['scales', 'per_scale_dim', 'fusion_method', 'attention_heads'],
                'typical_range': '300-800 params',
                'overfitting_risk': 'Medium-High',
                'dataset_suitability': ['ml-100k', 'ml-1m', 'lastfm'],  # Works across datasets
                'search_priority': ['scales', 'per_scale_dim', 'fusion_method', 'attention_heads', 'lr']
            },
            
            'ensemble': {
                'base_params': 500,
                'param_formula': lambda config: self._calculate_ensemble_params(config),
                'complexity': 'Very High',
                'description': 'Multiple models with learned combination weights',
                'capacity_factors': ['n_models', 'model_types', 'ensemble_method', 'diversity_loss'],
                'typical_range': '1000-5000+ params',
                'overfitting_risk': 'Very High',
                'dataset_suitability': ['ml-1m', 'lastfm'],  # Only very large datasets
                'search_priority': ['n_models', 'diversity_loss', 'ensemble_method', 'bootstrap_ratio', 'lr']
            }
        }
    
    def _calculate_neural_params(self, config: Dict) -> int:
        """Calculate parameters for neural filter"""
        hidden_dims_str = config.get('hidden_dims', '[128, 64]')
        try:
            hidden_dims = eval(hidden_dims_str)
        except:
            hidden_dims = [128, 64]
        
        # Assume input dimension is filter_order * n_eigen_factors
        input_dim = config.get('filter_order', 6) * 10  # Simplified assumption
        
        total_params = 0
        prev_dim = input_dim
        
        for dim in hidden_dims:
            total_params += prev_dim * dim + dim  # weights + bias
            if config.get('batch_norm', False):
                total_params += dim * 2  # batch norm parameters
            prev_dim = dim
        
        # Output layer
        total_params += prev_dim + 1
        
        return total_params
    
    def _calculate_deep_params(self, config: Dict) -> int:
        """Calculate parameters for deep filter"""
        base_params = self._calculate_neural_params(config)
        
        # Additional parameters for residual connections
        if config.get('residual_connections', True):
            base_params *= 1.3  # Approximate increase
        
        # Layer decay parameters
        if config.get('layer_decay', 1.0) < 1.0:
            base_params += 50  # Additional decay parameters
        
        return int(base_params)
    
    def _calculate_multiscale_params(self, config: Dict) -> int:
        """Calculate parameters for multiscale filter"""
        scales_str = config.get('scales', '[1, 2, 4]')
        try:
            scales = eval(scales_str)
        except:
            scales = [1, 2, 4]
        
        per_scale_dim = config.get('per_scale_dim', 64)
        
        # Parameters per scale
        params_per_scale = per_scale_dim * 20  # Simplified calculation
        total_params = len(scales) * params_per_scale
        
        # Fusion parameters
        if config.get('fusion_method') == 'attention':
            attention_heads = config.get('attention_heads', 4)
            total_params += len(scales) * per_scale_dim * attention_heads
        
        return total_params
    
    def _calculate_ensemble_params(self, config: Dict) -> int:
        """Calculate parameters for ensemble filter"""
        n_models = config.get('n_models', 5)
        
        # Each model is approximately a neural filter
        base_model_params = self._calculate_neural_params(config)
        total_params = n_models * base_model_params
        
        # Ensemble combination parameters
        if config.get('ensemble_method') == 'learned_weights':
            total_params += n_models * 10  # Weight learning parameters
        elif config.get('ensemble_method') == 'attention':
            total_params += n_models * 50  # Attention parameters
        
        return total_params
    
    def _define_dataset_characteristics(self) -> Dict[str, Dict]:
        """Define characteristics of each dataset"""
        return {
            'ml-100k': {
                'size': 'Small',
                'users': 943,
                'items': 1682,
                'ratings': 100000,
                'sparsity': 0.937,
                'recommended_max_params': 500,
                'overfitting_threshold': 300,
                'suitable_filters': ['original', 'basis', 'enhanced_basis', 'adaptive', 'adaptive_golden', 'multiscale']
            },
            'ml-1m': {
                'size': 'Medium',
                'users': 6040,
                'items': 3952,
                'ratings': 1000209,
                'sparsity': 0.958,
                'recommended_max_params': 2000,
                'overfitting_threshold': 1000,
                'suitable_filters': ['basis', 'enhanced_basis', 'adaptive_golden', 'neural', 'deep', 'multiscale', 'ensemble']
            },
            'lastfm': {
                'size': 'Large',
                'users': 1892,
                'items': 17632,
                'ratings': 92834,
                'sparsity': 0.997,
                'recommended_max_params': 1500,
                'overfitting_threshold': 800,
                'suitable_filters': ['enhanced_basis', 'adaptive_golden', 'neural', 'multiscale', 'ensemble']
            }
        }
    
    def analyze_filter_capacity(self, filter_name: str, config: Dict = None) -> Dict:
        """Analyze the capacity and characteristics of a specific filter"""
        if filter_name not in self.filter_specs:
            raise ValueError(f"Unknown filter: {filter_name}")
        
        if config is None:
            config = self._get_default_config(filter_name)
        
        spec = self.filter_specs[filter_name]
        estimated_params = spec['param_formula'](config)
        
        analysis = {
            'filter_name': filter_name,
            'estimated_parameters': estimated_params,
            'complexity_level': spec['complexity'],
            'description': spec['description'],
            'capacity_factors': spec['capacity_factors'],
            'overfitting_risk': spec['overfitting_risk'],
            'suitable_datasets': spec['dataset_suitability'],
            'search_priorities': spec['search_priority'],
            'config_used': config
        }
        
        # Add capacity category
        if estimated_params < 50:
            analysis['capacity_category'] = 'Low Capacity (<50 params)'
        elif estimated_params < 200:
            analysis['capacity_category'] = 'Medium Capacity (50-200 params)'
        elif estimated_params < 1000:
            analysis['capacity_category'] = 'High Capacity (200-1000 params)'
        else:
            analysis['capacity_category'] = 'Very High Capacity (1000+ params)'
        
        return analysis
    
    def _get_default_config(self, filter_name: str) -> Dict:
        """Get default configuration for a filter"""
        defaults = {
            'original': {'filter_order': 6},
            'basis': {'filter_order': 6, 'basis_expansion': 3},
            'enhanced_basis': {'filter_order': 6, 'basis_expansion': 3, 'adaptive_weights': True},
            'adaptive_golden': {'filter_order': 6, 'adaptation_rate': 0.1},
            'adaptive': {'filter_order': 6},
            'neural': {'filter_order': 6, 'hidden_dims': '[128, 64]', 'dropout_rate': 0.2, 'batch_norm': True},
            'deep': {'filter_order': 6, 'hidden_dims': '[256, 128, 64]', 'dropout_rate': 0.3, 'residual_connections': True},
            'multiscale': {'filter_order': 6, 'scales': '[1, 2, 4]', 'per_scale_dim': 64, 'fusion_method': 'attention'},
            'ensemble': {'filter_order': 6, 'n_models': 5, 'hidden_dims': '[128, 64]', 'ensemble_method': 'learned_weights'}
        }
        return defaults.get(filter_name, {})
    
    def recommend_search_strategy(self, filter_name: str, dataset: str, budget: int = 50) -> Dict:
        """Recommend hyperparameter search strategy for a filter-dataset combination"""
        if filter_name not in self.filter_specs:
            raise ValueError(f"Unknown filter: {filter_name}")
        
        if dataset not in self.dataset_characteristics:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        filter_spec = self.filter_specs[filter_name]
        dataset_spec = self.dataset_characteristics[dataset]
        
        # Check dataset suitability
        suitable = dataset in filter_spec['dataset_suitability']
        
        # Estimate parameter count with default config
        default_config = self._get_default_config(filter_name)
        estimated_params = filter_spec['param_formula'](default_config)
        
        # Risk assessment
        overfitting_risk = 'Low'
        if estimated_params > dataset_spec['overfitting_threshold']:
            overfitting_risk = 'High'
        elif estimated_params > dataset_spec['overfitting_threshold'] * 0.6:
            overfitting_risk = 'Medium'
        
        # Strategy recommendation
        if budget < 20:
            recommended_strategy = 'random'
            strategy_reason = 'Small budget: random search for broad exploration'
        elif budget < 50:
            recommended_strategy = 'bayesian'
            strategy_reason = 'Medium budget: Bayesian optimization for efficiency'
        else:
            if estimated_params > 500:
                recommended_strategy = 'bayesian'
                strategy_reason = 'High-capacity filter: Bayesian to avoid expensive evaluations'
            else:
                recommended_strategy = 'grid'
                strategy_reason = 'Large budget + manageable complexity: systematic grid search'
        
        # Budget allocation
        if estimated_params > 1000:
            recommended_budget = max(budget, 100)
            budget_reason = 'Very high capacity requires extensive search'
        elif estimated_params > 500:
            recommended_budget = max(budget, 50)
            budget_reason = 'High capacity benefits from thorough search'
        else:
            recommended_budget = budget
            budget_reason = 'Standard budget sufficient'
        
        return {
            'filter_name': filter_name,
            'dataset': dataset,
            'dataset_suitable': suitable,
            'estimated_parameters': estimated_params,
            'overfitting_risk': overfitting_risk,
            'recommended_strategy': recommended_strategy,
            'strategy_reason': strategy_reason,
            'recommended_budget': recommended_budget,
            'budget_reason': budget_reason,
            'search_priorities': filter_spec['search_priority'][:5],  # Top 5 priorities
            'regularization_needed': estimated_params > dataset_spec['overfitting_threshold'] * 0.5,
            'warnings': self._generate_warnings(filter_name, dataset, estimated_params, dataset_spec)
        }
    
    def _generate_warnings(self, filter_name: str, dataset: str, estimated_params: int, dataset_spec: Dict) -> List[str]:
        """Generate warnings for potentially problematic combinations"""
        warnings = []
        
        if estimated_params > dataset_spec['recommended_max_params']:
            warnings.append(f"Filter may be too complex for {dataset} (>{dataset_spec['recommended_max_params']} params recommended)")
        
        if filter_name in ['deep', 'ensemble'] and dataset == 'ml-100k':
            warnings.append("Very high-capacity filters may overfit on small datasets")
        
        if filter_name in ['neural', 'deep'] and dataset_spec['sparsity'] > 0.99:
            warnings.append("Neural filters may struggle with extremely sparse data")
        
        if estimated_params > 1000 and dataset_spec['ratings'] < 500000:
            warnings.append("Parameter count may exceed data capacity - consider regularization")
        
        return warnings
    
    def compare_filters(self, dataset: str, config: Dict = None) -> Dict:
        """Compare all filters for a specific dataset"""
        results = {}
        dataset_spec = self.dataset_characteristics[dataset]
        
        for filter_name in self.filter_specs.keys():
            analysis = self.analyze_filter_capacity(filter_name, config)
            recommendation = self.recommend_search_strategy(filter_name, dataset)
            
            results[filter_name] = {
                **analysis,
                'recommendation': recommendation,
                'fit_score': self._calculate_fit_score(filter_name, dataset)
            }
        
        # Rank by fit score
        ranked_filters = sorted(results.items(), key=lambda x: x[1]['fit_score'], reverse=True)
        
        return {
            'dataset': dataset,
            'dataset_characteristics': dataset_spec,
            'filter_analyses': dict(ranked_filters),
            'recommendations': {
                'best_overall': ranked_filters[0][0],
                'best_high_capacity': self._find_best_high_capacity(ranked_filters),
                'safest_choice': self._find_safest_choice(ranked_filters),
                'most_efficient': self._find_most_efficient(ranked_filters)
            }
        }
    
    def _calculate_fit_score(self, filter_name: str, dataset: str) -> float:
        """Calculate how well a filter fits a dataset (0-100)"""
        filter_spec = self.filter_specs[filter_name]
        dataset_spec = self.dataset_characteristics[dataset]
        
        score = 50  # Base score
        
        # Dataset suitability
        if dataset in filter_spec['dataset_suitability']:
            score += 30
        
        # Parameter count vs dataset size
        estimated_params = filter_spec['param_formula'](self._get_default_config(filter_name))
        param_ratio = estimated_params / dataset_spec['recommended_max_params']
        
        if param_ratio < 0.5:
            score += 10  # Good fit
        elif param_ratio < 1.0:
            score += 20  # Excellent fit
        elif param_ratio < 1.5:
            score -= 10  # Slightly too complex
        else:
            score -= 30  # Too complex
        
        # Overfitting risk adjustment
        risk_penalties = {'Very Low': 0, 'Low': 5, 'Medium': 10, 'High': 20, 'Very High': 30}
        score -= risk_penalties.get(filter_spec['overfitting_risk'], 15)
        
        return max(0, min(100, score))
    
    def _find_best_high_capacity(self, ranked_filters: List) -> str:
        """Find the best high-capacity filter"""
        for filter_name, analysis in ranked_filters:
            if analysis['estimated_parameters'] > 200:
                return filter_name
        return ranked_filters[0][0]  # Fallback to best overall
    
    def _find_safest_choice(self, ranked_filters: List) -> str:
        """Find the safest (lowest overfitting risk) choice"""
        safe_filters = [(name, analysis) for name, analysis in ranked_filters 
                       if analysis['overfitting_risk'] in ['Very Low', 'Low']]
        return safe_filters[0][0] if safe_filters else ranked_filters[0][0]
    
    def _find_most_efficient(self, ranked_filters: List) -> str:
        """Find the most parameter-efficient choice"""
        return min(ranked_filters, key=lambda x: x[1]['estimated_parameters'])[0]
    
    def print_analysis(self, analysis: Dict):
        """Pretty print analysis results"""
        if 'filter_analyses' in analysis:  # Compare filters result
            self._print_comparison(analysis)
        else:  # Single filter analysis
            self._print_single_analysis(analysis)
    
    def _print_single_analysis(self, analysis: Dict):
        """Print single filter analysis"""
        print(f"\n🔍 FILTER ANALYSIS: {analysis['filter_name'].upper()}")
        print("=" * 60)
        print(f"📊 Estimated Parameters: {analysis['estimated_parameters']}")
        print(f"🎯 Capacity Category: {analysis['capacity_category']}")
        print(f"⚡ Complexity Level: {analysis['complexity_level']}")
        print(f"⚠️ Overfitting Risk: {analysis['overfitting_risk']}")
        print(f"📝 Description: {analysis['description']}")
        
        print(f"\n🔧 Key Capacity Factors:")
        for factor in analysis['capacity_factors']:
            print(f"  • {factor}")
        
        print(f"\n🎯 Hyperparameter Search Priorities:")
        for i, priority in enumerate(analysis['search_priorities'], 1):
            print(f"  {i}. {priority}")
        
        print(f"\n📈 Suitable Datasets:")
        for dataset in analysis['suitable_datasets']:
            print(f"  • {dataset}")
    
    def _print_comparison(self, analysis: Dict):
        """Print filter comparison results"""
        print(f"\n🏆 FILTER COMPARISON: {analysis['dataset'].upper()}")
        print("=" * 80)
        
        dataset_char = analysis['dataset_characteristics']
        print(f"📊 Dataset: {dataset_char['size']} ({dataset_char['ratings']:,} ratings)")
        print(f"🎯 Sparsity: {dataset_char['sparsity']:.3f}")
        print(f"⚠️ Recommended Max Params: {dataset_char['recommended_max_params']}")
        
        print(f"\n🥇 RECOMMENDATIONS:")
        recs = analysis['recommendations']
        print(f"  Best Overall: {recs['best_overall']}")
        print(f"  Best High-Capacity: {recs['best_high_capacity']}")
        print(f"  Safest Choice: {recs['safest_choice']}")
        print(f"  Most Efficient: {recs['most_efficient']}")
        
        print(f"\n📋 DETAILED ANALYSIS:")
        print(f"{'Filter':<15} {'Params':<8} {'Fit':<6} {'Risk':<12} {'Status'}")
        print("-" * 65)
        
        for filter_name, filter_analysis in analysis['filter_analyses'].items():
            params = filter_analysis['estimated_parameters']
            fit_score = filter_analysis['fit_score']
            risk = filter_analysis['overfitting_risk']
            
            # Status indicator
            if filter_analysis['recommendation']['dataset_suitable']:
                status = "✅ Suitable"
            else:
                status = "⚠️ Risky"
            
            if filter_analysis['recommendation']['warnings']:
                status += " ⚠️"
            
            print(f"{filter_name:<15} {params:<8} {fit_score:<6.1f} {risk:<12} {status}")

def main():
    parser = argparse.ArgumentParser(description="Filter Capacity Analyzer")
    parser.add_argument('--analyze_all', action='store_true',
                       help='Analyze all filters')
    parser.add_argument('--recommend', type=str,
                       help='Get recommendations for specific filter')
    parser.add_argument('--dataset', type=str, default='ml-100k',
                       choices=['ml-100k', 'ml-1m', 'lastfm'],
                       help='Dataset to analyze for')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all filters for dataset')
    
    args = parser.parse_args()
    
    analyzer = FilterCapacityAnalyzer()
    
    if args.compare:
        comparison = analyzer.compare_filters(args.dataset)
        analyzer.print_analysis(comparison)
        
    elif args.analyze_all:
        print("🔍 ANALYZING ALL FILTER DESIGNS")
        print("=" * 60)
        
        for filter_name in analyzer.filter_specs.keys():
            analysis = analyzer.analyze_filter_capacity(filter_name)
            analyzer.print_analysis(analysis)
            print()
        
    elif args.recommend:
        if args.recommend not in analyzer.filter_specs:
            print(f"❌ Unknown filter: {args.recommend}")
            print(f"Available filters: {', '.join(analyzer.filter_specs.keys())}")
            return 1
        
        analysis = analyzer.analyze_filter_capacity(args.recommend)
        recommendation = analyzer.recommend_search_strategy(args.recommend, args.dataset)
        
        analyzer.print_analysis(analysis)
        
        print(f"\n🎯 SEARCH STRATEGY RECOMMENDATION:")
        print(f"  Strategy: {recommendation['recommended_strategy'].upper()}")
        print(f"  Reason: {recommendation['strategy_reason']}")
        print(f"  Budget: {recommendation['recommended_budget']} experiments")
        print(f"  Budget Reason: {recommendation['budget_reason']}")
        
        if recommendation['warnings']:
            print(f"\n⚠️ WARNINGS:")
            for warning in recommendation['warnings']:
                print(f"  • {warning}")
    
    else:
        # Show high-capacity filter summary
        print("🚀 HIGH-CAPACITY FILTER SUMMARY")
        print("=" * 60)
        
        high_capacity_filters = ['neural', 'deep', 'multiscale', 'ensemble']
        
        for filter_name in high_capacity_filters:
            analysis = analyzer.analyze_filter_capacity(filter_name)
            print(f"\n{filter_name.upper()}: {analysis['estimated_parameters']} params")
            print(f"  {analysis['description']}")
            print(f"  Risk: {analysis['overfitting_risk']}")
            print(f"  Best for: {', '.join(analysis['suitable_datasets'])}")
        
        print(f"\n💡 Use --compare --dataset {args.dataset} for detailed comparison")
        print(f"💡 Use --recommend FILTER --dataset {args.dataset} for specific guidance")
    
    return 0

if __name__ == "__main__":
    exit(main())