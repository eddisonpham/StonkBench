"""
Unified Evaluation Pipeline for Time Series Generative Models

This script provides a comprehensive evaluation framework using MLFlow to track and compare
all implemented models across different evaluation metrics:
- Diversity: Intra-Class Distance (ICD) with Euclidean and DTW metrics
- Efficiency: Runtime
- Fidelity: Feature-based metrics (MDD, MD, SDD, SD, KD, ACD) and Stylized Facts
- Visual Assessment: t-SNE and Distribution plots
"""

import sys
import os
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from pathlib import Path
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.preprocessing.preprocessing import preprocess_data

from src.models.base.base_model import (
    BaseGenerativeModel,
    ParametricModel,
    DeepLearningModel
)
from src.models.parametric.gbm import GeometricBrownianMotion
from src.models.parametric.ou_process import OrnsteinUhlenbeckProcess
from src.models.non_parametric.vanilla_gan import VanillaGAN
from src.models.non_parametric.wasserstein_gan import WassersteinGAN

from src.evaluation_measures.metrics.diversity import calculate_icd
from src.evaluation_measures.metrics.efficiency import measure_runtimes
from src.evaluation_measures.metrics.fidelity import (
    calculate_mdd, calculate_md, calculate_sdd, calculate_sd, calculate_kd, calculate_acd
)
from src.evaluation_measures.metrics.stylized_facts import (
    heavy_tails, autocorr_raw, volatility_clustering, long_memory_abs, non_stationarity
)
from src.evaluation_measures.visualizations.plots import visualize_tsne, visualize_distribution

from src.utils.display_utils import show_with_start_divider, show_with_end_divider
from src.utils.transformations_utils import (
    TimeSeriesDataset,
    create_dataloaders
)


class UnifiedEvaluator:
    """
    Unified evaluator for time series generative models using MLFlow for experiment tracking.
    """
    
    def __init__(self, experiment_name: str = "TimeSeries_Generation_Evaluation"):
        """
        Initialize the evaluator with MLFlow experiment.
        
        Args:
            experiment_name (str): Name of the MLFlow experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
        # Results storage
        self.results = {}
        self.results_dir = Path("data/evaluation_results") # Updated path
        self.results_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def evaluate_model(self, 
                      model: BaseGenerativeModel, 
                      model_name: str,
                      real_data: np.ndarray,
                      train_loader,
                      num_generated_samples: int = 500) -> Dict[str, Any]:
        """
        Evaluate a single model across all metrics.
        
        Args:
            model: The generative model to evaluate
            model_name: Name of the model for logging
            real_data: Real data for comparison (shape: R, L, N)
            train_loader: Training data loader for model fitting
            num_generated_samples: Number of samples to generate for evaluation
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        show_with_start_divider(f"Evaluating {model_name}")
        
        # Start MLFlow run
        with mlflow.start_run(run_name=f"{model_name}_{self.timestamp}"):
            # Log model name only
            mlflow.log_param("model_name", model_name)
            
            evaluation_results = {}
            
            # 1. Train the model
            print(f"Training {model_name}...")
            train_start = time.time()
            model.fit(train_loader)
            train_time = time.time() - train_start
            mlflow.log_metric("training_time", train_time)
            evaluation_results["training_time"] = train_time
            
            # 2. Generate synthetic data
            print(f"Generating {num_generated_samples} samples...")
            # Efficiency metrics: Measure generation time
            gen_time = measure_runtimes(model.generate, num_generated_samples)
            mlflow.log_metric("generation_time_500_samples", gen_time)
            evaluation_results["generation_time_500_samples"] = gen_time

            # Actually generate the synthetic data
            generated_data = model.generate(num_generated_samples)
            
            # Convert to numpy if needed
            if torch.is_tensor(generated_data):
                generated_data = generated_data.detach().cpu().numpy()
            
            # Ensure same shape as real data
            if generated_data.shape[1:] != real_data.shape[1:]:
                print(f"Warning: Shape mismatch. Real: {real_data.shape}, Synthetic: {generated_data.shape}")
                # Take only the matching dimensions
                min_length = min(real_data.shape[1], generated_data.shape[1])
                min_channels = min(real_data.shape[2], generated_data.shape[2])
                real_data = real_data[:, :min_length, :min_channels]
                generated_data = generated_data[:, :min_length, :min_channels]
            
            # 3. Diversity Metrics
            print("Computing diversity metrics...")
            diversity_results = self._evaluate_diversity(generated_data)
            evaluation_results.update(diversity_results)
            
            # 4. Fidelity Metrics
            print("Computing fidelity metrics...")
            fidelity_results = self._evaluate_fidelity(real_data, generated_data)
            evaluation_results.update(fidelity_results)
            
            # 5. Stylized Facts (for financial data)
            print("Computing stylized facts...")
            stylized_results = self._evaluate_stylized_facts(real_data, generated_data)
            evaluation_results.update(stylized_results)
            
            # 6. Visual Assessments
            print("Creating visual assessments...")
            self._create_visual_assessments(real_data, generated_data, model_name)
            
            # Log all metrics to MLFlow
            for metric_name, value in evaluation_results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, value)
                elif isinstance(value, np.ndarray):
                    mlflow.log_metric(f"{metric_name}_mean", float(np.mean(value)))
                    mlflow.log_metric(f"{metric_name}_std", float(np.std(value)))
            
            # Save model
            model_path = self.results_dir / f"{model_name}_{self.timestamp}"
            model.save_model(str(model_path))
            mlflow.log_artifacts(str(model_path))
            
            # Save synthetic data
            synthetic_path = self.results_dir / f"synthetic_{model_name}_{self.timestamp}.npy"
            np.save(synthetic_path, generated_data)
            mlflow.log_artifact(str(synthetic_path))
            
            print(f"Evaluation completed for {model_name}")
            return evaluation_results
    
    def _evaluate_diversity(self, synthetic_data: np.ndarray) -> Dict[str, float]:
        """Evaluate diversity metrics."""
        results = {}
        
        # Intra-Class Distance with Euclidean metric
        icd_euclidean = calculate_icd(synthetic_data, metric="euclidean")
        results["icd_euclidean"] = icd_euclidean
        
        # Intra-Class Distance with DTW metric
        icd_dtw = calculate_icd(synthetic_data, metric="dtw")
        results["icd_dtw"] = icd_dtw
        
        return results
    
    def _evaluate_fidelity(self, real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, float]:
        """Evaluate fidelity metrics."""
        results = {}
        
        # Feature-based metrics
        results["mdd"] = calculate_mdd(real_data, synthetic_data)
        results["md"] = calculate_md(real_data, synthetic_data)
        results["sdd"] = calculate_sdd(real_data, synthetic_data)
        results["sd"] = calculate_sd(real_data, synthetic_data)
        results["kd"] = calculate_kd(real_data, synthetic_data)
        results["acd"] = calculate_acd(real_data, synthetic_data)
        
        return results
    
    def _evaluate_stylized_facts(self, real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate stylized facts for financial data."""
        results = {}
        
        try:
            # Heavy tails (excess kurtosis)
            heavy_tails_real = heavy_tails(real_data)
            heavy_tails_synth = heavy_tails(synthetic_data)
            results["heavy_tails_real"] = heavy_tails_real.tolist()
            results["heavy_tails_synth"] = heavy_tails_synth.tolist()
            results["heavy_tails_diff"] = np.abs(heavy_tails_real - heavy_tails_synth).tolist()
            
            # Autocorrelation of raw returns
            autocorr_real = autocorr_raw(real_data)
            autocorr_synth = autocorr_raw(synthetic_data)
            results["autocorr_raw_real"] = autocorr_real.tolist()
            results["autocorr_raw_synth"] = autocorr_synth.tolist()
            results["autocorr_raw_diff"] = np.abs(autocorr_real - autocorr_synth).tolist()
            
            # Volatility clustering
            vol_clust_real = volatility_clustering(real_data)
            vol_clust_synth = volatility_clustering(synthetic_data)
            results["volatility_clustering_real"] = vol_clust_real.tolist()
            results["volatility_clustering_synth"] = vol_clust_synth.tolist()
            results["volatility_clustering_diff"] = np.abs(vol_clust_real - vol_clust_synth).tolist()
            
            # Long memory in absolute returns
            long_mem_real = long_memory_abs(real_data)
            long_mem_synth = long_memory_abs(synthetic_data)
            results["long_memory_real"] = long_mem_real.tolist()
            results["long_memory_synth"] = long_mem_synth.tolist()
            results["long_memory_diff"] = np.abs(long_mem_real - long_mem_synth).tolist()
            
            # Non-stationarity
            nonstat_real = non_stationarity(real_data)
            nonstat_synth = non_stationarity(synthetic_data)
            results["non_stationarity_real"] = nonstat_real.tolist()
            results["non_stationarity_synth"] = nonstat_synth.tolist()
            results["non_stationarity_diff"] = np.abs(nonstat_real - nonstat_synth).tolist()
            
        except Exception as e:
            print(f"Warning: Stylized facts evaluation failed: {e}")
            results["stylized_facts_error"] = str(e)
        
        return results
    
    def _create_visual_assessments(self, real_data: np.ndarray, synthetic_data: np.ndarray, model_name: str):
        """Create visual assessment plots."""
        try:
            # Create model-specific results directory
            model_results_dir = self.results_dir / f"visualizations_{model_name}_{self.timestamp}"
            model_results_dir.mkdir(exist_ok=True)
            
            # t-SNE visualization
            visualize_tsne(real_data, synthetic_data, str(model_results_dir), model_name)
            
            # Distribution visualization
            visualize_distribution(real_data, synthetic_data, str(model_results_dir), model_name)
            
            # Log visualizations to MLFlow
            mlflow.log_artifacts(str(model_results_dir))
            
        except Exception as e:
            print(f"Warning: Visual assessment failed: {e}")
    
    def run_complete_evaluation(self, 
                              dataset_config: Dict[str, Any],
                              models_config: Dict[str, Any],
                              num_samples: int = 500) -> Dict[str, Any]:
        """
        Run complete evaluation on all models with 500 generated samples per model.
        
        Args:
            dataset_config: Configuration for data preprocessing
            models_config: Configuration for models
            num_samples: Number of samples to generate for evaluation
            
        Returns:
            Dictionary containing results for all models
        """
        show_with_start_divider("Starting Complete Evaluation Pipeline")
        
        # 1. Preprocess data
        print("Preprocessing data...")
        train_data_np, valid_data_np = preprocess_data(dataset_config)
        
        # Create data loaders
        batch_size = 32
        train_loader, valid_loader = create_dataloaders(
            train_data_np, valid_data_np,
            batch_size=batch_size,
            train_seed=42,
            valid_seed=123,
            num_workers=0,
            pin_memory=False
        )
        
        # Get data dimensions
        num_samples_real, length, num_channels = train_data_np.shape
        print(f"Data shape: {train_data_np.shape}")
        
        # 2. Initialize models
        models = {}
        
        # Parametric models
        models["GBM"] = GeometricBrownianMotion(length=length, num_channels=num_channels)
        models["OU_Process"] = OrnsteinUhlenbeckProcess(length=length, num_channels=num_channels)
        
        # Non-parametric models
        models["Vanilla_GAN"] = VanillaGAN(
            length=length, 
            num_channels=num_channels, 
            latent_dim=models_config.get("latent_dim", 64),
            hidden_dim=models_config.get("hidden_dim", 128),
            lr=models_config.get("lr", 0.0002)
        )
        
        # Fix WassersteinGAN to inherit properly
        models["Wasserstein_GAN"] = WassersteinGAN(
            length=length, 
            num_channels=num_channels,
            latent_dim=models_config.get("latent_dim", 64),
            hidden_dim=models_config.get("hidden_dim", 128),
            lr=models_config.get("wgan_lr", 0.00005),
            n_critic=models_config.get("n_critic", 5),
            clip_value=models_config.get("clip_value", 0.01)
        )
        
        # 3. Evaluate each model
        all_results = {}
        for model_name, model in models.items():
            try:
                results = self.evaluate_model(
                    model=model,
                    model_name=model_name,
                    real_data=valid_data_np,  # Use validation data for evaluation
                    train_loader=train_loader,
                    num_generated_samples=num_samples
                )
                all_results[model_name] = results
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}
        
        # 4. Save comprehensive results
        results_file = self.results_dir / f"complete_evaluation_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Log results to MLFlow
        mlflow.log_artifact(str(results_file))
        
        show_with_end_divider("EVALUATION COMPLETE")
        print(f"Results saved to: {results_file}")
        print(f"MLFlow experiment: {self.experiment_name}")
        
        return all_results

def main():
    """Main function to run the evaluation pipeline."""
    
    # Configuration for data preprocessing
    dataset_config = {
        'original_data_path': str(project_root / 'data' / 'raw' / 'GOOG' / 'GOOG.csv'),
        'output_ori_path': str(project_root / 'data' / 'preprocessed'),
        'dataset_name': 'goog_stock_evaluation',
        'valid_ratio': 0.2,
        'do_normalization': True,
        'seed': 42
    }
    
    # Configuration for models
    models_config = {
        'latent_dim': 64,
        'hidden_dim': 128,
        'lr': 0.0002,
        'wgan_lr': 0.00005,
        'n_critic': 5,
        'clip_value': 0.01
    }
    
    # Initialize evaluator
    evaluator = UnifiedEvaluator(experiment_name="TimeSeries_Generation_Comprehensive_Evaluation")
    
    # Run complete evaluation
    results = evaluator.run_complete_evaluation(
        dataset_config=dataset_config,
        models_config=models_config,
        num_samples=500
    )
    
    # Print summary
    show_with_start_divider("EVALUATION SUMMARY") # Using utility function
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        if "error" in model_results:
            print(f"  Error: {model_results['error']}")
        else:
            print(f"  Training Time: {model_results.get('training_time', 'N/A'):.2f}s")
            print(f"  Generation Time (500 samples): {model_results.get('generation_time_500_samples', 'N/A'):.4f}s")
            print(f"  MDD: {model_results.get('mdd', 'N/A'):.4f}")
            print(f"  MD: {model_results.get('md', 'N/A'):.4f}")
            print(f"  SDD: {model_results.get('sdd', 'N/A'):.4f}")
            print(f"  ICD (Euclidean): {model_results.get('icd_euclidean', 'N/A'):.4f}")
            print(f"  ICD (DTW): {model_results.get('icd_dtw', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
