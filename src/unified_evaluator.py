"""
Unified Evaluator Pipeline

This script provides a unified pipeline to train and evaluate both parametric
and non-parametric time series generative models.

Usage:
    python -m src.unified_evaluator --seq_length 103 --num_samples 1000 --num_epochs 10
    python -m src.unified_evaluator --seq_length None --num_samples 500 --num_epochs 5
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.models.base.base_model import DeepLearningModel
from src.models.parametric.gbm import GeometricBrownianMotion
from src.models.parametric.ou_process import OUProcess
from src.models.parametric.merton_jump_diffusion import MertonJumpDiffusion
from src.models.parametric.garch11 import GARCH11
from src.models.parametric.de_jump_diffusion import DoubleExponentialJumpDiffusion
from src.models.non_parametric.block_bootstrap import BlockBootstrap
from src.models.non_parametric.quant_gan import QuantGAN
from src.models.non_parametric.time_vae import TimeVAE
# from src.models.non_parametric.takahashi import TakahashiDiffusion  # Commented out as requested

from src.utils.display_utils import show_with_start_divider, show_with_end_divider
from src.utils.preprocessing_utils import (
    create_dataloaders,
    preprocess_data,
    sliding_window_view,
    find_length,
)
from src.utils.configs_utils import get_dataset_cfgs
from src.utils.evaluation_classes_utils import (
    DiversityEvaluator,
    FidelityEvaluator,
    RuntimeEvaluator,
    StylizedFactsEvaluator,
    VisualAssessmentEvaluator,
    UtilityEvaluator
)


class UnifiedEvaluator:
    """
    Unified evaluator class to evaluate both parametric and non-parametric models.
    """

    def __init__(
        self,
        experiment_name: str,
        parametric_dataset_cfgs: Dict[str, Any],
        non_parametric_dataset_cfgs: Dict[str, Any],
        results_dir: Optional[Path] = None,
        seq_length: Optional[int] = None
    ):
        """
        Initialize the unified evaluator.

        Args:
            experiment_name: Name of the experiment
            parametric_dataset_cfgs: Configuration for parametric models
            non_parametric_dataset_cfgs: Configuration for non-parametric models
            seq_length: Sequence length for time series. If None, will be determined using autocorrelation
            results_dir: Directory to save results. If None, creates a timestamped directory
        """
        self.parametric_dataset_cfgs = parametric_dataset_cfgs
        self.non_parametric_dataset_cfgs = non_parametric_dataset_cfgs
        self.experiment_name = experiment_name
        self.seq_length = seq_length

        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if results_dir is None:
            self.results_dir = project_root / "results" / f"evaluation_{self.timestamp}"
        else:
            self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model,
        model_name: str,
        real_data: np.ndarray,
        train_data,
        generation_kwargs: Dict[str, Any],
        valid_loader=None,
        fit_kwargs: Dict[str, Any] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Unified evaluation for both parametric and non-parametric models.

        Args:
            model: The generative model to evaluate
            model_name: Name of the model for logging
            real_data: Real data for comparison (numpy array)
            train_data: Training data (tensor for parametric, DataLoader for non-parametric)
            valid_loader: Validation DataLoader (for non-parametric models)
            generation_kwargs: Kwargs for model.generate()
            fit_kwargs: Optional kwargs for model.fit() (e.g., num_epochs)
            seed: Random seed

        Returns:
            Dictionary containing all evaluation metrics
        """
        show_with_start_divider(f"Evaluating {model_name}")
        num_samples = generation_kwargs.get('num_samples', 500)
        
        model_dir = self.results_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        evaluation_results: Dict[str, Any] = {}

        print(f"Training {model_name}...")
        # --- For QuantGAN, skip training/generation and load from file ---
        is_quantgan = model_name.lower() == "quantgan"
        loaded_gen_numpy = None
        loaded_gen_torch = None
        if is_quantgan:
            length = generation_kwargs.get('generation_length', self.seq_length)
            # Find output file (assume outputs folder is at project root)
            outputs_path = project_root / "outputs"
            quantgan_fname = outputs_path / f"QuantGAN_fakes_{length}.pt"
            if not quantgan_fname.exists():
                raise FileNotFoundError(f"Could not find QuantGAN fake data at: {quantgan_fname}")
            print(f"Loading QuantGAN generated data from: {quantgan_fname}")
            loaded_gen_torch = torch.load(quantgan_fname)
            if loaded_gen_torch.shape[0] > num_samples:
                loaded_gen_torch = loaded_gen_torch[:num_samples]
            loaded_gen_numpy = loaded_gen_torch.cpu().numpy()
            # Runtime evaluation is not meaningful, but to keep result fields:
            runtime_results = {'generation_time_sec': None}
            evaluation_results.update(runtime_results)
            generated_data = loaded_gen_numpy
        else:
            if isinstance(model, DeepLearningModel):
                num_epochs = fit_kwargs.get('num_epochs', 10) if fit_kwargs else 10
                model.fit(train_data)
            else:
                model.fit(train_data)

            print(f"\nGenerating {num_samples} samples...")
            runtime_evaluator = RuntimeEvaluator(
                generate_func=model.generate,
                generation_kwargs=generation_kwargs
            )
            runtime_results = runtime_evaluator.evaluate()
            evaluation_results.update(runtime_results)

            generated_data = model.generate(**generation_kwargs)
            if isinstance(generated_data, torch.Tensor):
                generated_data = generated_data.cpu().numpy()

        # Convert to numpy if needed
        if isinstance(real_data, torch.Tensor):
            real_data = real_data.cpu().numpy()
        else:
            real_data = np.asarray(real_data)

        # Ensure real_data has the same shape as generated_data for comparison
        if real_data.ndim == 1:
            window_size = generation_kwargs.get('generation_length', 1)
            real_data = sliding_window_view(torch.from_numpy(real_data), window_size, 1).numpy()
        
        # Sample same number of real samples as generated samples
        if real_data.shape[0] > num_samples:
            idx = np.random.permutation(real_data.shape[0])[:num_samples]
            real_data = real_data[idx]

        print(f"Generated data shape: {generated_data.shape}")
        print(f"Real data shape: {real_data.shape}")

        evaluators = [
            FidelityEvaluator(real_data, generated_data),
            DiversityEvaluator(real_data, generated_data),
            StylizedFactsEvaluator(real_data, generated_data),
            VisualAssessmentEvaluator(real_data, generated_data, model_dir)
        ]

        for evaluator in evaluators:
            print(f"Computing {evaluator.__class__.__name__}...")
            try:
                results = evaluator.evaluate()
                if results is not None:
                    evaluation_results.update(results)
            except Exception as e:
                print(f"Warning: {evaluator.__class__.__name__} failed: {e}")

        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        print(f"Evaluation completed for {model_name} (results saved at {metrics_path}).")

        return evaluation_results
    
    def evaluate_utility(
        self,
        model_name: str,
        generated_data: np.ndarray,
        real_train_log_returns: torch.Tensor,
        real_val_log_returns: torch.Tensor,
        real_test_log_returns: torch.Tensor,
        real_train_initial: torch.Tensor,
        real_val_initial: torch.Tensor,
        real_test_initial: torch.Tensor,
        generation_length: int,
        num_epochs: int = 40,
        batch_size: int = 128,
        learning_rate: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Evaluate utility-based metrics for a model using deep hedging evaluation.
        
        Args:
            model_name: Name of the model
            generated_data: Generated synthetic data (num_samples, generation_length)
            real_train_log_returns: Real training log returns
            real_val_log_returns: Real validation log returns
            real_test_log_returns: Real test log returns
            real_train_initial: Real training initial prices
            real_val_initial: Real validation initial prices
            real_test_initial: Real test initial prices
            generation_length: Length of generated sequences
            num_epochs: Number of epochs for hedger training
            batch_size: Batch size for hedger training
            learning_rate: Learning rate for hedger training
            
        Returns:
            Dictionary containing utility evaluation results
        """
        print(f"\nEvaluating utility metrics for {model_name}...")
        
        # Convert generated data to torch tensor if needed
        if isinstance(generated_data, np.ndarray):
            generated_data = torch.from_numpy(generated_data).float()
        
        # Split generated data into train/val/test (80/10/10)
        num_samples = generated_data.shape[0]
        train_end = int(num_samples * 0.8)
        val_end = int(num_samples * 0.9)
        
        synthetic_train_log_returns = generated_data[:train_end]
        synthetic_val_log_returns = generated_data[train_end:val_end]
        synthetic_test_log_returns = generated_data[val_end:]
        
        # Create synthetic initial values (use mean of real initial values)
        mean_initial = float(real_train_initial.mean().item())
        device = real_train_initial.device
        
        synthetic_train_initial = torch.ones(train_end, device=device) * mean_initial
        synthetic_val_initial = torch.ones(val_end - train_end, device=device) * mean_initial
        synthetic_test_initial = torch.ones(num_samples - val_end, device=device) * mean_initial
        
        # Run utility evaluation
        utility_evaluator = UtilityEvaluator(
            real_train_log_returns=real_train_log_returns,
            real_val_log_returns=real_val_log_returns,
            real_test_log_returns=real_test_log_returns,
            synthetic_train_log_returns=synthetic_train_log_returns,
            synthetic_val_log_returns=synthetic_val_log_returns,
            synthetic_test_log_returns=synthetic_test_log_returns,
            real_train_initial=real_train_initial,
            real_val_initial=real_val_initial,
            real_test_initial=real_test_initial,
            synthetic_train_initial=synthetic_train_initial,
            synthetic_val_initial=synthetic_val_initial,
            synthetic_test_initial=synthetic_test_initial,
            seq_length=generation_length,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        try:
            utility_results = utility_evaluator.evaluate()
            return utility_results
        except Exception as e:
            print(f"Warning: Utility evaluation failed for {model_name}: {e}")
            return {"error": str(e)}

    def run_complete_evaluation(
        self, 
        num_samples: int = 500, 
        num_epochs: int = 10,
        batch_size: int = 32,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run complete evaluation on all models.

        Args:
            num_samples: Number of samples to generate per model
            num_epochs: Number of training epochs for non-parametric models
            batch_size: Batch size for DataLoaders
            seed: Random seed

        Returns:
            Dictionary containing results for all models
        """
        show_with_start_divider("Starting Complete Evaluation Pipeline")
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Data reduction ratio: we will randomly sample this fraction from each set
        data_reduction_ratio = 0.24

        # Preprocess parametric data
        train_data_para, valid_data_para, test_data_para, _, _, _ = preprocess_data(
            self.parametric_dataset_cfgs
        )

        # Randomly sample from each dataset (no overlap needed since splits are already disjoint)
        def random_subset(data, n, axis=0, always_return_tensor=True):
            """Returns a random subset of n rows from data (either np.ndarray or torch.Tensor)."""
            if n <= 0 or n >= data.shape[axis]:
                return data
            if isinstance(data, np.ndarray):
                idx = np.random.choice(data.shape[axis], size=n, replace=False)
                idx.sort()
                return data[idx]
            elif isinstance(data, torch.Tensor):
                idx = torch.randperm(data.shape[axis])[:n].sort()[0]
                return data.index_select(axis, idx)
            else:
                raise TypeError("Unsupported data type for random_subset")
        
        train_size = int(train_data_para.shape[0] * data_reduction_ratio)
        valid_size = int(valid_data_para.shape[0] * data_reduction_ratio)
        test_size = int(test_data_para.shape[0] * data_reduction_ratio)

        train_data_para = random_subset(train_data_para, train_size)
        valid_data_para = random_subset(valid_data_para, valid_size)
        test_data_para = random_subset(test_data_para, test_size)

        print(f"  - Parametric train data shape: {train_data_para.shape}")
        print(f"  - Parametric valid data shape: {valid_data_para.shape}")
        print(f"  - Parametric test data shape: {test_data_para.shape}")

        # Preprocess non-parametric data
        # Override seq_length in config if specified
        non_para_cfg = self.non_parametric_dataset_cfgs.copy()
        non_para_cfg['seq_length'] = self.seq_length
        print(f"  - Using specified sequence length: {self.seq_length}")

        (
            train_data_non_para,
            valid_data_non_para,
            test_data_non_para,
            train_initial_non_para,
            valid_initial_non_para,
            test_initial_non_para
        ) = preprocess_data(non_para_cfg)

        train_size = int(train_data_non_para.shape[0] * data_reduction_ratio)
        valid_size = int(valid_data_non_para.shape[0] * data_reduction_ratio)
        test_size = int(test_data_non_para.shape[0] * data_reduction_ratio)

        # Randomly sample from data and their corresponding initial values (keep sync!)
        def split_data_and_initial(data, initial, n):
            if n <= 0 or n >= data.shape[0]:
                return data, initial
            idx = np.random.choice(data.shape[0], size=n, replace=False)
            idx.sort()
            if isinstance(data, np.ndarray):
                return data[idx], initial[idx]
            elif isinstance(data, torch.Tensor):
                idx_torch = torch.tensor(idx, dtype=torch.long, device=data.device)
                return data.index_select(0, idx_torch), initial.index_select(0, idx_torch)
            else:
                raise TypeError("Unsupported data type in split_data_and_initial")

        train_data_non_para, train_initial_non_para = split_data_and_initial(train_data_non_para, train_initial_non_para, train_size)
        valid_data_non_para, valid_initial_non_para = split_data_and_initial(valid_data_non_para, valid_initial_non_para, valid_size)
        test_data_non_para, test_initial_non_para = split_data_and_initial(test_data_non_para, test_initial_non_para, test_size)

        train_loader_non_para, valid_loader_non_para, test_loader_non_para = create_dataloaders(
            train_data_non_para,
            valid_data_non_para,
            test_data_non_para,
            batch_size=batch_size,
            train_seed=seed,
            valid_seed=seed,
            test_seed=seed,
            train_initial=train_initial_non_para,
            valid_initial=valid_initial_non_para,
            test_initial=test_initial_non_para,
        )

        generation_length = train_data_non_para.shape[1]
        print(f"  - Non-parametric train data shape: {train_data_non_para.shape} (random sample, {data_reduction_ratio*100:.1f}%)")
        print(f"  - Non-parametric valid data shape: {valid_data_non_para.shape} (random sample, {data_reduction_ratio*100:.1f}%)")
        print(f"  - Non-parametric test data shape: {test_data_non_para.shape} (random sample, {data_reduction_ratio*100:.1f}%)")
        print(f"  - Generation length: {generation_length}")

        # Initialize parametric models
        parametric_models = {}
        parametric_models["GBM"] = GeometricBrownianMotion()
        parametric_models["OU Process"] = OUProcess()
        parametric_models["MJD"] = MertonJumpDiffusion()
        parametric_models["GARCH11"] = GARCH11()
        parametric_models["DEJD"] = DoubleExponentialJumpDiffusion()
        parametric_models["BlockBootstrap"] = BlockBootstrap(block_size=generation_length)

        non_parametric_models = {}

        # --- Patch for QuantGAN: This will not instantiate the model, but "fake" as if it is present ---
        # non_parametric_models["TimeVAE"] = TimeVAE(
        #     seq_len=generation_length,
        #     input_dim=1
        # )
        # Instead of : 
        # non_parametric_models["QuantGAN"] = QuantGAN(
        #     seq_len=generation_length,
        # )

        # So we just put a `"QuantGAN": None` as a placeholder for our logic in evaluate_model
        non_parametric_models["QuantGAN"] = None

        all_results = {}

        generation_kwargs_para = {
            'num_samples': num_samples,
            'generation_length': self.seq_length,
            'seed': seed
        }
        for model_name, model in parametric_models.items():
            results = self.evaluate_model(
                model=model,
                model_name=model_name,
                real_data=test_data_para.numpy() if isinstance(test_data_para, torch.Tensor) else test_data_para,
                train_data=train_data_para,
                generation_kwargs=generation_kwargs_para,
                valid_loader=None,
                fit_kwargs=None,
                seed=seed
            )
            
            print(f"\nRunning utility evaluation for {model_name}...")
            utility_num_samples = max(num_samples, 1000)
            utility_generation_kwargs = {
                'num_samples': utility_num_samples,
                'generation_length': self.seq_length,
                'seed': seed
            }
            generated_data = model.generate(**utility_generation_kwargs)
            if isinstance(generated_data, torch.Tensor):
                generated_data = generated_data.cpu().numpy()
            
            utility_results = self.evaluate_utility(
                model_name=model_name,
                generated_data=generated_data,
                real_train_log_returns=train_data_non_para,
                real_val_log_returns=valid_data_non_para,
                real_test_log_returns=test_data_non_para,
                real_train_initial=train_initial_non_para,
                real_val_initial=valid_initial_non_para,
                real_test_initial=test_initial_non_para,
                generation_length=self.seq_length,
                num_epochs=40,
                batch_size=64,
                learning_rate=1e-3
            )
            results['utility'] = utility_results
            
            # Update metrics.json file with utility results
            model_dir = self.results_dir / model_name
            metrics_path = model_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Updated metrics.json with utility results for {model_name}")
            
            all_results[model_name] = results

        generation_kwargs_non_para = {
            'num_samples': num_samples,
            'generation_length': self.seq_length,
            'seed': seed
        }
        fit_kwargs_non_para = {'num_epochs': num_epochs}
        for model_name, model in non_parametric_models.items():
            # Only valid for QuantGAN patch -- handle loading the generated data from file 
            results = self.evaluate_model(
                model=model,
                model_name=model_name,
                real_data=test_data_non_para.numpy() if isinstance(test_data_non_para, torch.Tensor) else test_data_non_para,
                train_data=train_loader_non_para,
                generation_kwargs=generation_kwargs_non_para,
                valid_loader=valid_loader_non_para,
                fit_kwargs=fit_kwargs_non_para,
                seed=seed
            )
            
            print(f"\nRunning utility evaluation for {model_name}...")
            # --- Instead of model.generate, re-use loaded QuantGAN data for utility too ---
            length = self.seq_length
            outputs_path = project_root / "outputs"
            quantgan_fname = outputs_path / f"QuantGAN_fakes_{length}.pt"
            if not quantgan_fname.exists():
                raise FileNotFoundError(f"Could not find QuantGAN fake data at: {quantgan_fname}")
            loaded_gen_torch = torch.load(quantgan_fname)
            utility_num_samples = max(num_samples, 1000)
            # If more than needed, take only first utility_num_samples
            if loaded_gen_torch.shape[0] > utility_num_samples:
                loaded_gen_torch = loaded_gen_torch[:utility_num_samples]
            generated_data = loaded_gen_torch.cpu().numpy()

            utility_results = self.evaluate_utility(
                model_name=model_name,
                generated_data=generated_data,
                real_train_log_returns=train_data_non_para,
                real_val_log_returns=valid_data_non_para,
                real_test_log_returns=test_data_non_para,
                real_train_initial=train_initial_non_para,
                real_val_initial=valid_initial_non_para,
                real_test_initial=test_initial_non_para,
                generation_length=self.seq_length,
                num_epochs=40,
                batch_size=64,
                learning_rate=1e-3
            )
            results['utility'] = utility_results
            
            # Update metrics.json file with utility results
            model_dir = self.results_dir / model_name
            metrics_path = model_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Updated metrics.json with utility results for {model_name}")
            
            all_results[model_name] = results
        # Save complete results
        results_file = self.results_dir / "complete_evaluation.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        show_with_end_divider("EVALUATION COMPLETE")
        print(f"Results saved to: {results_file}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Sequence length used: {self.seq_length}")

        return all_results


def main():
    """Main function to run the evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Unified Evaluator Pipeline')
    parser.add_argument(
        '--seq_length',
        type=lambda x: None if x.lower() == 'none' else int(x),
        default=None,
        help='Sequence length for time series. If None, will be determined using autocorrelation'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=500,
        help='Number of samples to generate per model (default: 500)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs for non-parametric models (default: 10)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for DataLoaders (default: 32)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='TimeSeries_Generation_Comprehensive_Evaluation',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='Directory to save results. If None, creates a timestamped directory'
    )

    args = parser.parse_args()

    non_parametric_dataset_cfgs, parametric_dataset_cfgs = get_dataset_cfgs()

    evaluator = UnifiedEvaluator(
        experiment_name=args.experiment_name,
        parametric_dataset_cfgs=parametric_dataset_cfgs,
        non_parametric_dataset_cfgs=non_parametric_dataset_cfgs,
        seq_length=args.seq_length,
        results_dir=Path(args.results_dir) if args.results_dir else None
    )

    evaluator.run_complete_evaluation(
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )

if __name__ == "__main__":
    main()

