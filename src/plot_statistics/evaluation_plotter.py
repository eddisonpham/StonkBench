"""
Main plotting class for evaluation results.
"""

import sys
from pathlib import Path
import shutil
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from utils.metric_plot_utils import (
    find_sequence_folders, load_evaluation_data, load_all_sequence_data, 
    create_output_directory
)
from utils.metric_plot_classes_utils import (
    PerformancePlot, DistributionPlot, SimilarityPlot, 
    StylizedFactsPlot, CombinedVisualizationPlot, UtilityPlot,
    SequenceLengthComparisonPlot
)

class EvaluationPlotter:
    """
    Comprehensive plotting class for evaluation results.
    """
    
    def __init__(self, data: Dict[str, Any], output_dir: Path, eval_results_dir: str = None):
        """
        Initialize the plotter with evaluation data.
        """
        self.data = data
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.performance_plot = PerformancePlot(data, self.output_dir)
        self.distribution_plot = DistributionPlot(data, self.output_dir)
        self.similarity_plot = SimilarityPlot(data, self.output_dir)
        self.stylized_facts_plot = StylizedFactsPlot(data, self.output_dir)
        self.combined_visualization_plot = CombinedVisualizationPlot(data, self.output_dir, eval_results_dir=eval_results_dir)
        
        # Add utility plot if utility data exists
        if any('utility' in model_data for model_data in data.values()):
            self.utility_plot = UtilityPlot(data, self.output_dir)
        else:
            self.utility_plot = None
        
    def generate_all_plots(self) -> None:
        """Generate all plots using the unified plot classes."""

        print("Generating combined visualization plot...")
        self.combined_visualization_plot.plot()
        
        print("Generating performance metrics plot...")
        self.performance_plot.plot()
        
        print("Generating distribution metrics plot...")
        self.distribution_plot.plot()
        
        print("Generating similarity metrics plot...")
        self.similarity_plot.plot()
        
        print("Generating stylized facts plots...")
        self.stylized_facts_plot.plot()
        
        if self.utility_plot:
            print("Generating utility plots...")
            self.utility_plot.plot()
        
        print(f"All plots saved to {self.output_dir}")


def main():
    """Main function to generate all evaluation plots."""
    try:
        print("Finding sequence folders...")
        seq_folders = find_sequence_folders()
        print(f"Found {len(seq_folders)} sequence folders: {[Path(f).name for f in seq_folders]}")
        
        output_base_dir = Path("evaluation_plots")
        output_base_dir.mkdir(exist_ok=True)
        
        # Clear output directory
        print(f"Clearing contents of '{output_base_dir}' before generating new plots...")
        for item in output_base_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        
        # Process each sequence folder
        for seq_folder in seq_folders:
            seq_name = Path(seq_folder).name
            print(f"\nProcessing {seq_name}...")
            
            print(f"Loading evaluation data for {seq_name}...")
            data = load_evaluation_data(seq_folder)
            print(f"Loaded data for {len(data)} models: {list(data.keys())}")
            
            # Create output directory for this sequence length
            seq_output_dir = output_base_dir / seq_name
            seq_output_dir.mkdir(exist_ok=True, parents=True)
            
            print(f"Initializing plotter for {seq_name}...")
            plotter = EvaluationPlotter(data, seq_output_dir, eval_results_dir=seq_folder)
            
            print(f"Generating all plots for {seq_name}...")
            plotter.generate_all_plots()
        
        # Generate sequence length comparison plots
        print("\nGenerating sequence length comparison plots...")
        all_seq_data = load_all_sequence_data(exclude_seq_52=True)
        if len(all_seq_data) > 1:  # Only if we have multiple sequence lengths
            seq_comp_output_dir = output_base_dir / "sequence_length_comparison"
            seq_comp_output_dir.mkdir(exist_ok=True, parents=True)
            seq_comparison_plot = SequenceLengthComparisonPlot(all_seq_data, seq_comp_output_dir)
            seq_comparison_plot.plot()
            print(f"Sequence length comparison plots saved to {seq_comp_output_dir}")
        
        print("\nAll plots generated successfully!")
        print(f"Plots saved to: {output_base_dir}")
        
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
