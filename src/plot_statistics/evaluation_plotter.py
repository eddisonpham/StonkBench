"""
Main plotting class for evaluation results.
"""

import sys
from pathlib import Path
import shutil
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from utils.metric_plot_utils import find_latest_evaluation_folder, load_evaluation_data, create_output_directory
from utils.metric_plot_classes_utils import (
    PerformancePlot, DistributionPlot, SimilarityPlot, 
    StylizedFactsPlot, CombinedVisualizationPlot
)

class EvaluationPlotter:
    """
    Comprehensive plotting class for evaluation results.
    """
    
    def __init__(self, data: Dict[str, Any], output_dir: str = "evaluation_plots", eval_results_dir: str = None):
        """
        Initialize the plotter with evaluation data.
        """
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.performance_plot = PerformancePlot(data, self.output_dir)
        self.distribution_plot = DistributionPlot(data, self.output_dir)
        self.similarity_plot = SimilarityPlot(data, self.output_dir)
        self.stylized_facts_plot = StylizedFactsPlot(data, self.output_dir)
        self.combined_visualization_plot = CombinedVisualizationPlot(data, self.output_dir, eval_results_dir=eval_results_dir)
        
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
        
        print(f"All plots saved to {self.output_dir}")


def main():
    """Main function to generate all evaluation plots."""
    try:
        print("Finding latest evaluation folder...")
        latest_folder = find_latest_evaluation_folder()
        print(f"Latest evaluation folder: {latest_folder}")
        
        print("Loading evaluation data...")
        data = load_evaluation_data(latest_folder)
        print(f"Loaded data for {len(data)} models: {list(data.keys())}")
        
        output_dir = "evaluation_plots"
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Clearing contents of '{output_dir}' before generating new plots...")
        for item in output_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        print("Creating output directory...")
        output_dir_actual = create_output_directory()
        print(f"Output directory: {output_dir_actual}")
        
        print("Initializing plotter...")
        plotter = EvaluationPlotter(data, output_dir_actual, eval_results_dir=latest_folder)
        
        print("Generating all plots...")
        plotter.generate_all_plots()
        
        print("All plots generated successfully!")
        print(f"Plots saved to: {output_dir_actual}")
        
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
