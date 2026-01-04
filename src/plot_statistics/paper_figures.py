"""
Module for generating publication-ready figures for main experiments and ablation studies.

This module coordinates the generation of all paper figures using specialized generator classes.
"""

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.figure_utils import (
    load_sequence_data,
    load_ablation_data,
)
from src.plot_statistics.figure_generators import (
    Figure1Generator,
    Figure2Generator,
    Figure3Generator,
    Figure4Generator,
    Figure5Generator,
    Figure6Generator,
    Figure7Generator,
)


class PaperFigureGenerator:
    """
    Generator class for creating publication-ready statistical and visual plots for experiments.
    
    This class coordinates the generation of all figures by delegating to specialized
    figure generator classes.
    """

    def __init__(self, results_dir: str = "results", output_dir: str = "evaluation_plots"):
        """
        Initialize the generator, load main experiment data and ablation data.

        Args:
            results_dir: Directory containing results.
            output_dir: Output directory for figures.
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data
        self.main_data = load_sequence_data(self.results_dir, seq_len=52)
        self.ablation_data = load_ablation_data(self.results_dir)

    def generate_all_figures(self):
        """Generate all figures for main experiments, ablation studies, and appendix."""
        # Create output directories
        main_dir = self.output_dir / "main_experiment"
        main_dir.mkdir(exist_ok=True, parents=True)
        ablation_dir = self.output_dir / "ablation_studies"
        ablation_dir.mkdir(exist_ok=True, parents=True)
        appendix_dir = self.output_dir / "appendix"
        appendix_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize generators
        generators = {
            'main': {
                'figure_1': Figure1Generator(
                    self.main_data, self.ablation_data, self.results_dir, self.output_dir
                ),
                'figure_2': Figure2Generator(
                    self.main_data, self.ablation_data, self.results_dir, self.output_dir
                ),
            },
            'ablation': {
                'figure_3': Figure3Generator(
                    self.main_data, self.ablation_data, self.results_dir, self.output_dir
                ),
            },
            'appendix': {
                'figure_4': Figure4Generator(
                    self.main_data, self.ablation_data, self.results_dir, self.output_dir
                ),
                'figure_5': Figure5Generator(
                    self.main_data, self.ablation_data, self.results_dir, self.output_dir
                ),
                'figure_6': Figure6Generator(
                    self.main_data, self.ablation_data, self.results_dir, self.output_dir
                ),
                'figure_7': Figure7Generator(
                    self.main_data, self.ablation_data, self.results_dir, self.output_dir
                ),
            }
        }
        
        # Generate main experiment figures
        print("Generating main experiment figures...")
        generators['main']['figure_1'].generate(main_dir)
        generators['main']['figure_2'].generate(main_dir)
        
        # Generate ablation study figures
        print("Generating ablation study figures...")
        generators['ablation']['figure_3'].generate(ablation_dir)
        
        # Generate appendix figures
        print("Generating appendix figures...")
        generators['appendix']['figure_4'].generate(appendix_dir)
        generators['appendix']['figure_5'].generate(appendix_dir)
        generators['appendix']['figure_6'].generate(appendix_dir)
        generators['appendix']['figure_7'].generate(appendix_dir)
        
        print(f"\nAll figures generated successfully! Figures saved to: {self.output_dir}")


if __name__ == "__main__":
    generator = PaperFigureGenerator()
    generator.generate_all_figures()
