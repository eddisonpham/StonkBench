"""
Main plotting script for evaluation results.

This script generates all paper figures by coordinating the PaperFigureGenerator.
"""

import sys
from pathlib import Path
import shutil
import traceback

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.plot_statistics.paper_figures import PaperFigureGenerator


def clear_output_directory(output_dir: Path) -> None:
    """
    Clear all contents of the output directory.
    
    Args:
        output_dir: Directory to clear.
    """
    if not output_dir.exists():
        return
    
    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def main():
    """Main function to generate all evaluation plots."""
    try:
        print("=" * 80)
        print("Generating Paper Figures")
        print("=" * 80)
        
        output_base_dir = Path("evaluation_plots")
        output_base_dir.mkdir(exist_ok=True)
        
        # Clear output directory
        print(f"\nClearing contents of '{output_base_dir}' before generating new plots...")
        clear_output_directory(output_base_dir)
        
        # Generate paper figures
        generator = PaperFigureGenerator(
            results_dir="results", 
            output_dir=str(output_base_dir)
        )
        generator.generate_all_figures()
        
        print("\n" + "=" * 80)
        print("All figures generated successfully!")
        print(f"Figures saved to: {output_base_dir}")
        print("=" * 80)

    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
