"""
Unified evaluation script.

Taxonomies of evaluation:
- Diversity: 
    - Synthetic to Synthetic Similarity (STS)
    - Intra-class Distance (ICD) 
        - Dynamic Time Warp
        - Euclidean Distance
- Efficiency:
    - Generation Time (ms)
- Fidelity:
    - Feature-based metrics: 
        - Marginal Distribution Distance
        - Autocorrelation Difference
        - Skewness Difference
        - Kurtosis Difference
        - Standard Deviation Difference
        - Mean Difference
    - Visual Assessment:
        - t-SNE
        - Distribution Plots
- Utility: Based on Hedging Effectiveness Task
    - Test Synthetic, Train Real (TSTR)
        - Replication Error
        - ERM, CVar
        - Max Profits
    - Augmentation Test
    - Algorithm Comparison
"""

