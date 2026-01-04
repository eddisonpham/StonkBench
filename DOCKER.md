# Docker Orchestration Guide

This project uses Docker Compose to orchestrate the entire ML pipeline from data download to evaluation and plotting.

## Architecture

The pipeline consists of the following services, executed in sequence:

1. **data-download**: Downloads and preprocesses time series data
2. **gen-param**: Generates synthetic data using parametric models
3. **gen-nonparam**: Generates synthetic data using non-parametric (deep learning) models
4. **eval**: Evaluates all generated data using unified evaluator
5. **plot**: Generates publication-ready figures from evaluation results

## Quick Start

### Build the base image

```bash
docker-compose build base
```

### Run the entire pipeline

```bash
docker-compose up
```

This will run all services in sequence based on their dependencies.

### Run specific services

```bash
# Run only data download
docker-compose up data-download

# Run data download and parametric generation
docker-compose up data-download gen-param

# Run everything except plotting
docker-compose up --scale plot=0
```

## Environment Variables

You can customize the pipeline using environment variables in a `.env` file or by exporting them:

```bash
# Data download settings
export INDEX=spxusd          # Index to download (default: spxusd)
export YEAR=2023              # Year(s) to download (default: 2023)

# Sequence lengths to generate
export SEQ_LENGTHS="52 60 120 180 240 300"

# Number of samples per model
export NUM_SAMPLES=500

# Random seed
export SEED=42

# Number of epochs for deep learning models
export NUM_EPOCHS=15

# Device for training (cpu or cuda)
export DEVICE=cpu

# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# Skip utility evaluation (faster, but less complete)
export SKIP_UTILITY=""

# Max samples for evaluation
export MAX_SAMPLES=""
```

## Volume Mounts

The following directories are mounted as volumes:

- `./data` → `/data` (raw and processed data)
- `./generated_data` → `/generated_data` (generated synthetic data)
- `./results` → `/results` (evaluation results)
- `./evaluation_plots` → `/evaluation_plots` (generated figures)
- `./configs` → `/app/configs` (read-only configuration files)

## Service Details

### data-download

Downloads time series data from histdata and processes it.

**Dependencies**: None

**Outputs**: 
- `/data/raw/` - Raw downloaded data
- `/data/processed/` - Processed CSV files

### gen-param

Generates synthetic data using parametric models (GBM, OU Process, MJD, DEJD, GARCH11, BlockBootstrap).

**Dependencies**: `data-download`

**Outputs**: 
- `/generated_data/<ModelName>/<ModelName>_seq_<L>.pt` - Generated artifacts

**Health Check**: Verifies `GBM_seq_52.pt` exists

### gen-nonparam

Generates synthetic data using non-parametric models (QuantGAN, TimeVAE).

**Dependencies**: `gen-param` (must complete successfully)

**Outputs**: 
- `/generated_data/<ModelName>/<ModelName>_seq_<L>.pt` - Generated artifacts

**Health Check**: Verifies `QuantGAN_seq_52.pt` exists

### eval

Evaluates all generated data using the parallelizer script, which calls the unified evaluator.

**Dependencies**: `gen-param` and `gen-nonparam` (both must complete successfully)

**Outputs**: 
- `/results/seq_<L>/<ModelName>/metrics.json` - Evaluation metrics
- `/results/seq_<L>/<ModelName>/visualizations/` - Visualization files

**Health Check**: Verifies `seq_52/GBM/metrics.json` exists

**Note**: Uses `parallelizer_script.py` which can be extended for parallel evaluation processing.

### plot

Generates publication-ready figures from evaluation results.

**Dependencies**: `eval` (must complete successfully)

**Outputs**: 
- `/evaluation_plots/main_experiment/` - Main experiment figures
- `/evaluation_plots/ablation_studies/` - Ablation study figures
- `/evaluation_plots/appendix/` - Appendix figures

## Troubleshooting

### View logs for a specific service

```bash
docker-compose logs -f gen-param
```

### Rebuild after code changes

```bash
docker-compose build base
docker-compose up
```

### Run a single service in isolation

```bash
# Build first
docker-compose build base

# Run with specific command
docker-compose run --rm gen-param python src/scripts/generate_parametric_data.py --seq_lengths 52
```

### Clean up

```bash
# Stop all containers
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## Best Practices

1. **Layer Caching**: The Dockerfile is optimized for layer caching. Dependencies are installed before copying source code.

2. **Non-root User**: The base image runs as a non-root user (`appuser`) for security.

3. **Health Checks**: Services include health checks to verify successful completion.

4. **Read-only Mounts**: Configuration files are mounted read-only to prevent accidental modification.

5. **Service Dependencies**: Services use `condition: service_completed_successfully` to ensure proper execution order.

6. **Volume Persistence**: All data persists in local directories, so you can inspect results outside Docker.

## Development

For local development without Docker, ensure you have:

- Python 3.11+
- All dependencies from `requirements.txt`
- Proper `PYTHONPATH` or project root in `sys.path`

The scripts can be run directly:

```bash
python src/data_downloader.py
python src/scripts/generate_parametric_data.py --seq_lengths 52
python src/scripts/generate_non_parametric_data.py --seq_lengths 52
python src/unified_evaluator.py
python src/plot_statistics/evaluation_plotter.py
```

