# StonkBench Data Flow (Current)

## End-to-end

```text
src/data_downloader.py
  -> data/combined_data.csv

src/data_preprocessing.py
  -> data/preprocessed/dl_set.pt
  -> data/preprocessed/statsmodel_set.pt

src/experiments/run_benchmark.py
  -> src/experiments/<model>/artifacts/*.pt

src/unified_evaluator.py
  -> results/seq_<L>/<model>/metrics.json
```

## Transform logic

- Price columns (non-`_volume`) are transformed to log returns.
- Volume columns (`*_volume`) are transformed to log(volume).
- Train/test split is 80/20.
- DL preprocessing uses sliding windows with `window_size=21`, `stride=1`.
- A temporal gap of `window_size - 1` is inserted between train and test to avoid overlap leakage.
- Statistical preprocessing keeps full transformed train/test series (no windowing for fit).

## Config contract

`configs/dataset_cfgs.yaml` points to preprocessed datasets:

```yaml
deep_learning_dataset_cfg:
  preprocessed_data_path: "data/preprocessed/dl_set.pt"

statistical_dataset_cfg:
  preprocessed_data_path: "data/preprocessed/statsmodel_set.pt"
```

## Notes

- Generation/evaluation no longer perform CSV-time preprocessing.
- `src/utils/preprocessing_utils.py` is intentionally minimal and only keeps log-return inversion used by utility metrics.
