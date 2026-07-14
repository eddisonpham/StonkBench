# StonkBench Data Flow (Current)

## End-to-end

```text
src/data_downloader.py
  -> data/combined_data.csv

src/data_preprocessing.py
  -> data/preprocessed/dl_set.pt
  -> data/preprocessed/statsmodel_set.pt

src/experiments/run_benchmark.py
  -> output/<RUN_ID>/results/<model>/artifacts/<model>_seq_<L>.pt

src/unified_evaluator.py
  -> output/<RUN_ID>/evaluation/seq_<L>/<model>/metrics.json
```

## Transform logic

- Price columns (non-`_volume`) are transformed to log returns.
- Volume columns (`*_volume`) are transformed to log(volume).
- Train/test split is 80/20.
- DL preprocessing uses sliding windows with `window_size=100`, `stride=1`.
- A temporal gap of `window_size - 1` is inserted between train and test to avoid overlap leakage.
- Statistical preprocessing keeps full transformed train/test series (no windowing for fit).

## Dataset paths

Preprocessed dataset paths are defined in `src/utils/preprocessed_data_utils.py`:

- `DL_SET_PATH`
- `STATS_SET_PATH`

## Notes

- Generation/evaluation no longer perform CSV-time preprocessing.
- `src/utils/preprocessing_utils.py` is intentionally minimal and only keeps log-return inversion used by utility metrics.
