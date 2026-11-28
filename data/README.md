# Market data (not in git)

MT5 1-minute exports are ~120MB each. Download:

```bash
python scripts/download_mt5.py
```

Source folder: https://drive.google.com/drive/folders/1gwTD6535FilsKqRUsi7vi6WlT0vgdhui

Then:

```bash
python scripts/build_dataset.py --csv data/mt5_raw/EURUSD_M1_202101131130_202605270000_2020_2026.csv
```

Timestamps in the CSVs are naive broker-server **open** times, not UTC. See `docs/stage1-results.md` Step 0.
