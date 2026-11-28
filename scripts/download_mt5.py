#!/usr/bin/env python3
"""Download the MT5 1m Google Drive folder."""

from __future__ import annotations

from pathlib import Path

FOLDER = "https://drive.google.com/drive/folders/1gwTD6535FilsKqRUsi7vi6WlT0vgdhui"


def main() -> None:
    import gdown

    dest = Path("data/mt5_raw")
    dest.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(FOLDER, output=str(dest), quiet=False)


if __name__ == "__main__":
    main()
