#!/usr/bin/env python3
"""Download and extract WeatherBench 1 2m_temperature (5.625 deg) NetCDF data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermassm.data.download import download_weatherbench_netcdf

if __name__ == "__main__":
    out = download_weatherbench_netcdf()
    print(f"Data ready at: {out}")
