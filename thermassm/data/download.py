"""WeatherBench / ERA5 data download and extraction."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def download_weatherbench_netcdf(
    data_dir: str = "data",
    url: str | None = None,
) -> str:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "2m_temperature_5.625deg.zip"
    if not zip_path.exists():
        import subprocess

        url = url or (
            "https://dataserv.ub.tum.de/s/m1524895/download?path="
            "%2F5.625deg%2F2m_temperature"
        )
        subprocess.run(
            ["wget", url, "-O", str(zip_path)],
            check=True,
        )
    extract_dir = data_dir / "weatherbench_data"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not any(extract_dir.rglob("*.nc")):
        import zipfile

        nested_zips = list(extract_dir.rglob("*.zip"))
        if not nested_zips:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            nested_zips = list(extract_dir.rglob("*.zip"))
        for nested in nested_zips:
            with zipfile.ZipFile(nested, "r") as zf:
                zf.extractall(nested.parent)
    return str(extract_dir)


def extract_point_series(
    nc_dir: str,
    lat: float,
    lon: float,
    start_year: int = 1979,
    end_year: int = 2022,
    var: str = "t2m",
) -> tuple[np.ndarray, np.ndarray]:
    import xarray as xr

    files = sorted(Path(nc_dir).rglob("*.nc"))
    if not files:
        found = [str(p) for p in Path(nc_dir).rglob("*")][:50]
        raise OSError(f"No .nc files found under {nc_dir}. Contents: {found}")
    ds = xr.open_mfdataset(files, combine="by_coords")
    lon = float(lon)
    lon_coords = ds.lon.values
    if lon_coords.min() < 0:
        lon = ((lon + 180) % 360) - 180
    else:
        lon = lon % 360
    ds = ds.sel(lat=lat, lon=lon, method="nearest")
    ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    ds = ds[var].resample(time="1D").mean()
    dates = ds.time.values
    values = ds.values.astype(np.float32)
    return dates, values
