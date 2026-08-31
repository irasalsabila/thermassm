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

    lat = float(lat)
    lon = float(lon)
    pieces = []
    for f in files:
        with xr.open_dataset(f) as ds:
            lon_coords = ds.lon.values
            if lon_coords.min() < 0:
                lon_sel = ((lon + 180) % 360) - 180
            else:
                lon_sel = lon % 360
            vname = var if var in ds else ("2m_temperature" if "2m_temperature" in ds else None)
            if vname is None:
                continue
            p = ds[vname].sel(lat=lat, lon=lon_sel, method="nearest")
            p = p.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
            p = p.resample(time="1D").mean()
            pieces.append(p)

    if not pieces:
        raise OSError("No matching data found for the requested variable/time range.")

    series = xr.concat(pieces, dim="time").sortby("time")
    dates = series.time.values
    values = series.values.astype(np.float32)
    return dates, values
