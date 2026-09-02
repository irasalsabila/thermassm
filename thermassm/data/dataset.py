"""PyTorch dataset for direct-block and next-step forecasting."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .insolation import daily_insolation
from .synthetic import generate_synthetic
from .download import extract_point_series


def build_features(
    dates: np.ndarray,
    t2m: np.ndarray,
    lat: float,
    lon: float,
) -> np.ndarray:
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    ins = daily_insolation(lat, doy.astype(float))
    doy_sin = np.sin(2 * np.pi * doy / 365.0)
    doy_cos = np.cos(2 * np.pi * doy / 365.0)
    lat_norm = np.full_like(t2m, lat / 90.0)
    lon_norm = np.full_like(t2m, lon / 180.0)
    return np.stack([t2m, ins, doy_sin, doy_cos, lat_norm, lon_norm], axis=-1).astype(
        np.float32
    )


def load_series(cfg) -> tuple[np.ndarray, np.ndarray]:
    if cfg.data.use_synthetic:
        dates, t2m = generate_synthetic(
            cfg.data.lat, cfg.data.train_years[0], cfg.data.test_years[1]
        )
    else:
        from .download import download_weatherbench_netcdf

        wb_dir = download_weatherbench_netcdf(cfg.data.data_dir)
        dates, t2m = extract_point_series(
            wb_dir, cfg.data.lat, cfg.data.lon,
            cfg.data.train_years[0], cfg.data.test_years[1],
        )
    return dates, t2m


class ClimateDataset(Dataset):
    def __init__(
        self,
        dates: np.ndarray,
        t2m: np.ndarray,
        lat: float,
        lon: float,
        input_len: int,
        mode: str = "next",
        predict_len: int = 1,
    ):
        self.input_len = input_len
        self.mode = mode
        self.predict_len = predict_len
        self.features = build_features(dates, t2m, lat, lon)
        self.t2m = t2m
        if mode == "next":
            self.n = len(t2m) - input_len - 1
        else:
            self.n = len(t2m) - input_len - predict_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.input_len]
        if self.mode == "next":
            y = self.t2m[idx + 1 : idx + 1 + self.input_len]
            return torch.from_numpy(x), torch.from_numpy(y)
        if self.mode == "direct":
            # deterministic future forcing only: [Q, sin(DOY), cos(DOY)]
            forcing = self.features[
                idx + self.input_len : idx + self.input_len + self.predict_len, 1:4
            ]
            y = self.t2m[idx + self.input_len : idx + self.input_len + self.predict_len]
            return torch.from_numpy(x), torch.from_numpy(forcing), torch.from_numpy(y)
        y = self.t2m[idx + self.input_len : idx + self.input_len + self.predict_len]
        return torch.from_numpy(x), torch.from_numpy(y)
