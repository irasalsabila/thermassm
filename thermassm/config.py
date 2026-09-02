"""Configuration for PhysSSM (90-day input -> 30-day direct block forecasting)."""
from dataclasses import dataclass, field


@dataclass
class AnchorConfig:
    """Effective insolation-forced thermal response (thermal anchor).

    mu[t+1] = rho * mu[t] + (1 - rho) * (a + b * Q[t])
    rho = exp(-1 / tau),  tau > 0
    """

    tau_init: float = 30.0  # effective thermal-response timescale (days)
    a_init: float = 280.0   # effective local equilibrium offset (K)
    b_init: float = 0.05    # forcing sensitivity (K per W/m^2)


@dataclass
class ModelConfig:
    d_model: int = 64
    d_state: int = 64
    input_dim: int = 6          # [T, Q, sin(DOY), cos(DOY), lat_norm, lon_norm]
    decoder_hidden: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    forecast_horizon: int = 30  # direct 30-day block
    s4d_layers: int = 2
    s4d_init: str = "s4d-lin"
    # residual output amplitude bound (fallback; overridden by training quantile)
    res_amp: float = 10.0

    # legacy baseline knobs kept for baselines.py compatibility
    pint_block: int = 30
    patch_len: int = 16
    patch_stride: int = 8
    patch_layers: int = 3
    patch_heads: int = 16
    patch_d_model: int = 128
    patch_ffn: int = 256
    patch_dropout: float = 0.2
    patch_horizon: int = 96
    patch_lookback: int = 336
    s4d_init_mode: str = "s4d"


@dataclass
class DataConfig:
    lat: float = 40.0
    lon: float = -105.0
    input_len: int = 90
    forecast_len: int = 30
    horizons: tuple = (1, 7, 14, 30, 365, 730)
    train_years: tuple = (1979, 2015)
    val_years: tuple = (2016, 2018)
    test_years: tuple = (2019, 2022)
    use_synthetic: bool = False
    data_dir: str = "data"


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    # Soft physics regularizer weight for the PINT baselines only (not PhysSSM).
    lambda_physics: float = 0.001


@dataclass
class Config:
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Seven-site publication set (3 PINT comparison sites + 4 cross-climate sites).
SITES = {
    "Seoul": (37.57, 126.98),
    "Beijing": (39.90, 116.40),
    "Washington_DC": (38.90, -77.04),
    "Padang": (-0.95, 100.35),
    "Denver": (39.74, -104.99),
    "Abu_Dhabi": (24.45, 54.38),
    "Oulu": (65.01, 25.47),
}

PINT_SITES = ["Seoul", "Beijing", "Washington_DC"]
CROSS_CLIMATE_SITES = ["Padang", "Denver", "Abu_Dhabi", "Oulu"]

# Gate 1 sites: Denver plus one PINT comparison site.
GATE1_SITES = ["Denver", "Seoul"]
