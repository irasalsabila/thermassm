from dataclasses import dataclass, field


@dataclass
class PhysicsConfig:
    solar_constant: float = 1361.0
    stefan_boltzmann: float = 5.67e-8
    heat_capacity: float = 5.0e7
    emissivity: float = 0.61
    albedo_land: float = 0.28
    albedo_ice: float = 0.62
    freeze_temp: float = 273.15
    albedo_width: float = 1.0
    init_eps: float = 0.61
    init_heat_capacity: float = 5.0e7


@dataclass
class ModelConfig:
    d_model: int = 64
    d_state: int = 64
    input_dim: int = 6
    delta: float = 0.1
    dt_min: float = 0.001
    dt_max: float = 0.1
    decoder_hidden: int = 128


@dataclass
class DataConfig:
    lat: float = 40.0
    lon: float = -105.0
    input_len: int = 90
    horizons: tuple = (365, 730, 1095)
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
    lambda_ebm: float = 0.1
    lambda_smooth: float = 0.01
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
