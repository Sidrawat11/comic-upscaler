"""
Configuration module for the ManwhaHDFyer pipeline.

Defines dataclasses for engine, directory, output, and pipeline settings
used as the default configuration throughout the project.
"""

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class EngineConfig:
    scale: int = 2
    tile: int = 512
    half: bool = True
    chunk_size: int = 720
    overlap: int = 128
    ratio_threshold: float = 4.0
    cudnn_benchmark: bool = True
    sharpen_strength: float = 0.3 
    sharpen_radius: float = 1.0

@dataclass
class DirectoryConfig:
    manwha_dir: Path
    model_path: Path = Path('models/4x-UltraSharp.pth')
    result_dir: Path = Path('results')
    log_dir: Path = Path('Logs/')

@dataclass
class OutputConfig:
    format: str = 'jpg'
    quality: int = 92
    skip_final: int = 2
    limit: int = 0

@dataclass
class PipelineConfig:
    dirs: DirectoryConfig
    engine: EngineConfig = field(default_factory=EngineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

