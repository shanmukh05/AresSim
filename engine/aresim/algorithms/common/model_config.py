"""Architecture dimensions shared by symbolic-observation encoders.

Safe to import without Torch or Ray; PPO retains its historical ModelConfig alias.
"""
from __future__ import annotations
from dataclasses import dataclass
from .config_decode import positive_integer

@dataclass(frozen=True)
class ModelConfig:
    """Architecture dimensions for the shared local observation encoder.

    Expects exactly two convolution and two telemetry hidden layers. Embeddings
    size the terrain, objective-type, pad-proximity, and weather branches.
    """

    terrain_embedding: int = 4
    objective_embedding: int = 4
    conv_channels: tuple[int, int] = (32, 64)
    telemetry_layers: tuple[int, int] = (128, 128)
    fused_width: int = 256

    def validate(self) -> None:
        """Raise ``ValueError`` when layer counts or dimensions are invalid."""
        if len(self.conv_channels) != 2 or len(self.telemetry_layers) != 2:
            raise ValueError("the built-in model requires two convolution and telemetry layers")
        values = (self.terrain_embedding, self.objective_embedding, *self.conv_channels, *self.telemetry_layers, self.fused_width)
        for value in values:
            positive_integer(value, "model dimension")

