
from dataclasses import dataclass, field

from torch import Tensor, nn, randn
from torchvision.transforms import InterpolationMode


@dataclass()
class Config:
    image_shape: tuple[int, int] = field(init=False)

    # Maximums as obtained with the "stats" command.
    max_width: int = 3993
    max_height: int = 3325

    in_channels: int = 1
    divider: int = 5
    embed_dim: int = 256
    mlp_dim: int = 256

    num_heads: int = 8
    patch_size: int = 16
    dropout: float = 0.1
    num_layers = 4

    batch_size: int = 8
    valid_split: float = 0.8

    interpolation: InterpolationMode = InterpolationMode.BILINEAR
    antialias: bool = True

    def scale_to_patch(self, value: int) -> int:
        ret = value // self.divider
        return int(round(ret / self.patch_size) * self.patch_size)

    def __post_init__(self):
        self.image_shape = (
            self.scale_to_patch(self.max_height),
            self.scale_to_patch(self.max_width),
        )
        assert self.patch_size ** 2 == self.embed_dim


class PatchEmbedding(nn.Module):

    config: Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_patch = (
            config.image_shape[0] // config.patch_size,
            config.image_shape[1] // config.patch_size)
        self.proj = nn.Conv2d(config.in_channels, config.embed_dim,
                              kernel_size=config.patch_size, stride=config.patch_size)
        self.pos_embed = nn.Parameter(0.02 * randn(
            self.num_patch[0] * self.num_patch[1], config.embed_dim))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x += self.pos_embed
        return self.dropout(x)


class TransformerBlock(nn.Module):

    config: Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(
            config.embed_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):

    config: Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.transformer = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.mask_head = nn.Linear(
            config.embed_dim,
            config.patch_size ** 2
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.mask_head(x)

        Hp = self.config.image_shape[0] // self.config.patch_size
        Wp = self.config.image_shape[1] // self.config.patch_size
        B, _, C = x.shape
        x = x.view(
            B, Hp, Wp,
            self.config.patch_size, self.config.patch_size)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(
            B, Hp * self.config.patch_size,
            Wp * self.config.patch_size
        )

        return x
        return x
