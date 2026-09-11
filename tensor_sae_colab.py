# %% [markdown]
# # Tensor-SAE: reproduction notebook
#
# Reimplementation of the experiments in *Tensor-SAE: Structured Sparse Autoencoders for
# Interpretable and Efficient Image Representations* (GRaM workshop at ICLR 2026, published in
# PMLR). The original experiment code was lost, and this notebook was written from the paper's
# abstract and figures as described there, so every hyper-parameter that the abstract does not
# pin down is a choice made here. All of those choices sit in the `Config` dataclass in the
# next cell and are named again in the markdown cell next to the code that uses them.
#
# **The method.** A sparse autoencoder maps an image to a non-negative latent vector `z` and
# reconstructs the image as `x̂ = Σ_k z_k a_k + b`. A Dense-SAE keeps every atom `a_k` as a free
# 3×32×32 tensor. Tensor-SAE constrains each atom to be rank one, `a_k = c_k ⊗ h_k ⊗ w_k` with
# a colour factor `c_k ∈ R³`, a row factor `h_k ∈ R³²` and a column factor `w_k ∈ R³²`, so an
# atom costs 67 numbers instead of 3072. The latents are trained with an L1 penalty (the
# abstract's "light sparsity prior"), and a TopK gate is available as an option.
#
# **Two run modes.** Set `RUN_MODE` in the next cell.
#
# * `"smoke"`: an end-to-end check that takes about three minutes on a two-core CPU. It uses
#   synthetic 32×32 images made of coloured rectangles and blobs (nothing is downloaded) and
#   dictionaries of 32 to 128 atoms. Its numbers test the code, not the paper's claims.
# * `"full"`: CIFAR-10 through torchvision, dictionaries of 1024 to 4096 atoms, 30 epochs per
#   model. Roughly 30 to 60 minutes on a free Colab T4.
#
# **Outputs.** Every trained model, its per-epoch history and every table and figure are written
# to `results/`. Expensive steps are cached there, so re-running the notebook after a Colab
# restart resumes where it stopped.

# %%
"""Setup: run mode, imports, configuration, seeding and caching helpers."""
import dataclasses
import functools
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

RUN_MODE = os.environ.get("TSAE_RUN_MODE", "smoke")  # "smoke" or "full"
SAVE_TO_DRIVE = False  # True: mount Google Drive in Colab and mirror results/ there at the end
assert RUN_MODE in ("smoke", "full"), RUN_MODE

try:
    import torchvision  # noqa: F401
except ImportError:  # pragma: no cover - only on a bare kernel
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchvision"])

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SHAPE = (3, 32, 32)
D_IN = 3 * 32 * 32
print(f"run mode = {RUN_MODE}, device = {DEVICE}, torch {torch.__version__}")


@dataclass
class Config:
    """Every size, count and hyper-parameter of one run mode.

    The paper's abstract does not state dictionary sizes, the sparsity weight, the optimiser
    or the schedule, so the values below are choices made for this reimplementation.
    """

    run_mode: str
    seed: int = 0
    results_dir: str = "results"
    # data
    n_train: int = 50_000            # CIFAR-10 has 50k training images; smoke: synthetic count
    n_test: int = 10_000
    # model sweep: one Tensor-SAE per size, each with a parameter-matched Dense-SAE and ConvAE
    tensor_ks: Tuple[int, ...] = (1024, 2048, 4096)
    topk: Optional[int] = None       # None: ReLU + L1 only; an int adds TopK gating on top
    # training
    epochs: int = 30                 # epochs for the two SAEs
    conv_epochs: int = 30            # epochs for the ConvAE (smoke lowers it: convolutions dominate CPU time)
    batch_size: int = 256
    lr: float = 1e-3
    l1: float = 0.3                  # λ in  L = ||x - x̂||² + λ ||z||₁  (per image), SAEs only
    # evaluation
    n_eval: int = 2000               # test images used for the per-epoch metrics
    n_strength_latents: int = 64     # latents tracked for intervention strength each epoch
    n_strength_images: int = 32
    n_interv_images: int = 64        # images × latents × alphas for the linearity test
    n_interv_latents: int = 16
    alphas: Tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0)
    n_top_atoms: int = 32            # atoms shown in the qualitative grids
    hist_images: int = 1000          # images behind the activation histograms


SMOKE = Config(
    run_mode="smoke", n_train=2000, n_test=1000, tensor_ks=(32, 64, 128), epochs=40, conv_epochs=6,
    batch_size=32, lr=3e-3, n_eval=500, n_strength_latents=16, n_strength_images=16,
    n_interv_images=32, n_interv_latents=8, n_top_atoms=16, hist_images=500,
)
FULL = Config(run_mode="full")
CFG = SMOKE if RUN_MODE == "smoke" else FULL
os.makedirs(CFG.results_dir, exist_ok=True)

# Numbers reported in the paper's abstract, printed next to the reproduced ones. The abstract
# gives one number; everything else is marked as not reported there.
PAPER = {"tensor_r2": 0.93}
NOT_REPORTED = "not reported in abstract"
PALETTE = {"Tensor-SAE": "#2a78d6", "Dense-SAE": "#eb6834", "ConvAE": "#1baf7a"}
MARKERS = {"Tensor-SAE": "o", "Dense-SAE": "s", "ConvAE": "^"}


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def result_path(name: str) -> str:
    """Path of a file inside results/, prefixed with the run mode."""
    return os.path.join(CFG.results_dir, f"{CFG.run_mode}_{name}")


def cached_json(name: str, build: Callable[[], dict]) -> dict:
    """Load a JSON result from results/ if present, otherwise build and save it."""
    path = result_path(name)
    if os.path.exists(path):
        print(f"[cache] loading {path}")
        with open(path) as handle:
            return json.load(handle)
    out = build()
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    return out


def savefig(fig: matplotlib.figure.Figure, name: str) -> None:
    """Save a figure to results/ at 150 dpi and show it inline when a display exists."""
    fig.tight_layout()
    fig.savefig(result_path(name), dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


seed_everything(CFG.seed)
print(json.dumps(dataclasses.asdict(CFG), indent=1))

# %% [markdown]
# ## 1. Data
#
# The paper trains on CIFAR-10 (3×32×32 pixels). `full` mode loads it through torchvision:
# the 50k training images are used for training, and a fixed subset of the 10k test images for
# every metric. Pixels are scaled to [0, 1] and nothing else is done to them; the decoder bias
# is initialised to the mean image so the atoms only have to explain deviations from it.
#
# `smoke` mode cannot download anything, so it uses synthetic images with the same geometry:
# a dim background colour with one to four axis-aligned coloured rectangles and soft blobs on
# top. Rectangles are exactly rank-one in (row, column) with a single colour, so the smoke data
# has the structure Tensor-SAE atoms can represent, and the blobs give the models something
# that a rank-one atom can only approximate.

# %%
"""Synthetic images for smoke mode, CIFAR-10 for full mode."""


SMOKE_HUES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.float32)


def synthetic_images(n: int, seed: int) -> torch.Tensor:
    """Images of coloured rectangles and blobs on a dim background, in [0, 1].

    Each image has one to four parts. A part lives in one cell of a 4×4 grid of 8×8-pixel
    cells, is either a rectangle with a random inset or a Gaussian blob, and takes one of six
    saturated hues at a random brightness. There are 16 × 6 = 96 (cell, hue) combinations, so a
    dictionary of about a hundred rank-one atoms can represent the data.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
    out = np.empty((n, 3, 32, 32), dtype=np.float32)
    for i in range(n):
        img = np.ones((3, 32, 32), dtype=np.float32) * rng.uniform(0.0, 0.25, size=(3, 1, 1))
        for cell in rng.choice(16, size=rng.integers(1, 5), replace=False):
            r0, c0 = 8 * (cell // 4), 8 * (cell % 4)
            colour = SMOKE_HUES[rng.integers(0, 6)] * rng.uniform(0.6, 1.0)
            if rng.random() < 0.7:
                top, left = rng.integers(0, 3, size=2)
                height, width = rng.integers(4, 9 - top), rng.integers(4, 9 - left)
                mask = np.zeros((32, 32), dtype=np.float32)
                mask[r0 + top:r0 + top + height, c0 + left:c0 + left + width] = 1.0
            else:
                sigma = rng.uniform(1.5, 3.0)
                mask = np.exp(-((yy - r0 - 3.5) ** 2 + (xx - c0 - 3.5) ** 2) / (2 * sigma ** 2)).astype(np.float32)
            img = img * (1 - mask) + colour[:, None, None] * mask
        out[i] = img
    return torch.from_numpy(out)


def load_cifar10(train: bool) -> torch.Tensor:
    """CIFAR-10 as a float tensor N×3×32×32 in [0, 1]; downloads to data/ on first use."""
    from torchvision import datasets

    dataset = datasets.CIFAR10(root="data", train=train, download=True)
    return torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() / 255.0


if CFG.run_mode == "smoke":
    TRAIN_X = synthetic_images(CFG.n_train, seed=CFG.seed)
    TEST_X = synthetic_images(CFG.n_test, seed=CFG.seed + 1)
else:
    TRAIN_X = load_cifar10(train=True)[: CFG.n_train]
    TEST_X = load_cifar10(train=False)[: CFG.n_test]
TRAIN_X = TRAIN_X.to(DEVICE)
TEST_X = TEST_X.to(DEVICE)
EVAL_X = TEST_X[: CFG.n_eval]
print(f"train {tuple(TRAIN_X.shape)}, test {tuple(TEST_X.shape)}, eval subset {len(EVAL_X)}")

fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
for ax, img in zip(axes.flat, TRAIN_X[:16].cpu()):
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.axis("off")
fig.suptitle(f"Training images ({'synthetic rectangles and blobs' if CFG.run_mode == 'smoke' else 'CIFAR-10'})")
savefig(fig, "fig0_data.png")

# %% [markdown]
# ## 2. Models
#
# All three models share the same interface: `encode` gives non-negative latents, `decode`
# maps latents back to a 3×32×32 image, `add_to_latent` implements an intervention, and
# `flops_per_sample` counts the multiply-adds of one forward pass.
#
# **Tensor-SAE.** Encoder `z = ReLU(W (x - b) + b_enc)`, with an optional TopK gate. Decoder
# `x̂ = Σ_k z_k c_k ⊗ h_k ⊗ w_k + b`. Each factor is normalised to unit length at every forward
# pass, so every atom has unit Frobenius norm (‖c ⊗ h ⊗ w‖ = ‖c‖‖h‖‖w‖) and the latent
# magnitudes of different atoms are comparable. The decoder never materialises the K×3072
# atom bank: it scales the latents by the colour factors and contracts with the K×1024 bank of
# spatial maps `h_k w_kᵀ`, which is rebuilt from the factors each step. The encoder is a free
# linear map, initialised to the transposed atoms; tying it would save parameters but the
# abstract only describes the decoder as factorised.
#
# **Dense-SAE.** The same encoder and training loss, with a free K'×3072 dictionary whose rows
# are normalised to unit length. K' is chosen so that the total parameter count matches the
# Tensor-SAE it is paired with. Because both models carry a free 3072×K encoder, the match
# gives K' ≈ K/2; the decoder alone is 46 times cheaper for Tensor-SAE, and both ratios are
# printed below.
#
# **ConvAE.** A four-layer convolutional encoder to a ReLU bottleneck of `d` channels at 8×8,
# and a mirrored transposed-convolution decoder. Its width is searched so that the parameter
# count matches the same budget. Interventions add to one channel at the central bottleneck
# position, which is the closest analogue of adding one spatially localised atom.
#
# The cell ends with two checks: the factorised decoder agrees with an explicit `z @ atoms`
# product, and the TopK gate never leaves more than k latents active.

# %%
"""Tensor-SAE, Dense-SAE and ConvAE with a common interface."""


def topk_gate(z: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    """Keep the k largest entries of each row of z and zero the rest (identity when k is None)."""
    if k is None or k >= z.shape[1]:
        return z
    values, indices = z.topk(k, dim=1)
    return torch.zeros_like(z).scatter_(1, indices, values)


def unit_rows(matrix: torch.Tensor) -> torch.Tensor:
    """Normalise every row of a matrix to unit L2 norm."""
    return matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-8)


class SparseAutoencoder(nn.Module):
    """Shared encoder and interface of the two SAEs; subclasses define the decoder."""

    family = "SAE"

    def __init__(self, n_latents: int, topk: Optional[int]) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.topk = topk
        self.encoder = nn.Linear(D_IN, n_latents)
        self.b_dec = nn.Parameter(torch.zeros(D_IN))

    def atoms(self) -> torch.Tensor:
        """The K×3072 bank of unit-norm atoms (materialised for analysis only)."""
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decoder_flops(self) -> int:
        raise NotImplementedError

    def init_from_data(self, x: torch.Tensor) -> None:
        """Decoder bias at the mean image, encoder rows equal to the atoms (a standard SAE init)."""
        with torch.no_grad():
            self.b_dec.copy_(x.flatten(1).mean(0))
            self.encoder.weight.copy_(self.atoms())
            self.encoder.bias.zero_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x.flatten(1) - self.b_dec)
        return topk_gate(F.relu(pre), self.topk)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @staticmethod
    def add_to_latent(z: torch.Tensor, k: int, amount: torch.Tensor) -> torch.Tensor:
        """Return a copy of z with `amount` (shape B or scalar) added to latent k."""
        out = z.clone()
        out[:, k] = out[:, k] + amount
        return out

    @staticmethod
    def latent_matrix(z: torch.Tensor) -> torch.Tensor:
        """Latents as a B×K matrix, one column per intervention target."""
        return z

    def flops_per_sample(self) -> int:
        return D_IN * self.n_latents + self.decoder_flops()


class TensorSAE(SparseAutoencoder):
    """SAE whose atoms are rank-one tensors c_k ⊗ h_k ⊗ w_k over (colour, row, column)."""

    family = "Tensor-SAE"

    def __init__(self, n_latents: int, topk: Optional[int] = None) -> None:
        super().__init__(n_latents, topk)
        self.colour = nn.Parameter(torch.randn(n_latents, 3))
        self.rows = nn.Parameter(torch.randn(n_latents, 32))
        self.cols = nn.Parameter(torch.randn(n_latents, 32))

    def factors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unit-norm colour, row and column factors."""
        return unit_rows(self.colour), unit_rows(self.rows), unit_rows(self.cols)

    def spatial_maps(self) -> torch.Tensor:
        """The K×32×32 bank of rank-one spatial maps h_k w_kᵀ (unit Frobenius norm each)."""
        _, rows, cols = self.factors()
        return rows[:, :, None] * cols[:, None, :]

    def atoms(self) -> torch.Tensor:
        colour, rows, cols = self.factors()
        return torch.einsum("kc,kh,kw->kchw", colour, rows, cols).flatten(1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        colour, rows, cols = self.factors()
        scaled = z[:, :, None] * colour[None]                       # B×K×3: z_k c_k
        spatial = (rows[:, :, None] * cols[:, None, :]).flatten(1)  # K×1024: h_k w_kᵀ
        image = scaled.transpose(1, 2).reshape(-1, self.n_latents) @ spatial  # (B·3)×1024
        return image.view(z.shape[0], *IMG_SHAPE) + self.b_dec.view(IMG_SHAPE)

    def decoder_flops(self) -> int:
        """Multiply-adds of `decode` for one image: colour scaling plus the spatial contraction."""
        return self.n_latents * 3 + 3 * self.n_latents * 32 * 32


class DenseSAE(SparseAutoencoder):
    """Ordinary SAE with a free dictionary of K'×3072 unit-norm atoms."""

    family = "Dense-SAE"

    def __init__(self, n_latents: int, topk: Optional[int] = None) -> None:
        super().__init__(n_latents, topk)
        self.dictionary = nn.Parameter(torch.randn(n_latents, D_IN))

    def atoms(self) -> torch.Tensor:
        return unit_rows(self.dictionary)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.atoms() + self.b_dec).view(z.shape[0], *IMG_SHAPE)

    def decoder_flops(self) -> int:
        return self.n_latents * D_IN


class ConvAE(nn.Module):
    """Plain convolutional autoencoder with a ReLU bottleneck of `width` channels at 8×8.

    Both stride-2 layers come first so that most of the arithmetic happens at 8×8; a
    parameter-matched ConvAE still costs far more multiply-adds per image than the SAEs.
    """

    family = "ConvAE"

    def __init__(self, width: int) -> None:
        super().__init__()
        w = width
        self.n_latents = w  # intervention targets are bottleneck channels
        self.enc_net = nn.Sequential(
            nn.Conv2d(3, w, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(w, 2 * w, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.ReLU(),
            nn.Conv2d(4 * w, w, 1), nn.ReLU(),
        )
        self.dec_net = nn.Sequential(
            nn.Conv2d(w, 4 * w, 1), nn.ReLU(),
            nn.Conv2d(4 * w, 2 * w, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(2 * w, w, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(w, 3, 4, stride=2, padding=1),
        )

    def init_from_data(self, x: torch.Tensor) -> None:
        """Bias of the last layer at the mean pixel value per channel."""
        with torch.no_grad():
            self.dec_net[-1].bias.copy_(x.mean(dim=(0, 2, 3)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_net(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_net(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @staticmethod
    def add_to_latent(z: torch.Tensor, k: int, amount: torch.Tensor) -> torch.Tensor:
        """Add `amount` to channel k at the central bottleneck position (4, 4)."""
        out = z.clone()
        out[:, k, 4, 4] = out[:, k, 4, 4] + amount
        return out

    @staticmethod
    def latent_matrix(z: torch.Tensor) -> torch.Tensor:
        """Channel activations at the central position, B×d, as the intervention targets."""
        return z[:, :, 4, 4]

    def flops_per_sample(self) -> int:
        """Multiply-adds of one forward pass, counted from the conv layer shapes with hooks."""
        counts: List[int] = []

        def hook(module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            taps = module.kernel_size[0] * module.kernel_size[1]
            if isinstance(module, nn.ConvTranspose2d):  # every input element is spread over out_channels × taps
                counts.append(inputs[0][0].numel() * module.out_channels * taps)
            else:  # every output element gathers in_channels × taps
                counts.append(output[0].numel() * module.in_channels * taps)

        handles = [m.register_forward_hook(hook) for m in self.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
        with torch.no_grad():
            self(torch.zeros(1, *IMG_SHAPE, device=next(self.parameters()).device))
        for handle in handles:
            handle.remove()
        return int(sum(counts))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def matched_dense_latents(tensor_params: int) -> int:
    """K' such that a Dense-SAE has (as nearly as possible) `tensor_params` parameters."""
    per_latent = D_IN + 1 + D_IN  # encoder row + encoder bias + dictionary row
    return max(1, round((tensor_params - D_IN) / per_latent))


@functools.lru_cache(maxsize=None)
def matched_conv_width(target_params: int) -> int:
    """ConvAE width whose parameter count is closest to `target_params`."""
    best_width, best_gap = 4, float("inf")
    for width in range(4, 2048, 2):
        params = count_params(ConvAE(width))
        if abs(params - target_params) < best_gap:
            best_width, best_gap = width, abs(params - target_params)
        if params > target_params:
            break
    return best_width


def build_triple(k_tensor: int) -> Dict[str, nn.Module]:
    """A Tensor-SAE with K atoms and its parameter-matched Dense-SAE and ConvAE."""
    tensor = TensorSAE(k_tensor, CFG.topk)
    budget = count_params(tensor)
    dense = DenseSAE(matched_dense_latents(budget), CFG.topk)
    conv = ConvAE(matched_conv_width(budget))
    models = {"Tensor-SAE": tensor, "Dense-SAE": dense, "ConvAE": conv}
    for model in models.values():
        model.to(DEVICE)
        model.init_from_data(TRAIN_X[:5000])
    return models


def model_name(family: str, k_tensor: int) -> str:
    return f"{family}_K{k_tensor}"


SIZE_ROWS = []
for _k in CFG.tensor_ks:
    _models = build_triple(_k)
    for _family, _model in _models.items():
        SIZE_ROWS.append({
            "family": _family, "K_tensor": _k, "n_latents": _model.n_latents,
            "params": count_params(_model), "flops_per_sample": _model.flops_per_sample(),
            "decoder_params": (_model.n_latents * 67 + D_IN if _family == "Tensor-SAE"
                               else _model.n_latents * D_IN + D_IN if _family == "Dense-SAE"
                               else count_params(_model.dec_net)),
        })
SIZES = pd.DataFrame(SIZE_ROWS)
SIZES.to_csv(result_path("model_sizes.csv"), index=False)
print("Model sizes (one row per model; Dense-SAE and ConvAE are matched to the Tensor-SAE budget)")
print(SIZES.to_string(index=False))

_check = TensorSAE(8, topk=3).to(DEVICE)
_z = _check.encode(TRAIN_X[:4])
assert (_z > 0).sum(1).max() <= 3, "TopK gate left more than k latents active"
_explicit = (_z @ _check.atoms() + _check.b_dec).view(-1, *IMG_SHAPE)
assert torch.allclose(_check.decode(_z), _explicit, atol=1e-5), "factorised decoder disagrees with z @ atoms"
assert torch.allclose(_check.atoms().norm(dim=1), torch.ones(8, device=DEVICE), atol=1e-5)
print("checks passed: TopK gate, factorised decoder = z @ atoms, unit-norm atoms")

# %% [markdown]
# ## 3. Training
#
# The two SAEs minimise the per-image loss `‖x − x̂‖² + λ‖z‖₁` with Adam; the L1 term is the
# abstract's light sparsity prior. The ConvAE is a plain autoencoder baseline and minimises
# the squared error alone. The abstract does not give λ, the optimiser, the learning rate or
# the number of epochs; the values used are `Config.l1`, Adam at `Config.lr` and
# `Config.epochs`, with no schedule and no dead-latent resampling. In `full` mode every model
# trains for the same number of epochs; in `smoke` mode the ConvAE gets `Config.conv_epochs`
# (fewer) because convolutions dominate CPU time.
#
# After every epoch the following are measured on the fixed evaluation subset:
#
# * reconstruction MSE per pixel and PSNR (pixels in [0, 1], reconstructions clamped to [0, 1]),
# * L0, the mean number of non-zero latents per image (for the ConvAE: non-zero bottleneck
#   units, out of `d·64`),
# * the dead-latent fraction (latents that never fire on the evaluation subset),
# * the intervention strength of a fixed random set of latents, defined in Section 7: the norm
#   of the clamped decoded change for a unit latent change, and for a change of one typical
#   activation of that latent.
#
# Models and histories are cached in `results/`, so this cell is skipped on a rerun.

# %%
"""Training loop with per-epoch metrics; every model in the sweep is trained here."""


@torch.no_grad()
def batched(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, batch: int = 512) -> torch.Tensor:
    """Apply fn to x in batches and concatenate the results."""
    return torch.cat([fn(x[i:i + batch]) for i in range(0, len(x), batch)])


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def typical_activation(model: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Mean activation of each latent over the images where it fires (global mean for dead ones)."""
    zm = z if z.dim() == 2 else z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])  # ConvAE: every position is a sample
    active = zm > 0
    counts = active.sum(0)
    means = (zm * active).sum(0) / counts.clamp_min(1)
    fallback = zm[active].mean() if active.any() else torch.tensor(1.0, device=z.device)
    return torch.where(counts > 0, means, fallback)


@torch.no_grad()
def intervention_strength(model: nn.Module, x: torch.Tensor, latents: Sequence[int],
                          scale: torch.Tensor) -> Tuple[float, float]:
    """Mean norm of the clamped decoded change per unit latent change, and per typical activation."""
    z = model.encode(x)
    base = clamp01(model.decode(z))
    unit, typical = [], []
    for k in latents:
        unit.append((clamp01(model.decode(model.add_to_latent(z, k, torch.tensor(1.0, device=x.device)))) - base).flatten(1).norm(dim=1).mean())
        typical.append((clamp01(model.decode(model.add_to_latent(z, k, scale[k]))) - base).flatten(1).norm(dim=1).mean())
    return float(torch.stack(unit).mean()), float(torch.stack(typical).mean())


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, batch: int = 512) -> Tuple[Dict[str, float], torch.Tensor]:
    """Reconstruction and sparsity metrics on x, plus the per-image L0 (computed in batches)."""
    model.eval()
    sq_err, per_image, fired = 0.0, [], None
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        z = model.encode(xb)
        sq_err += float(((clamp01(model.decode(z)) - xb) ** 2).sum())
        active = z > 0
        per_image.append(active.flatten(1).sum(1).float())
        fired_here = active.any(0) if z.dim() == 2 else active.permute(1, 0, 2, 3).flatten(1).any(1)
        fired = fired_here if fired is None else fired | fired_here
        n_units = active[0].numel()
    l0 = torch.cat(per_image)
    mse = sq_err / x.numel()
    metrics = {
        "mse": mse, "psnr": 10 * math.log10(1.0 / max(mse, 1e-12)),
        "l0": float(l0.mean()), "l0_frac": float(l0.mean() / n_units),
        "dead_frac": float(1 - fired.float().mean()),
    }
    return metrics, l0.cpu()


def train_model(name: str, model: nn.Module) -> pd.DataFrame:
    """Train one model (cached), returning its per-epoch history."""
    ckpt, hist = result_path(f"{name}.pt"), result_path(f"{name}_history.csv")
    if os.path.exists(ckpt) and os.path.exists(hist):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        print(f"[cache] loaded {ckpt}")
        return pd.read_csv(hist)
    rng = np.random.default_rng(CFG.seed)
    tracked = sorted(rng.choice(model.n_latents, size=min(CFG.n_strength_latents, model.n_latents), replace=False).tolist())
    strength_x = EVAL_X[: CFG.n_strength_images]
    optimiser = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    generator = torch.Generator(device="cpu").manual_seed(CFG.seed)
    rows = []
    t0 = time.time()
    n_epochs = CFG.conv_epochs if isinstance(model, ConvAE) else CFG.epochs
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(TRAIN_X), generator=generator)
        total, n_batches = 0.0, 0
        for start in range(0, len(TRAIN_X), CFG.batch_size):
            x = TRAIN_X[order[start:start + CFG.batch_size].to(DEVICE)]
            recon, z = model(x)
            loss = ((recon - x) ** 2).flatten(1).sum(1).mean()
            if isinstance(model, SparseAutoencoder):
                loss = loss + CFG.l1 * z.sum(1).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.item()
            n_batches += 1
        metrics, _ = evaluate(model, EVAL_X)
        unit, typical = intervention_strength(model, strength_x, tracked, typical_activation(model, batched(model.encode, EVAL_X)))
        rows.append({"epoch": epoch, "train_loss": total / n_batches, **metrics,
                     "strength_unit": unit, "strength_typical": typical, "seconds": time.time() - t0})
        print(f"  {name} epoch {epoch:2d}: loss {rows[-1]['train_loss']:.2f} mse {metrics['mse']:.5f} "
              f"psnr {metrics['psnr']:.2f} L0 {metrics['l0']:.1f} dead {metrics['dead_frac']:.2f} "
              f"strength {unit:.3f}/{typical:.3f} ({rows[-1]['seconds']:.0f} s)")
    history = pd.DataFrame(rows)
    history.to_csv(hist, index=False)
    torch.save(model.state_dict(), ckpt)
    model.eval()
    return history


MODELS: Dict[str, nn.Module] = {}
HISTORIES: Dict[str, pd.DataFrame] = {}
for _k in CFG.tensor_ks:
    for _family, _model in build_triple(_k).items():
        _name = model_name(_family, _k)
        print(f"training {_name}: {_model.n_latents} latents, {count_params(_model):,} parameters")
        HISTORIES[_name] = train_model(_name, _model)
        MODELS[_name] = _model
MAIN_K = max(CFG.tensor_ks)
MAIN = {family: MODELS[model_name(family, MAIN_K)] for family in PALETTE}
print(f"main models for the detailed analysis: K = {MAIN_K}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for _name, _history in HISTORIES.items():
    _family = _name.split("_K")[0]
    _alpha = 0.35 + 0.65 * (CFG.tensor_ks.index(int(_name.split("_K")[1])) + 1) / len(CFG.tensor_ks)
    for ax, col in zip(axes, ("psnr", "l0", "dead_frac")):
        ax.plot(_history["epoch"], _history[col], color=PALETTE[_family], alpha=_alpha, marker=MARKERS[_family], ms=3, label=_name)
for ax, title in zip(axes, ("PSNR on the evaluation subset (dB)", "L0 (active latents per image)", "dead-latent fraction")):
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.25)
axes[1].set_yscale("log")
axes[0].legend(fontsize=6)
savefig(fig, "fig1_training_curves.png")

# %% [markdown]
# ## 4. Efficiency per parameter and per FLOP
#
# The abstract says Tensor-SAE "is more efficient per FLOP and per parameter". This section
# plots test PSNR against parameter count and against forward multiply-adds for every trained
# model. The parameter count is exact. The FLOP count is the analytic multiply-add count of the
# forward pass as implemented: `3072·K` for the linear encoder, `K·3 + 3·K·1024` for the
# factorised decoder, `K'·3072` for the dense decoder, and the usual conv arithmetic for the
# ConvAE. Two things are worth knowing when reading the FLOP plot. A rank-one atom still has
# to be written to 3072 output pixels, so per active atom the factorised decoder costs the same
# as a dense one; the saving is in parameters, not in decoder arithmetic. And a sparse decoder
# only needs the active atoms, so the table also lists the cost with `L0` active atoms, which is
# what an implementation that gathers active atoms would pay. The abstract does not say which
# accounting the paper used.

# %%
"""Efficiency table and the two scatter plots."""


def efficiency_rows() -> List[dict]:
    rows = []
    for name, model in MODELS.items():
        family, k_tensor = name.split("_K")[0], int(name.split("_K")[1])
        metrics, _ = evaluate(model, TEST_X)
        params, flops = count_params(model), model.flops_per_sample()
        if isinstance(model, SparseAutoencoder):
            sparse_flops = D_IN * model.n_latents + metrics["l0"] * (model.decoder_flops() / model.n_latents)
        else:
            sparse_flops = float(flops)
        rows.append({"model": name, "family": family, "K_tensor": k_tensor, "n_latents": model.n_latents,
                     "params": params, "flops": flops, "flops_sparse_decode": sparse_flops, **metrics})
    return rows


EFFICIENCY = pd.DataFrame(cached_json("efficiency.json", lambda: {"rows": efficiency_rows()})["rows"])
EFFICIENCY.to_csv(result_path("efficiency.csv"), index=False)
print("Test-set reconstruction quality against model cost (paper: curves only, no numbers in the abstract)")
print(EFFICIENCY[["model", "n_latents", "params", "flops", "flops_sparse_decode", "mse", "psnr", "l0", "dead_frac"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for family in PALETTE:
    sub = EFFICIENCY[EFFICIENCY["family"] == family].sort_values("K_tensor")
    for ax, col in zip(axes, ("params", "flops")):
        ax.plot(sub[col], sub["psnr"], color=PALETTE[family], marker=MARKERS[family], label=family)
        for _, row in sub.iterrows():
            ax.annotate(f"K={row['n_latents']}", (row[col], row["psnr"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
axes[0].set_xlabel("parameters")
axes[1].set_xlabel("forward multiply-adds per image (encoder + decoder)")
for ax in axes:
    ax.set_xscale("log")
    ax.set_ylabel("test PSNR (dB)")
    ax.grid(alpha=0.25)
    ax.legend()
fig.suptitle("Reconstruction quality per parameter and per FLOP")
savefig(fig, "fig2_efficiency.png")

# %% [markdown]
# ## 5. Spatial-atom entropy
#
# The abstract's first qualitative claim is that Tensor-SAE "learns low-entropy spatial atoms".
# For every atom, the spatial energy map is `E_k(h, w) = Σ_c |a_k[c, h, w]|`, normalised to sum
# to one, and its Shannon entropy in bits is the measure of how spread out the atom is: a single
# pixel has entropy 0, a uniform 32×32 map has 10 bits. For a Tensor-SAE atom the map is
# `|h_k| ⊗ |w_k|` (the colour factor scales it uniformly), so this is exactly the entropy of the
# separable spatial factor. Dense-SAE atoms are treated with the same formula on their free
# 3×32×32 weights. `2^H` is the equivalent number of uniformly-weighted pixels. The ConvAE has
# no atom bank, so it does not appear here. Only atoms that fire at least once on the test set
# are counted; dead atoms keep their random initialisation.

# %%
"""Spatial entropy of the atoms of the two main SAEs."""


def spatial_energy(atoms: torch.Tensor) -> torch.Tensor:
    """K×1024 normalised spatial energy maps, Σ_c |a_k[c, h, w]| scaled to sum to one."""
    energy = atoms.view(-1, *IMG_SHAPE).abs().sum(1).flatten(1)
    return energy / energy.sum(1, keepdim=True).clamp_min(1e-12)


def entropy_bits(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of each row of a stochastic matrix, in bits."""
    return -(p * torch.log2(p.clamp_min(1e-12))).sum(1)


@torch.no_grad()
def alive_mask(model: SparseAutoencoder, x: torch.Tensor) -> torch.Tensor:
    return (batched(model.encode, x) > 0).any(0)


ENTROPY: Dict[str, np.ndarray] = {}
ENTROPY_ROWS = []
for _family in ("Tensor-SAE", "Dense-SAE"):
    _model = MAIN[_family]
    with torch.no_grad():
        _alive = alive_mask(_model, TEST_X)
        _h = entropy_bits(spatial_energy(_model.atoms()))[_alive].cpu().numpy()
    ENTROPY[_family] = _h
    ENTROPY_ROWS.append({"model": _family, "atoms_alive": int(_alive.sum()), "atoms_total": _model.n_latents,
                         "entropy_mean_bits": float(_h.mean()), "entropy_median_bits": float(np.median(_h)),
                         "equivalent_pixels_median": float(2 ** np.median(_h)),
                         "frac_below_6_bits": float((_h < 6).mean()), "paper": NOT_REPORTED})
ENTROPY_TABLE = pd.DataFrame(ENTROPY_ROWS)
ENTROPY_TABLE.to_csv(result_path("spatial_entropy.csv"), index=False)
print("Spatial-atom entropy (uniform 32×32 map = 10 bits); paper: 'low-entropy spatial atoms', no numbers in the abstract")
print(ENTROPY_TABLE.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

fig, ax = plt.subplots(figsize=(6.5, 4))
_bins = np.linspace(0, 10, 41)
for _family, _h in ENTROPY.items():
    ax.hist(_h, bins=_bins, color=PALETTE[_family], alpha=0.6, label=f"{_family} (median {np.median(_h):.2f} bits)")
ax.axvline(10, color="k", ls="--", lw=1, label="uniform map (10 bits)")
ax.set_xlabel("spatial entropy of the atom (bits)")
ax.set_ylabel("atoms")
ax.legend()
ax.grid(alpha=0.25)
savefig(fig, "fig3_spatial_entropy.png")

# %% [markdown]
# ## 6. Colour factors and the atom gallery
#
# The second claim is "clean colour factors". Each Tensor-SAE atom has an explicit colour
# vector `c_k`; for a Dense-SAE atom the closest thing is the leading left singular vector of
# the atom reshaped to 3×1024, together with the fraction of the atom's energy that this
# rank-one colour⊗space approximation captures (1 by construction for Tensor-SAE). Cleanliness
# is measured as the angle between the colour vector and the nearest of the four canonical
# axes red, green, blue and grey (sign-invariant, since flipping `c_k` and `h_k` together leaves
# the atom unchanged); the table reports the mean angle and the fraction of atoms within 15°.
# The scatter plot shows every colour vector in the chromaticity plane orthogonal to grey, and
# the galleries show the most frequently used atoms of both SAEs.

# %%
"""Colour-factor cleanliness, the chromaticity plot and the atom galleries."""

CANONICAL_AXES = F.normalize(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.float32), dim=1)


@torch.no_grad()
def colour_vectors(model: SparseAutoencoder) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit colour vector per atom and the energy fraction captured by the colour⊗space rank-one fit.

    The sign is fixed so that the colour vector is the colour the atom adds where its spatial
    map is positive on balance (the atom itself is unchanged by flipping both factors).
    """
    if isinstance(model, TensorSAE):
        colour, rows, cols = model.factors()
        sign = torch.sign(rows.sum(1) * cols.sum(1))
        return colour * torch.where(sign == 0, 1.0, sign)[:, None], torch.ones(model.n_latents, device=colour.device)
    matrices = model.atoms().view(-1, 3, 32 * 32)
    u, s, vh = torch.linalg.svd(matrices, full_matrices=False)
    sign = torch.sign(vh[:, 0, :].sum(1))
    return u[:, :, 0] * torch.where(sign == 0, 1.0, sign)[:, None], s[:, 0] ** 2 / (s ** 2).sum(1)


def axis_angles(colour: torch.Tensor) -> torch.Tensor:
    """Angle in degrees from each colour vector to the nearest canonical axis."""
    cosines = (colour @ CANONICAL_AXES.to(colour.device).T).abs().clamp(max=1.0)
    return torch.rad2deg(torch.acos(cosines.max(1).values))


@torch.no_grad()
def activation_frequency(model: SparseAutoencoder, x: torch.Tensor) -> torch.Tensor:
    return (batched(model.encode, x) > 0).float().mean(0)


COLOUR_ROWS, COLOUR_DATA = [], {}
for _family in ("Tensor-SAE", "Dense-SAE"):
    _model = MAIN[_family]
    with torch.no_grad():
        _alive = alive_mask(_model, TEST_X)
        _colour, _sep = colour_vectors(_model)
        _angles = axis_angles(_colour)[_alive]
    COLOUR_DATA[_family] = (_colour[_alive].cpu(), _angles.cpu())
    COLOUR_ROWS.append({"model": _family, "atoms_alive": int(_alive.sum()),
                        "angle_to_nearest_axis_mean_deg": float(_angles.mean()),
                        "frac_within_15_deg": float((_angles < 15).float().mean()),
                        "colour_space_separability": float(_sep[_alive].mean()), "paper": NOT_REPORTED})
COLOUR_TABLE = pd.DataFrame(COLOUR_ROWS)
COLOUR_TABLE.to_csv(result_path("colour_factors.csv"), index=False)
print("Colour-factor cleanliness (angle to the nearest of R, G, B, grey); paper: 'clean colour factors', no numbers in the abstract")
print(COLOUR_TABLE.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def chromaticity(colour: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D coordinates in the plane orthogonal to grey, and a display colour for each vector."""
    c = colour.numpy()
    u = (c[:, 0] - c[:, 1]) / math.sqrt(2)
    v = (c[:, 0] + c[:, 1] - 2 * c[:, 2]) / math.sqrt(6)
    display = np.clip(0.5 + 0.5 * c / np.abs(c).max(1, keepdims=True), 0, 1)
    return u, v, display


fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, (_family, (_colour, _angles)) in zip(axes, COLOUR_DATA.items()):
    _u, _v, _display = chromaticity(_colour)
    ax.scatter(_u, _v, c=_display, s=18, edgecolors="k", linewidths=0.3)
    for _axis, _label in zip(CANONICAL_AXES[:3], ("R", "G", "B")):
        _au, _av, _ = chromaticity(_axis[None])
        ax.annotate(_label, (_au[0], _av[0]), fontsize=10, ha="center", va="center", weight="bold")
    ax.set_title(f"{_family}: colour vectors of {len(_colour)} live atoms\n(mean angle to nearest axis {float(_angles.mean()):.1f}°)")
    ax.set_xlabel("(R − G) / √2")
    ax.set_ylabel("(R + G − 2B) / √6")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
savefig(fig, "fig4_colour_factors.png")


def atom_image(atom: torch.Tensor) -> np.ndarray:
    """Render a 3072-vector atom as an RGB image with grey at zero."""
    img = atom.view(*IMG_SHAPE).permute(1, 2, 0).cpu().numpy()
    return np.clip(0.5 + 0.5 * img / (np.abs(img).max() + 1e-12), 0, 1)


for _family in ("Tensor-SAE", "Dense-SAE"):
    _model = MAIN[_family]
    with torch.no_grad():
        _top = activation_frequency(_model, TEST_X).argsort(descending=True)[: CFG.n_top_atoms].cpu()
        _atoms = _model.atoms()[_top.to(DEVICE)].cpu()
        _energy = spatial_energy(_atoms).view(-1, 32, 32)
        _colour, _ = colour_vectors(_model)
        _colour = _colour[_top.to(DEVICE)].cpu()
    _cols = 8
    _n_rows = math.ceil(len(_top) / _cols)
    fig, axes = plt.subplots(2 * _n_rows, _cols, figsize=(1.3 * _cols, 2.7 * _n_rows))
    for i in range(_n_rows * _cols):
        r, c = divmod(i, _cols)
        for ax in (axes[2 * r, c], axes[2 * r + 1, c]):
            ax.axis("off")
        if i >= len(_top):
            continue
        axes[2 * r, c].imshow(atom_image(_atoms[i]))
        axes[2 * r, c].set_title(f"atom {int(_top[i])}", fontsize=7)
        axes[2 * r + 1, c].imshow(_energy[i].numpy(), cmap="magma")
        _rgb = np.clip(0.5 + 0.5 * _colour[i].numpy() / (_colour[i].abs().max().item() + 1e-12), 0, 1)
        axes[2 * r + 1, c].add_patch(plt.Rectangle((-0.5, -0.5), 5, 5, color=_rgb))
    fig.suptitle(f"{_family}: the {len(_top)} most frequently active atoms (top: atom, bottom: spatial energy map with colour swatch)", fontsize=9)
    savefig(fig, f"fig5_atoms_{_family.lower().replace('-', '_')}.png")

# %% [markdown]
# ## 7. Intervention linearity and intervention strength
#
# The abstract's central quantitative claim is that Tensor-SAE "yields linearly predictable
# intervention effects (R² ≈ 0.93)". An intervention adds `α·s_k` to latent `k` of an encoded
# image, where `s_k` is the typical activation of that latent (its mean over the images where it
# fires) so that `α` is measured in natural units and the same α grid serves every model. The
# predicted pixel change is `α` times the unit response `D(z + s_k e_k) − D(z)`, which for the
# two SAEs is exactly `α s_k a_k`; the actual change is measured on the decoded images after
# clamping to the pixel range. Actual is regressed on predicted over every pixel of every
# (image, latent, α) triple, and R² and the slope are reported.
#
# The abstract does not say whether the paper measured the change directly at the decoder
# output or after re-encoding the edited image, so both are computed. The **direct** protocol
# only sees the decoder (for the SAEs the pixel clamp is the only non-linearity, for the ConvAE
# the whole decoder is non-linear). The **round-trip** protocol re-encodes the edited image and
# decodes it again, so an edit that activates other latents when read back, or that the
# encoder cannot represent, shows up as a departure from linearity. That is the property a
# controllable edit needs, and it is where a structured dictionary can differ from a dense one.
#
# Intervention strength, tracked during training in Section 3, is the norm of the clamped
# decoded change for a unit change of one latent (for unit-norm atoms this is at most 1 and
# equals 1 unless clamping bites) and for a change of one typical activation `s_k` (which also
# tracks the scale the model gives its latents). The coefficient of variation over epochs is the
# stability number.

# %%
"""Intervention linearity (direct and round-trip) and the strength-stability table."""


class LinearFit:
    """Streaming least-squares of actual on predicted over every pixel.

    A subsample of the pixels inside the predicted edit's support is kept for the scatter plot,
    since most pixels of a localised edit are untouched and would pile up at the origin.
    """

    def __init__(self, keep: int = 4000, seed: int = 0) -> None:
        self.sums = torch.zeros(6, dtype=torch.float64)
        self.keep, self.rng = keep, np.random.default_rng(seed)
        self.sample_pred: List[np.ndarray] = []
        self.sample_act: List[np.ndarray] = []

    def update(self, predicted: torch.Tensor, actual: torch.Tensor) -> None:
        p, a = predicted.flatten().double(), actual.flatten().double()
        self.sums += torch.tensor([float(p.sum()), float(a.sum()), float((p * p).sum()), float((a * a).sum()),
                                   float((p * a).sum()), float(len(p))], dtype=torch.float64)
        support = torch.nonzero(p.abs() > 1e-3).flatten().cpu().numpy()  # pixels the edit is meant to touch
        if len(support):
            idx = self.rng.choice(support, size=min(200, len(support)), replace=False)
            self.sample_pred.append(p[idx].cpu().numpy())
            self.sample_act.append(a[idx].cpu().numpy())

    def result(self) -> Dict[str, float]:
        sp, sa, spp, saa, spa, n = self.sums.tolist()
        cov = spa / n - (sp / n) * (sa / n)
        var_p, var_a = spp / n - (sp / n) ** 2, saa / n - (sa / n) ** 2
        slope = cov / max(var_p, 1e-30)
        return {"r2": cov ** 2 / max(var_p * var_a, 1e-30), "slope": slope, "intercept": sa / n - slope * sp / n, "n_pixels": n}

    def samples(self) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.rng.choice(sum(map(len, self.sample_pred)), size=min(self.keep, sum(map(len, self.sample_pred))), replace=False)
        return np.concatenate(self.sample_pred)[idx], np.concatenate(self.sample_act)[idx]


@torch.no_grad()
def most_active_latents(model: nn.Module, x: torch.Tensor, n: int) -> List[int]:
    """Indices of the n latents that fire on the largest number of images."""
    z = model.latent_matrix(batched(model.encode, x))
    return (z > 0).float().mean(0).argsort(descending=True)[:n].tolist()


@torch.no_grad()
def linearity_test(model: nn.Module, x: torch.Tensor, latents: Sequence[int], alphas: Sequence[float]) -> Dict[str, LinearFit]:
    """Regress actual decoded change on the linear prediction, directly and after a round trip."""
    z = model.encode(x)
    scale = typical_activation(model, batched(model.encode, EVAL_X))
    base = clamp01(model.decode(z))
    base_rt = clamp01(model.decode(model.encode(base)))
    fits = {"direct": LinearFit(seed=CFG.seed), "round_trip": LinearFit(seed=CFG.seed + 1)}
    for k in latents:
        unit = model.decode(model.add_to_latent(z, k, scale[k])) - model.decode(z)
        for alpha in alphas:
            edited = clamp01(model.decode(model.add_to_latent(z, k, alpha * scale[k])))
            fits["direct"].update(alpha * unit, edited - base)
            fits["round_trip"].update(alpha * unit, clamp01(model.decode(model.encode(edited))) - base_rt)
    return fits


def linearity_rows() -> List[dict]:
    rows = []
    for family, model in MAIN.items():
        latents = most_active_latents(model, EVAL_X, CFG.n_interv_latents)
        fits = linearity_test(model, TEST_X[: CFG.n_interv_images], latents, CFG.alphas)
        row = {"model": family, "n_latents_tested": len(latents)}
        for protocol, fit in fits.items():
            row.update({f"{protocol}_{key}": value for key, value in fit.result().items()})
            pred, act = fit.samples()
            np.savez_compressed(result_path(f"linearity_samples_{family}_{protocol}.npz"), predicted=pred, actual=act)
        rows.append(row)
    return rows


LINEARITY = pd.DataFrame(cached_json("linearity.json", lambda: {"rows": linearity_rows()})["rows"])
LINEARITY["paper_r2"] = [f"{PAPER['tensor_r2']:.2f}" if m == "Tensor-SAE" else NOT_REPORTED for m in LINEARITY["model"]]
LINEARITY.to_csv(result_path("linearity.csv"), index=False)
print("Intervention linearity: R² of actual pixel change against α × unit response (paper: Tensor-SAE R² ≈ 0.93)")
print(LINEARITY[["model", "direct_r2", "direct_slope", "round_trip_r2", "round_trip_slope", "paper_r2"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
for j, _family in enumerate(MAIN):
    for i, _protocol in enumerate(("direct", "round_trip")):
        _data = np.load(result_path(f"linearity_samples_{_family}_{_protocol}.npz"))
        _row = LINEARITY[LINEARITY["model"] == _family].iloc[0]
        ax = axes[i, j]
        ax.scatter(_data["predicted"], _data["actual"], s=3, alpha=0.35, color=PALETTE[_family])
        _lim = np.abs(_data["predicted"]).max() * 1.05 + 1e-6
        ax.plot([-_lim, _lim], [-_lim, _lim], "k--", lw=1, label="y = x")
        ax.set_xlim(-_lim, _lim)
        ax.set_xlabel("predicted pixel change (α × unit response)")
        ax.set_ylabel("actual pixel change")
        ax.text(0.03, 0.95, "pixels inside the predicted edit's support", transform=ax.transAxes, fontsize=7, va="top")
        ax.set_title(f"{_family}, {_protocol.replace('_', ' ')}: R² = {_row[f'{_protocol}_r2']:.3f}, slope = {_row[f'{_protocol}_slope']:.2f}", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
fig.suptitle(f"Intervention linearity on {CFG.n_interv_images} images × {CFG.n_interv_latents} latents × α ∈ [{min(CFG.alphas)}, {max(CFG.alphas)}] (paper: Tensor-SAE R² ≈ {PAPER['tensor_r2']})")
savefig(fig, "fig6_intervention_linearity.png")

STABILITY_ROWS = []
for _family, _model in MAIN.items():
    _history = HISTORIES[model_name(_family, MAIN_K)]
    for col in ("strength_unit", "strength_typical"):
        _values = _history[col].values
        STABILITY_ROWS.append({"model": _family, "measure": col, "mean": float(_values.mean()), "std": float(_values.std()),
                               "coefficient_of_variation": float(_values.std() / max(_values.mean(), 1e-12)),
                               "first_epoch": float(_values[0]), "last_epoch": float(_values[-1]), "paper": NOT_REPORTED})
STABILITY = pd.DataFrame(STABILITY_ROWS)
STABILITY.to_csv(result_path("strength_stability.csv"), index=False)
print("\nIntervention strength during training (paper: 'keeps intervention strength stable', no numbers in the abstract)")
print(STABILITY.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for _family in MAIN:
    _history = HISTORIES[model_name(_family, MAIN_K)]
    axes[0].plot(_history["epoch"], _history["strength_unit"], color=PALETTE[_family], marker=MARKERS[_family], label=_family)
    axes[1].plot(_history["epoch"], _history["strength_typical"], color=PALETTE[_family], marker=MARKERS[_family], label=_family)
axes[0].set_title("‖decoded change‖ per unit latent change")
axes[1].set_title("‖decoded change‖ per typical activation s_k")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.25)
    ax.legend()
savefig(fig, "fig7_intervention_strength.png")

# %% [markdown]
# ## 8. Sparsity
#
# "Produces consistently sparse latents": the table gives L0 (active latents per image) with
# its spread across images and the dead-latent fraction on the test set, and the figure shows
# the distribution of non-zero activation values and of per-image L0 for each main model. For
# the ConvAE the units are the `d·8·8` bottleneck activations, so L0 is also given as a fraction
# of the available units.

# %%
"""Sparsity table and activation histograms."""

SPARSITY_ROWS, ACTIVATIONS, L0_PER_IMAGE = [], {}, {}
for _family, _model in MAIN.items():
    _metrics, _l0 = evaluate(_model, TEST_X)
    _per_image = _l0.numpy()
    with torch.no_grad():
        _z = batched(_model.encode, TEST_X[: CFG.hist_images])
    _values = _z[_z > 0].cpu().numpy()
    ACTIVATIONS[_family], L0_PER_IMAGE[_family] = _values, _per_image
    SPARSITY_ROWS.append({"model": _family, "units": _z[0].numel(), "l0_mean": float(_per_image.mean()),
                          "l0_std_across_images": float(_per_image.std()), "l0_cv": float(_per_image.std() / max(_per_image.mean(), 1e-12)),
                          "l0_frac": _metrics["l0_frac"], "dead_frac": _metrics["dead_frac"],
                          "activation_median": float(np.median(_values)) if len(_values) else float("nan"), "paper": NOT_REPORTED})
SPARSITY = pd.DataFrame(SPARSITY_ROWS)
SPARSITY.to_csv(result_path("sparsity.csv"), index=False)
print("Sparsity on the test set (paper: 'consistently sparse latents', no numbers in the abstract)")
print(SPARSITY.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for _family in MAIN:
    if len(ACTIVATIONS[_family]):
        axes[0].hist(ACTIVATIONS[_family], bins=50, histtype="step", lw=1.5, color=PALETTE[_family], label=_family, density=True)
    axes[1].hist(L0_PER_IMAGE[_family], bins=40, histtype="step", lw=1.5, color=PALETTE[_family], label=_family, density=True)
axes[0].set_yscale("log")
axes[0].set_xlabel("non-zero activation value")
axes[0].set_ylabel("density")
axes[0].set_title("activation values")
axes[1].set_xlabel("active latents per image (L0)")
axes[1].set_title("per-image L0")
for ax in axes:
    ax.grid(alpha=0.25)
    ax.legend()
savefig(fig, "fig8_sparsity.png")

# %% [markdown]
# ## 9. Controllable editing
#
# The framework "enables linear, spatially localised interventions for controllable image
# editing". With the main Tensor-SAE, each row of the figure takes a test image, shows its
# reconstruction, then removes the image's strongest atom (its latent set to zero), and then
# adds a spatially localised atom (the lowest-entropy atom among the frequently used ones) at
# `α = 1` and `α = 2` typical activations. The last column shows the added atom. Because the
# decoder is linear, the edit is exactly `α s_k a_k` added to the reconstruction, and its
# support is the product of the row and column factors, so it touches only a rectangle of
# pixels.

# %%
"""Editing demo with the main Tensor-SAE."""

_model = MAIN["Tensor-SAE"]
with torch.no_grad():
    _freq = activation_frequency(_model, TEST_X)
    _frequent = _freq.argsort(descending=True)[: max(8, CFG.n_top_atoms)]
    _entropy = entropy_bits(spatial_energy(_model.atoms()))
    _local = int(_frequent[_entropy[_frequent].argmin()])
    _scale = typical_activation(_model, batched(_model.encode, EVAL_X))
    _x = TEST_X[:4]
    _z = _model.encode(_x)
    _recon = clamp01(_model.decode(_z))
    _removed = _z.clone()
    _strongest = _z.argmax(1)
    _removed[torch.arange(4), _strongest] = 0.0
    _panels = [_x, _recon, clamp01(_model.decode(_removed)),
               clamp01(_model.decode(_model.add_to_latent(_z, _local, 1.0 * _scale[_local]))),
               clamp01(_model.decode(_model.add_to_latent(_z, _local, 2.0 * _scale[_local])))]
    _atom_img = atom_image(_model.atoms()[_local])
_titles = ["input", "reconstruction", "strongest atom removed", f"+ atom {_local} (α = 1)", f"+ atom {_local} (α = 2)", f"atom {_local}"]
fig, axes = plt.subplots(4, 6, figsize=(10.5, 7.2))
for r in range(4):
    for c, panel in enumerate(_panels):
        axes[r, c].imshow(panel[r].permute(1, 2, 0).cpu().numpy())
    axes[r, 5].imshow(_atom_img)
    for c in range(6):
        axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(_titles[c], fontsize=8)
fig.suptitle(f"Controllable editing with Tensor-SAE (K = {MAIN_K}); atom {_local} has spatial entropy {float(_entropy[_local]):.2f} bits", fontsize=10)
savefig(fig, "fig9_editing_demo.png")

# %% [markdown]
# ## 10. Summary table
#
# One row per claim of the abstract, with the paper's number where the abstract gives one. The
# only number in the abstract is the Tensor-SAE intervention R² of about 0.93; every other cell
# in the paper column says so.

# %%
"""Summary of the main models against the abstract's claims."""

_eff = EFFICIENCY.set_index("model")
_lin = LINEARITY.set_index("model")
_stab = STABILITY[STABILITY["measure"] == "strength_typical"].set_index("model")
_ent = ENTROPY_TABLE.set_index("model")
_col = COLOUR_TABLE.set_index("model")
SUMMARY_ROWS = []
for _family in MAIN:
    _name = model_name(_family, MAIN_K)
    SUMMARY_ROWS.append({
        "model": _family, "latents": int(_eff.loc[_name, "n_latents"]), "params": int(_eff.loc[_name, "params"]),
        "test PSNR (dB)": _eff.loc[_name, "psnr"], "L0": _eff.loc[_name, "l0"], "dead frac": _eff.loc[_name, "dead_frac"],
        "R2 direct": _lin.loc[_family, "direct_r2"], "R2 round-trip": _lin.loc[_family, "round_trip_r2"],
        "strength CV": _stab.loc[_family, "coefficient_of_variation"],
        "spatial entropy (bits)": _ent.loc[_family, "entropy_median_bits"] if _family in _ent.index else float("nan"),
        "colour angle (deg)": _col.loc[_family, "angle_to_nearest_axis_mean_deg"] if _family in _col.index else float("nan"),
    })
SUMMARY = pd.DataFrame(SUMMARY_ROWS)
SUMMARY.to_csv(result_path("summary.csv"), index=False)
print(f"Summary of the main models (K = {MAIN_K}, {CFG.run_mode} mode)")
print(SUMMARY.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nPaper (abstract): Tensor-SAE intervention R² ≈ {PAPER['tensor_r2']}; all other quantities: {NOT_REPORTED}.")
if CFG.run_mode == "smoke":
    print("Smoke mode: these numbers come from synthetic images and tiny dictionaries and say nothing about CIFAR-10.")

if SAVE_TO_DRIVE:
    from google.colab import drive  # type: ignore[import-not-found]
    import shutil

    drive.mount("/content/drive")
    target = f"/content/drive/MyDrive/tensor-sae-results"
    shutil.copytree(CFG.results_dir, target, dirs_exist_ok=True)
    print(f"results mirrored to {target}")
