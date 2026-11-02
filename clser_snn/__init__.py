"""Root package for the CLS-ER + SD2-SNN hybrid.

Two papers are combined here:

* **CLS-ER** (Arani et al., ICLR 2022) — the *learning scheme*: a working model
  plus two EMA "semantic memories" (stable / plastic) that mimic the
  hippocampal-neocortical complementary learning system, an episodic reservoir
  buffer, and a selector-based consistency loss on replayed logits.
* **SD^2-SNN** (Xie et al.) — the *substrate*: a spiking ResNet-18 trained with
  direct-encoding over T timesteps, plus optional structured weight
  decomposition, firing-rate sparsity and sparse-subspace memory.

CLS-ER's mechanisms are kept intact; SD^2-SNN contributes the spiking backbone
and (optionally) its three regularisers.
"""

from .config import HybridConfig
from .backbone import SpikingResNet18, build_backbone
from .buffer import ReservoirBuffer
from .calibration import CosineHead, LogitCalibrator, build_calibrator, install_cosine_head
from .memories import SemanticMemories
from .model import SpikingCLSER
from .ncm import NCMClassifier

__all__ = [
    "HybridConfig",
    "SpikingResNet18",
    "build_backbone",
    "ReservoirBuffer",
    "CosineHead",
    "LogitCalibrator",
    "build_calibrator",
    "install_cosine_head",
    "NCMClassifier",
    "SemanticMemories",
    "SpikingCLSER",
]
__version__ = "0.1.0"
