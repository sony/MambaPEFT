__version__ = "1.1.1"

from mamba_ssm_1p1p1.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm_1p1p1.modules.mamba_simple import Mamba
from mamba_ssm_1p1p1.models.mixer_seq_simple import MambaLMHeadModel
