# This file is modified from: https://github.com/vllm-project/vllm
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
# @article{yang2025lserve,
#   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
#   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
#   year={2025}
# }

from typing import Any, Dict, List, Optional


class QServeQuantConfig:
    def __init__(
        self,
        weight_bits: int = 8,
    ) -> None:
        self.weight_bits = weight_bits

        if self.weight_bits not in [4, 8]:
            raise ValueError(
                "Currently, QServe only supports 4 or 8 bits for weight quantization, "
                f" but got {self.weight_bits}-bit weights here."
            )

    def __repr__(self) -> str:
        return f"QServeQuantConfig(weight_bits={self.weight_bits})"

    @classmethod
    def get_name(cls) -> str:
        return "qserve"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QServeQuantConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        return cls(weight_bits)

    @classmethod
    def get_packed_dim(cls, tensor_name: str) -> Optional[int]:
        return None

    @classmethod
    def is_transposed(cls, tensor_name: str) -> bool:
        return False

    @classmethod
    def get_col_parallel_tensor_names(cls) -> List[str]:
        return ["weight"]

    @classmethod
    def get_row_parallel_tensor_names(cls) -> List[str]:
        return ["weight"]


class NVFP4QuantConfig:
    """Config for the experimental NVFP4 (W4A8-FP) path (Blackwell-only).

    Weights are stored as E2M1 (NVFP4) with per-block E4M3 scales and a global
    FP32 scale. Accuracy is intentionally traded for efficiency in this
    experiment.
    """

    def __init__(self, block_size: int = 16) -> None:
        self.weight_bits = 4
        self.block_size = block_size

    def __repr__(self) -> str:
        return f"NVFP4QuantConfig(block_size={self.block_size})"

    @classmethod
    def get_name(cls) -> str:
        return "nvfp4"
