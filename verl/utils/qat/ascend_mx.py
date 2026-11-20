# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer-side export for Ascend ModelSlim microscaling rollouts.

The rollout engine is a vllm-ascend ``AscendModelSlimConfig``: the per-layer
scheme and the layers left in high precision come from the model directory's
``quant_model_description.json``, whose values are vllm-ascend's
``@register_scheme`` keys. This exporter reads that same file so the trainer
quantizes exactly what the engine was built to receive:

* ``W8A8_MXFP8`` -- ``float8_e4m3fn`` weights, one ``uint8`` scale per 32 input
  elements along the last dim.
* ``W4A8_MXFP``  -- ``uint8`` weights holding two fp4 (e2m1) values per byte
  (last dim halved), same 32-element ``uint8`` scales.

Both come from one NPU op, ``torch_npu.npu_dynamic_mx_quant``. The table below
is the only place that knows which destination dtype and which vllm-ascend
restore marker a scheme uses; adding a scheme is one entry.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import torch

logger = logging.getLogger(__name__)

MODELSLIM_CONFIG_FILENAME = "quant_model_description.json"

#: ModelSlim marks layers it leaves in high precision with this string.
UNQUANTIZED = "FLOAT"

#: mx microscaling block size along the quantized axis.
DEFAULT_GROUP_SIZE = 32


@dataclass(frozen=True)
class AscendMxScheme:
    """How one vllm-ascend microscaling scheme is produced and restored."""

    name: str
    #: ``torch_npu`` attribute passed as ``dst_type`` to ``npu_dynamic_mx_quant``.
    dst_type_attr: str
    #: Marker vllm-ascend sets once inference layout is applied; its presence
    #: means ``restore_weights_for_rl_loading`` must run before a reload.
    restore_attr: str
    #: Two 4-bit values share a byte, so the quantized last dim is halved.
    packed_fp4: bool


ASCEND_MX_SCHEMES: dict[str, AscendMxScheme] = {
    "W8A8_MXFP8": AscendMxScheme("W8A8_MXFP8", "float8_e4m3fn", "_mxfp8_transformed", False),
    "W4A8_MXFP": AscendMxScheme("W4A8_MXFP", "float4_e2m1fn_x2", "_mxfp4_transformed", True),
}

_HIGH_PRECISION_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

# ``model.layers.0.mlp.experts.0.gate_proj.weight`` -> ``model.layers.0.mlp.experts``.
_EXPERTS = re.compile(r"^(?P<base>.+\.mlp\.experts)(?:\..*)?$")


def scale_name(weight_name: str) -> str:
    """ModelSlim name of the scale that accompanies ``weight_name``."""
    return f"{weight_name}_scale"


def ascend_quantization_config(model_path: str | Path) -> dict[str, Any]:
    """vLLM ``quantization_config`` for a ModelSlim-prepared model directory.

    Only the method is carried: vllm-ascend's ``AscendModelSlimConfig`` reads the
    per-layer schemes from ``quant_model_description.json`` itself, and so does
    :class:`AscendMxWeightExporter`.

    This is a module-level function rather than a config-object method because
    the dispatch path receives a raw OmegaConf ``DictConfig``, on which method
    lookup raises ``ConfigAttributeError``.

    Raises:
        FileNotFoundError: The model directory is missing the description.
    """
    path = Path(model_path) / MODELSLIM_CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Ascend ModelSlim rollouts need the quantization description "
            "that vllm-ascend reads; point actor_rollout_ref.model.path at a ModelSlim-prepared "
            "directory that contains it."
        )
    return {"quant_method": "ascend"}


def _is_layer_entry(key: str) -> bool:
    """Whether a description key names a layer rather than metadata."""
    return "." in key


class AscendMxWeightExporter:
    """Quantize exported HF weights to match an Ascend ModelSlim description.

    A callable over ``(name, tensor)`` pairs, so it plugs into
    ``Engine.set_rollout_weight_quantizer``. Tensors the description marks
    unquantized -- or does not mention -- pass through unchanged. A tensor the
    description *does* mark quantized but whose shape or dtype cannot be
    quantized raises instead of silently shipping high-precision weights into a
    quantized engine.
    """

    def __init__(self, description: Mapping[str, Any], *, group_size: Optional[int] = None):
        self.description = dict(description)
        self.group_size = int(group_size or self.description.get("group_size") or DEFAULT_GROUP_SIZE)
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")

        unsupported = sorted(
            {
                value
                for key, value in self.description.items()
                if _is_layer_entry(key)
                and isinstance(value, str)
                and value != UNQUANTIZED
                and value not in ASCEND_MX_SCHEMES
            }
        )
        if unsupported:
            raise NotImplementedError(
                f"Unsupported Ascend ModelSlim quantization types {unsupported}; "
                f"this exporter handles {sorted(ASCEND_MX_SCHEMES)} and {UNQUANTIZED}."
            )

    @classmethod
    def from_model_path(cls, model_path: str | Path, **kwargs) -> "AscendMxWeightExporter":
        """Load the description the rollout engine will be built from.

        Args:
            model_path: Directory holding ``quant_model_description.json`` (the
                same directory the vLLM engine starts from).

        Raises:
            FileNotFoundError: The model directory is missing the description.
        """
        path = Path(model_path) / MODELSLIM_CONFIG_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Ascend ModelSlim rollouts need the quantization description "
                "that vllm-ascend reads; point actor_rollout_ref.model.path at a ModelSlim-prepared "
                "directory that contains it."
            )
        return cls(json.loads(path.read_text()), **kwargs)

    # ------------------------------------------------------------------ selection

    def _scheme_for(self, name: str) -> Optional[str]:
        """Scheme for a weight name, or None to pass it through unquantized."""
        candidates = [name]
        if name.endswith(".weight"):
            candidates.append(name[: -len(".weight")])
        experts = _EXPERTS.match(name)
        if experts is not None:
            # One bare entry per MoE layer covers every expert projection.
            candidates.extend((experts["base"], f"{experts['base']}.weight"))

        for candidate in candidates:
            value = self.description.get(candidate)
            if value is None:
                continue
            if value == UNQUANTIZED:
                return None
            if value not in ASCEND_MX_SCHEMES:
                raise NotImplementedError(f"Unsupported Ascend ModelSlim quant type {value!r} for {name}")
            return value
        return None

    # ---------------------------------------------------------------- quantization

    def _quantize(self, name: str, weight: torch.Tensor, scheme_name: str) -> Iterator[tuple[str, torch.Tensor]]:
        if weight.dim() != 2:
            raise ValueError(f"{name}: Ascend microscaling expects 2D weights, got shape {tuple(weight.shape)}")
        if weight.dtype not in _HIGH_PRECISION_DTYPES:
            raise ValueError(f"{name}: expected a float weight to quantize, got {weight.dtype}")

        import torch_npu

        scheme = ASCEND_MX_SCHEMES[scheme_name]
        dst_type = getattr(torch_npu, scheme.dst_type_attr, None) or getattr(torch, scheme.dst_type_attr)
        quantized, scale = torch_npu.npu_dynamic_mx_quant(weight.to(torch.bfloat16), axis=-1, dst_type=dst_type)

        groups = (weight.shape[-1] + self.group_size - 1) // self.group_size
        # The op may return the block scales shaped like the weight (optionally with
        # a trailing singleton dim); flatten every trailing axis into the group axis
        # so 2-D, 3-D and 4-D scale layouts all work.
        scale = scale.reshape(*weight.shape[:-1], -1)
        if scale.shape[-1] < groups:
            raise ValueError(f"{name}: mx quantizer returned {scale.shape[-1]} groups, expected {groups}")

        # vllm-ascend's W4A8 parameters are uint8. Some torch_npu builds return the
        # packed bytes as ``float4_e2m1fn_x2`` instead; reinterpret the same bytes so
        # the refit copy never depends on which of the two dtypes the op picked.
        if scheme.packed_fp4 and quantized.dtype != torch.uint8:
            quantized = quantized.view(torch.uint8)

        yield name, quantized.contiguous()
        yield scale_name(name), scale[..., :groups].contiguous()

    def __call__(self, weights) -> Iterator[tuple[str, torch.Tensor]]:
        from verl.utils.qat.compressed_tensors import _PACKED_EXPERTS, _unpack_experts

        for name, tensor in weights:
            packed = _PACKED_EXPERTS.match(name) if tensor.dim() == 3 else None
            scheme = self._scheme_for(name)
            if scheme is None:
                yield name, tensor
                continue
            if packed is not None:
                # Quantizing after the split is exact: scales are computed per row
                # along the last dim, which is untouched by splitting dim 0.
                for sub_name, sub_weight in _unpack_experts(name, tensor):
                    yield from self._quantize(sub_name, sub_weight, scheme)
            else:
                yield from self._quantize(name, tensor, scheme)
