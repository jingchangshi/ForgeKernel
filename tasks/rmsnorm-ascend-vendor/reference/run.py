"""Vendor RMSNorm timing reference; correctness is checked by the task judge."""

import torch_npu


def run(x, w):
    return torch_npu.npu_rms_norm(x, w, 1e-6)[0]
