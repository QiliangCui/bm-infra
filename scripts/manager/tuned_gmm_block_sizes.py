# Copyright 2026 Google LLC
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
"""Auto-tuned block sizes for GMM kernel."""

# Key:
#   - m: int, total number of tokens/rows
#   - k: int, input feature dimension
#   - n: int, output feature dimension per group
#   - num_total_groups: int, total experts in the model
#   - num_current_groups: int, experts assigned to this TPU shard
#   - lhs_dtype: str, data type of the LHS matrix
#   - rhs_dtype: str, data type of the RHS (weights) matrix
#   - quant_block_size: int, granularity of quantization scales
# Value:
#   - tm: int, m-dimension tile size
#   - tk: int, k-dimension tile size
#   - tn: int, n-dimension tile size

TUNED_BLOCK_SIZES = {
    # Gate/Up Projections (K=6144, N=5120) for Qwen3-Coder-480B
    (128, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (128, 2048, 1280),
    (256, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (128, 2048, 1280),
    (512, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (128, 2048, 1280),
    (1024, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (128, 2048, 1280),
    (2048, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (256, 2048, 1280),
    (4096, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (512, 2048, 1280),
    (8192, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (512, 2048, 1280),
    (16384, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (512, 2048, 1280),
    (32768, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (512, 2048, 1280),
    (65536, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (512, 2048, 1280),

    # Down Projections (K=2560, N=6144) for Qwen3-Coder-480B
    (128, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (128, 1280, 2048),
    (256, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (128, 1280, 2048),
    (512, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (128, 1280, 2048),
    (1024, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (128, 1280, 2048),
    (2048, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (256, 1280, 2048),
    (4096, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (512, 1280, 2048),
    (8192, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (512, 1280, 2048),
    (16384, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (512, 1280, 2048),
    (32768, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (512, 1280, 2048),
    (65536, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (512, 1280, 2048),
}

def get_tuned_tiling(
    m: int,
    k: int,
    n: int,
    num_total_groups: int,
    num_current_groups: int,
    lhs_dtype: str,
    rhs_dtype: str,
    quant_block_size: int,
):
    """Returns the optimized (tm, tk, tn) tiling for a given GMM configuration."""
    key = (m, k, n, num_total_groups, num_current_groups, str(lhs_dtype),
           str(rhs_dtype), quant_block_size)
    
    # Return the tuned tiling if available, otherwise return None to use defaults
    return TUNED_BLOCK_SIZES.get(key, None)