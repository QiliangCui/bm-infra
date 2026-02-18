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
    (128, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 5,
    ),
    (128, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (128, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (128, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (128, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (256, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (256, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (256, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 5,
        256 * 24,
    ),
    (256, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 20,
    ),
    (256, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (512, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (512, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (512, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (512, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 5,
    ),
    (512, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 6,
    ),
    (1024, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (1024, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (1024, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 8,
    ),
    (1024, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 20,
    ),
    (1024, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (2048, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (2048, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (2048, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (2048, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 20,
    ),
    (2048, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (4096, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (4096, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (4096, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (4096, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 5,
    ),
    (4096, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (8192, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (8192, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (8192, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (8192, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 5,
    ),
    (8192, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (16384, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (16384, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (16384, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (16384, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 5,
    ),
    (16384, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 6,
    ),
    (32768, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (32768, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (32768, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (32768, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 5,
    ),
    (32768, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (65536, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (65536, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (65536, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (65536, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 5,
    ),
    (65536, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 6,
    ),
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