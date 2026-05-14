# Nightly Build Failure Log

## 2026-05-13

### Plain `nightly build` ([#17411](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17411)) — FAILED

**Pass / fail summary**: 927 passed, 17 failed, 15 waiting_failed (out of 960). After filtering 9 downstream "Record …" / "TPU Test Notification" reporters, **8 real root-cause failures** across six buckets.

**Root cause categories:**

- **CompilationConfig pydantic ValidationError (enable_qk_norm_rope_fusion)** (2 jobs): unchanged from yesterday's [#17288](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288) — vllm upstream LKG still has `pass_config.enable_qk_norm_rope_fusion` as non-Optional `bool` while tpu-inference still passes `None`. Both TPU generations affected on every variant today.

  | Test | Short error |
  |---|---|
  | [tpu7x Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17411#019e2023-3332-4322-82c4-8afeda2f29d1) | `pydantic_core._pydantic_core.ValidationError: 1 validation error for CompilationConfig — pass_config.enable_qk_norm_rope_fusion Input should be a valid boolean [type=bool_type, input_value=None, input_type=NoneType]` |
  | [tpu6e Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17411#019e2023-1856-47b5-a3e4-58f056326359) | Same `ValidationError` from `vllm/entrypoints/llm.py:307` `_make_config(CompilationConfig)` during `LLM(**engine_args)`. |

- **Accuracy collapse to 0.0** (1 job): lm_eval finishes cleanly but gsm8k score is 0.0 vs expected 0.41 — **fourth consecutive nightly** ([#17118](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17118), [#17166](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166), [#17288](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288), now [#17411](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17411)).

  | Test | Short error |
  |---|---|
  | [tpu7x Accuracy for google/gemma-4-26B-A4B-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17411#019e2023-328f-470e-ac90-53dc1eb33fc3) | `AssertionError: Expected: 0.41 \| Measured: 0.0` / `assert np.float64(0.0) >= (0.41 - 0.03)` (gsm8k, RTOL 0.03) |

- **Performance threshold failure (Kimi-K2.6 throughput)** (1 job): same Kimi-K2.6 throughput shortfall as the last two nightlies. Still under the 1500 tok/s bar.

  | Test | Short error |
  |---|---|
  | [tpu7x Unit tests for moonshotai/Kimi-K2.6](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17411#019e2023-32d3-4a92-ad9d-fd15e66cf044) | `Total token throughput: 1304.19` / `Total token throughput comparison (>= 1500): FAILED` |

- **Docker registry 504 (python:3.12-slim-bookworm)** (2 jobs): docker.io returned `504 Gateway Time-out` on the base image manifest fetch; the agent retried for ~30s and gave up. Same fingerprint as yesterday's [#17288](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288), but flaked on two different tpu6e steps this time.

  | Test | Short error |
  |---|---|
  | [tpu6e Correctness tests for SP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17411#019e2023-17d0-4d98-a248-6c6f4dc56663) | `ERROR: failed to solve: python:3.12-slim-bookworm: ... unexpected status code https://registry-1.docker.io/v2/library/python/blobs/sha256:905b...: 504 Gateway Time-out` |
  | [tpu6e Correctness tests for structured_decoding](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17411#019e2023-1844-44dc-ba82-3f84b4bdef48) | `ERROR: failed to solve: python:3.12-slim-bookworm: ... manifests/sha256:d193...: 504 Gateway Time-out` |

- **[NEW CATEGORY] Docker pip-install IncompleteRead (download.pytorch.org)** (1 job): the docker build step `pip install -r requirements/tpu.txt --extra-index-url https://download.pytorch.org/whl/cpu` died mid-download. Pure pip-vendored urllib3 transport flake, no tpu-inference code involved.

  | Test | Short error |
  |---|---|
  | [tpu6e E2E MLPerf tests for JAX + vLLM models on multiple chips](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17411#019e2022-d126-454a-8413-f1a8092fb2e1) | `pip._vendor.urllib3.exceptions.ProtocolError: ('Connection broken: IncompleteRead(216863552 bytes read, 10678430 more expected)', IncompleteRead(...))` (Dockerfile:61 `RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/tpu.txt --retries 3`) |

- **[NEW CATEGORY] P/D disagg EngineCore death on first request (Qwen3-0.6B)** (1 job): server starts cleanly, TPUConnector init succeeds, `/health` returns 200, then the very first benchmark request triggers `vllm.v1.engine.exceptions.EngineDeadError` from `AsyncLLM output_handler`. All 200 benchmark requests return 500 (request_throughput=0.00, failed=200). EngineCore subprocess stderr (the actual root cause) is not captured in the buildkite log; only `EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.` is visible. P/D disagg path is broken on `main` again, two nightlies after the [#17166](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166) `TPUConnector` 2-arg-signature fix.

  | Test | Short error |
  |---|---|
  | [tpu7x E2E test for Single Host DCN-based P/D disaggregation](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17411#019e2022-d699-4156-89da-0788b5d5f4dc) | `(APIServer pid=13) ERROR async_llm.py:704 AsyncLLM output_handler failed. ... vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.` → benchmark output: `Successful requests: 0 / Failed requests: 200 / Request throughput (req/s): 0.00` |

**Day-over-day vs [#17288](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288):**

| Category | Yesterday | Today | Delta |
|---|---|---|---|
| Performance threshold failure (DP single-host) | 2 | 0 | **−2 (FIXED)** — likely from PR #2598 [Fix nightly DP performance regression] that landed 2026-05-12 at [#17334](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17334) |
| Performance threshold failure (Kimi-K2.6 throughput) | 1 | 1 | = (third consecutive) |
| CompilationConfig pydantic ValidationError | 2 | 2 | = (unfixed; affects all three variants) |
| Accuracy collapse to 0.0 (gemma-4-26B) | 1 | 1 | = (fourth consecutive) |
| Docker registry 504 | 1 | 2 | +1 (different tpu6e jobs flaked; infra) |
| HF tokenizer vocab load failure | 1 | 0 | −1 (cleared, was likely cache-corruption flake) |
| **[NEW]** Docker pip IncompleteRead from download.pytorch.org | 0 | 1 | +1 (infra/network) |
| **[NEW]** P/D disagg EngineCore death on first request | 0 | 1 | +1 (functional regression — disagg broken again) |

### `nightly build with MODEL_IMPL_TYPE=vllm` ([#17418](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17418)) — FAILED

**Pass / fail summary**: 600 passed, 13 failed, 17 waiting_failed (out of 631). After filtering 7 downstream reporters, **6 real failures** across four buckets.

- **CompilationConfig pydantic ValidationError (enable_qk_norm_rope_fusion)** (2 jobs): same root cause as the plain variant — `pass_config.enable_qk_norm_rope_fusion` is no longer Optional in vllm LKG.

  | Test | Short error |
  |---|---|
  | [tpu7x Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17418#019e2075-6a2f-42db-8967-ae3eb11be4b3) | `pydantic_core._pydantic_core.ValidationError: 1 validation error for CompilationConfig — pass_config.enable_qk_norm_rope_fusion Input should be a valid boolean [type=bool_type, input_value=None]` |
  | [tpu6e Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17418#019e2075-4d80-444f-9766-4712ee04f3fc) | Same `ValidationError` from `vllm/entrypoints/llm.py:307`. |

- **[NEW CATEGORY] gemma-4 multimodal embed AttributeError ('bool' object has no attribute 'to')** (2 jobs): vllm-impl-only. `vllm/model_executor/models/gemma4_mm.py:1306` calls `is_multimodal.to(input_ids.device, non_blocking=True)` but `is_multimodal` arrives as a plain Python `bool` instead of a tensor — vllm-LKG regression in gemma-4 multimodal embedding when there are no images. Kills EngineCore at `embed_input_ids_func` during warmup.

  | Test | Short error |
  |---|---|
  | [tpu7x Accuracy for google/gemma-4-31B-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17418#019e2075-69df-4985-971d-9f371cf1d4af) | `(EngineCore pid=534) AttributeError: 'bool' object has no attribute 'to'` at `vllm/model_executor/models/gemma4_mm.py:1306 embed_input_ids` → `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}` |
  | [tpu6e Accuracy for google/gemma-4-31B-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17418#019e2075-4d13-4606-ba6a-93c05528b882) | Same `AttributeError: 'bool' object has no attribute 'to'` at `gemma4_mm.py:1306`. |

- **Unsupported activation function (`gelu_tanh`) in vllm impl** (1 job): same as #17173 / #17292 — gmm_v2 `apply_act_fn` registry missing `gelu_tanh` for the gemma-4 MoE path under vllm impl.

  | Test | Short error |
  |---|---|
  | [tpu7x Unit tests for google/gemma-4-26B-A4B-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17418#019e2075-69cb-4fcc-8c9a-e6aa6b139ff4) | `NotImplementedError: Unsupported activation function: gelu_tanh` (from `tpu_inference/kernels/megablox/gmm_v2.py:73 apply_act_fn`, EngineCore pid=531) |

- **Performance threshold failure (Kimi-K2.6 throughput)** (1 job): same Kimi throughput shortfall as plain.

  | Test | Short error |
  |---|---|
  | [tpu7x Unit tests for moonshotai/Kimi-K2.6](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17418#019e2075-69f7-4168-8943-eebd25c7128e) | `Total token throughput: 1305.95` / `Total token throughput comparison (>= 1500): FAILED` |

### `nightly build with MODEL_IMPL_TYPE=flax_nnx` ([#17440](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17440)) — FAILED

**Pass / fail summary**: 294 passed, 7 failed, 5 waiting_failed (out of 307). After filtering 4 downstream reporters, **3 real failures** across two buckets.

- **CompilationConfig pydantic ValidationError (enable_qk_norm_rope_fusion)** (2 jobs): same vllm-LKG regression — confirmed third nightly in a row affecting all three variants.

  | Test | Short error |
  |---|---|
  | [tpu7x Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17440#019e20c7-c619-44e5-84ff-f10ec8b0a8ec) | `pydantic_core._pydantic_core.ValidationError: 1 validation error for CompilationConfig — pass_config.enable_qk_norm_rope_fusion Input should be a valid boolean [type=bool_type, input_value=None]` |
  | [tpu6e Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17440#019e20c7-ac86-4a35-9885-c232cded5688) | Same `ValidationError` from `vllm/entrypoints/llm.py:307`. |

- **Performance threshold failure (Out-of-tree flax_nnx vs vllm gap)** (1 job): same fingerprint as yesterday's [#17292](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17292) (8.02% vs 8.00%); borderline by 0.02pp, almost certainly a flake.

  | Test | Short error |
  |---|---|
  | [tpu6e Performance tests for Out-of-tree model support](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17440#019e20c7-ac7e-45b3-b312-0dc9c91d7600) | `AssertionError: The performance difference between flax_nnx and vllm is too high. Difference: 8.02%, Threshold: 8.00%` / `assert 0.0801551389786684 < 0.08` |

---

## 2026-05-12

### Plain `nightly build` ([#17288](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288)) — FAILED

**Pass / fail summary**: 921 passed, 17 failed, 19 waiting_failed (out of 958). After filtering 9 downstream "Record …" reporters, **8 real root-cause failures** across five buckets.

**Root cause categories:**

- **Performance threshold failure** (3 jobs): DP single-host speedup is below the 1.05x bar on both TPU generations again; Kimi-K2.6 total token throughput is still below the 1500 tok/s bar. Same shape as yesterday's [#17166](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166) bucket — no movement.

  | Test | Short error |
  |---|---|
  | [tpu7x Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288#019e1afc-e41b-45f2-90f9-c28ba06157ba) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.97x` |
  | [tpu6e Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288#019e1afc-c32a-4451-bee9-483bdfbda995) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.96x` |
  | [tpu7x Unit tests for moonshotai/Kimi-K2.6](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288#019e1afc-e48e-4d11-af41-ddaa74998990) | `Total token throughput: 1324.65` / `Total token throughput comparison (>= 1500): FAILED` |

- **[NEW CATEGORY] CompilationConfig pydantic ValidationError (enable_qk_norm_rope_fusion)** (2 jobs): vllm upstream LKG flipped `pass_config.enable_qk_norm_rope_fusion` to a non-Optional `bool`, but tpu-inference still passes `None`, so every `LLM(**engine_args)` that goes through `_make_config` for `CompilationConfig` fails before construction. Breaks multimodal correctness on both TPU generations.

  | Test | Short error |
  |---|---|
  | [tpu7x Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288#019e1afc-e4d7-4396-aeb4-082194efbf88) | `pydantic_core._pydantic_core.ValidationError: 1 validation error for CompilationConfig — pass_config.enable_qk_norm_rope_fusion Input should be a valid boolean [type=bool_type, input_value=None, input_type=NoneType]` |
  | [tpu6e Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288#019e1afc-c37f-434d-9d6b-cc75ce74757f) | Same `ValidationError` raised from `vllm/entrypoints/llm.py:307` `_make_config(CompilationConfig)` during `LLM(**engine_args)`. |

- **Accuracy collapse to 0.0** (1 job): lm_eval completes 1319/1319 prompts cleanly but produces no correct answers. Unchanged from [#17166](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166) and [#17118](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17118) — third consecutive nightly with `measured=0.0 vs expected=0.41`.

  | Test | Short error |
  |---|---|
  | [tpu7x Accuracy for google/gemma-4-26B-A4B-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288#019e1afc-e43d-4fba-8346-6614dd3d8019) | `AssertionError: Expected: 0.41 \|  Measured: 0.0` (gsm8k, RTOL 0.03) |

- **[NEW CATEGORY] Docker registry 504 (python:3.12-slim-bookworm)** (1 job): docker.io returned `504 Gateway Time-out` while resolving the base image manifest; the agent retried for ~30s and gave up. Pure docker.io flake, no tpu-inference code involved.

  | Test | Short error |
  |---|---|
  | [tpu6e Unit tests for Qwen/Qwen3-4B](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288#019e1afc-c33b-4758-98a6-7277e383a490) | `ERROR: failed to solve: python:3.12-slim-bookworm: failed to resolve source metadata for docker.io/library/python:3.12-slim-bookworm: ... 504 Gateway Time-out` |

- **[NEW CATEGORY] HF tokenizer vocab load failure** (1 job): the gemma-3-27b-it tokenizer load raised `OSError: Unable to load vocabulary from file` inside transformers `_from_pretrained`, blocking `MultiModalBudget` construction. Likely a corrupted / partial HF cache on the tpu6e worker — not a code change.

  | Test | Short error |
  |---|---|
  | [tpu6e Unit tests for google/gemma-3-27b-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17288#019e1afc-c34a-4b97-8d0f-0a69509044b7) | `OSError: Unable to load vocabulary from file. Please check that the provided vocabulary is accessible and not corrupted.` (raised inside `vllm/model_executor/models/gemma3_mm.py:get_dummy_text`) |

**Day-over-day vs [#17166](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166):**

| Category | Yesterday | Today | Delta |
|---|---|---|---|
| Performance threshold failure | 3 | 3 | = |
| KV connector API mismatch (TPUConnector 2-arg signature) | 2 | 0 | -2 (fixed) |
| Accuracy collapse to 0.0 | 1 | 1 | = (third consecutive) |
| Server startup timeout | 1 | 0 | -1 |
| CompilationConfig pydantic ValidationError | 0 | 2 | +2 (new) |
| Docker registry 504 | 0 | 1 | +1 (new, infra) |
| HF tokenizer vocab load failure | 0 | 1 | +1 (new, infra) |

### `nightly build with MODEL_IMPL_TYPE=vllm` ([#17292](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17292)) — FAILED

**Pass / fail summary**: 590 passed, 19 failed, 13 waiting_failed (out of 623). After filtering 10 downstream reporters, **9 real failures** across four buckets.

- **Performance threshold failure** (5 jobs): DP single-host below 1.05x on both gens; Kimi-K2.6 throughput still under 1500 tok/s; gemma-4-31B-it perf benchmark request throughput collapsed to 0.62 req/s (bar is 1.8); Out-of-tree flax_nnx-vs-vllm gap clipped the 8% threshold by 0.02 pp.

  | Test | Short error |
  |---|---|
  | [tpu7x Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17292#019e1b4e-faa6-4d40-8884-46dfee975206) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.95x` |
  | [tpu6e Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17292#019e1b4e-df23-4112-b903-e6e46cf201fa) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.96x` |
  | [tpu7x Unit tests for moonshotai/Kimi-K2.6](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17292#019e1b4e-faf5-45eb-bc7c-c9b1439fa5b3) | `Total token throughput: 1334.09` / `Total token throughput comparison (>= 1500): FAILED` |
  | [tpu7x Performance benchmarks for google/gemma-4-31B-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17292#019e1b4e-fada-426c-b618-dea6087c0543) | `Request throughput: 0.62` / `Request throughput comparison (>= 1.8): FAILED` (token throughput 1606.17 tok/s) |
  | [tpu6e Performance tests for Out-of-tree model support](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17292#019e1b4e-dfdb-4c3d-b0ae-fbe362d4be3c) | `AssertionError: The performance difference between flax_nnx and vllm is too high. Difference: 8.02%, Threshold: 8.00%` |

- **[NEW CATEGORY] CompilationConfig pydantic ValidationError (enable_qk_norm_rope_fusion)** (2 jobs): same root cause as in the plain variant — `pass_config.enable_qk_norm_rope_fusion` is no longer Optional in vllm LKG.

  | Test | Short error |
  |---|---|
  | [tpu7x Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17292#019e1b4e-fb32-4cfe-8067-7ff84cdd63d3) | `pydantic_core._pydantic_core.ValidationError: 1 validation error for CompilationConfig — pass_config.enable_qk_norm_rope_fusion Input should be a valid boolean [type=bool_type, input_value=None]` |
  | [tpu6e Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17292#019e1b4e-dfe4-4fe4-96e1-fea125d63962) | Same `ValidationError` from `_make_config(CompilationConfig)` during `LLM(**engine_args)`. |

- **[NEW CATEGORY] Unsupported activation function (`gelu_tanh`) in vllm impl** (1 job): EngineCore init crashes the moment the model graph hits gemma-4's `gelu_tanh` MLP activation — the tpu-inference vllm-impl activation registry is missing it. Specific to `MODEL_IMPL_TYPE=vllm` (the plain build's Kimi/gemma jobs got past activation registration).

  | Test | Short error |
  |---|---|
  | [tpu7x Unit tests for google/gemma-4-26B-A4B-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17292#019e1b4e-fabf-4e60-8c36-7cd36f4fbf6e) | `NotImplementedError: Unsupported activation function: gelu_tanh` (EngineCore pid=531) |

- **[NEW CATEGORY] Docker registry 504 (python:3.12-slim-bookworm)** (1 job): same docker.io manifest 504 as the plain build, this time on the Qwen3-4B perf-bench step.

  | Test | Short error |
  |---|---|
  | [tpu6e Performance benchmarks for Qwen/Qwen3-4B](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17292#019e1b4e-df47-424d-a976-9cd025b89752) | `ERROR: failed to solve: python:3.12-slim-bookworm: ... 504 Gateway Time-out` |

### `nightly build with MODEL_IMPL_TYPE=flax_nnx` ([#17300](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17300)) — FAILED

**Pass / fail summary**: 280 passed, 13 failed, 5 waiting_failed (out of 299). After filtering 7 downstream reporters, **6 real failures** across three buckets.

- **Performance threshold failure** (2 jobs): DP single-host still under 1.05x on both gens.

  | Test | Short error |
  |---|---|
  | [tpu7x Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17300#019e1ba1-91b5-4237-a015-ddb375c792ce) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.96x` |
  | [tpu6e Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17300#019e1ba1-78d4-4625-8b5d-dfb377ab1c46) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.97x` |

- **[NEW CATEGORY] CompilationConfig pydantic ValidationError (enable_qk_norm_rope_fusion)** (2 jobs): same vllm-LKG regression — affects all three variants today.

  | Test | Short error |
  |---|---|
  | [tpu7x Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17300#019e1ba1-91f3-4f7c-a53c-7b310439db8d) | `pydantic_core._pydantic_core.ValidationError: 1 validation error for CompilationConfig — pass_config.enable_qk_norm_rope_fusion Input should be a valid boolean [type=bool_type, input_value=None]` |
  | [tpu6e Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17300#019e1ba1-790c-4b2f-81a3-2d2e29bee45a) | Same `ValidationError` from `vllm/entrypoints/llm.py:307`. |

- **Engine core init failure** (2 jobs): two different real root causes, both surfacing as `RuntimeError: Engine core initialization failed`. The tpu7x Runai test died on a leftover libtpu lockfile (TPU device-busy variant); the tpu6e Runai test died on an httpx GCS read-timeout while streaming the model from `gs://vertex-model-garden-public-us/llama3/llama3-8b-hf`.

  | Test | Short error |
  |---|---|
  | [tpu7x Correctness Test \| Runai Model Streamer Torchax RayDistributedExecutor](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17300#019e1ba1-91ec-499e-92f5-988c2f8a8d02) | `RuntimeError: Unable to initialize backend 'tpu': ABORTED: Internal error when accessing libtpu multi-process lockfile. Run "$ sudo rm /tmp/libtpu_lockfile".` |
  | [tpu6e Correctness Test \| Runai Model Streamer JAX UniProcExecutor](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17300#019e1ba1-7904-4ee1-9bdf-6825f031c5c2) | `httpx.ReadTimeout: The read operation timed out` (while loading `gs://vertex-model-garden-public-us/llama3/llama3-8b-hf` via runai_streamer) |

## 2026-05-11

### Plain `nightly build` ([#17166](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166)) — FAILED

**Pass / fail summary**: 936 passed, 14 failed, 8 waiting_failed (out of 959). After filtering 6 downstream "Record …" / "Notify …" / "TPU Test Notification" reporters, **7 real root-cause failures** across four buckets.

**Root cause categories:**

- **Performance threshold failure** (3 jobs): DP single-host speedup is below the 1.05x bar on both TPU generations; Kimi-K2.6 throughput is below the 1500 tok/s bar (yesterday's startup-timeout patch landed — Kimi now starts in time, and the throughput shortfall is the new gating issue).

  | Test | Short error |
  |---|---|
  | [tpu7x Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166#019e15d6-90bd-4508-853e-737427d8f035) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.94x` |
  | [tpu6e Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166#019e15d6-7226-42ad-bc02-5ac66804a7fe) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.93x` |
  | [tpu7x Unit tests for moonshotai/Kimi-K2.6](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166#019e15d6-9124-4edd-a8db-331f1dfba01d) | `Total token throughput: 1328.34 (>= 1500): FAILED` |

- **[NEW CATEGORY] KV connector API mismatch (TPUConnector 2-arg signature)** (2 jobs): vllm upstream LKG changed the `KVConnectorBase_V1` contract; tpu-inference's `TPUConnector` still uses the old 2-arg `super().__init__(vllm_config, role)` signature. P/D disagg E2E is fully broken on both single-host and multi-host.

  | Test | Short error |
  |---|---|
  | [tpu7x E2E test for Single Host DCN-based P/D disaggregation](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166#019e15d6-2d07-4e17-a0a3-eab9c43421e5) | `ValueError: Connector TPUConnector uses deprecated 2-argument constructor signature. External v1 KV connectors must accept kv_cache_config as the third constructor argument and pass it to super().__init__().` |
  | [tpu7x E2E test for Multihost DCN-based P/D disaggregation](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166#019e15d6-2cfc-44bf-a4ec-a569ae2b60fc) | Same `ValueError: Connector TPUConnector uses deprecated 2-argument constructor signature.` raised from `KVConnectorFactory.create_connector` during `_initialize_kv_caches`. |

- **Accuracy collapse to 0.0** (1 job): lm_eval completes 1319/1319 prompts cleanly but produces no correct answers. Unchanged from [#17118](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17118) — second nightly in a row with `measured=0.0 vs expected=0.41`.

  | Test | Short error |
  |---|---|
  | [tpu7x Accuracy for google/gemma-4-26B-A4B-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166#019e15d6-90dc-4b78-9600-881c5c0a1d1a) | `AssertionError: Expected: 0.41 \| Measured: 0.0` (gsm8k, RTOL 0.03) |

- **Server startup timeout** (1 job): a literal 1-second boundary race — `Application startup complete.` *was* logged at the very end of the job, but the wait wrapper polled just after the 960s deadline. Same job has flaked on perf-bench startup since [#16572](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/16572); budget was already bumped 720→960 s.

  | Test | Short error |
  |---|---|
  | [tpu7x Performance benchmarks for google/gemma-4-31B-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17166#019e15d6-9100-47e6-8b9e-e2c6678ba318) | `TIMEOUT: Waited 961 seconds (limit was 960). The string 'Application startup complete.' was NOT found.` |

**Day-over-day vs [#17118](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17118):**

| Category | Yesterday | Today | Delta |
|---|---|---|---|
| Performance threshold failure (DP single-host) | 2 jobs (tpu7x 1.01x, tpu6e 0.93x) | 2 jobs (tpu7x 0.94x, tpu6e 0.93x) | UNCHANGED — DP still 6–7% under threshold; tpu7x slightly regressed |
| Performance threshold failure (async scheduler) | 1 job (1.04x borderline) | — | CLEARED (likely flake — was just under threshold) |
| Server startup timeout (Kimi-K2.6) | 1 job (2753s/2750s) | — | FIXED — Kimi now starts in time (eval ran to completion) |
| Performance threshold failure (Kimi-K2.6 throughput) | — | 1 job (1328 vs ≥1500) | NEW — replaces yesterday's Kimi startup timeout (startup-timeout patch unmasked a throughput shortfall) |
| Server startup timeout (gemma-4-31B-it perf) | — | 1 job (961s/960s) | NEW (1-second overshoot; same fingerprint as the historical pre-05-10 gemma-4 perf timeouts) |
| Accuracy collapse to 0.0 (gemma-4-26B-A4B-it) | 1 job | 1 job | UNCHANGED — still `measured=0.0 vs expected=0.41` |
| **[NEW] KV connector API mismatch (TPUConnector 2-arg signature)** | — | 2 jobs | NEW — disagg P/D path broken by vllm LKG bump |

### `nightly build with MODEL_IMPL_TYPE=vllm` ([#17173](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17173)) — FAILED

**Pass / fail summary**: 594 passed, 15 failed, 13 waiting_failed (out of 623). 7 real failures after filtering.

- **Performance threshold failure** (4 jobs): same DP fingerprint as plain; plus gemma-4-31B-it throughput far below the bar and the same Kimi throughput shortfall as plain.

  | Test | Short error |
  |---|---|
  | [tpu7x Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17173#019e1628-9b46-4773-ba7a-aafc8f33fcde) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.95x` |
  | [tpu6e Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17173#019e1628-820e-40c4-827f-cafaa0e91cd8) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.96x` |
  | [tpu7x Performance benchmarks for google/gemma-4-31B-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17173#019e1628-9b62-4fc5-99b1-61cec93e0df3) | `Request throughput: 0.62, comparison (>= 1.8): FAILED` (server starts here; on the plain variant the same model hits a startup-timeout instead) |
  | [tpu7x Unit tests for moonshotai/Kimi-K2.6](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17173#019e1628-9b71-47f8-9471-2aa95cea5f03) | `Total token throughput: 1292.52 (>= 1500): FAILED` |

- **Unsupported activation function `gelu_tanh`** (1 job): vllm-impl-only `gelu_tanh` gap in the MoE GMM kernel's `apply_act_fn`. Same root cause as 05-10 [#17120](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17120).

  | Test | Short error |
  |---|---|
  | [tpu7x Unit tests for google/gemma-4-26B-A4B-it](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17173#019e1628-9b53-4a55-9492-f03228981ab5) | `NotImplementedError: Unsupported activation function: gelu_tanh` (from `tpu_inference/kernels/megablox/gmm_v2.py:73 apply_act_fn`) |

- **XLATensor / torch.Tensor mixed math in multimodal** (2 jobs): vllm-impl-specific; flax_nnx/default paths route the vision tower differently. Same as 05-10 [#17120](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17120); both `[False]` and `[True]` parametrizations fail.

  | Test | Short error |
  |---|---|
  | [tpu7x Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17173#019e1628-9b96-461f-b638-963dd46f35a1) | `AssertionError: Expect a Tensor or a View but got <class 'torch.Tensor'>; usually this means there is a mixed math between XLATensor and torch.Tensor` |
  | [tpu6e Correctness tests for Multimodal Inputs](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17173#019e1628-828a-4e2a-a585-a9be42b9615c) | Same `AssertionError: Expect a Tensor or a View but got <class 'torch.Tensor'>` — Qwen2.5-VL-3B-Instruct vision path. |

### `nightly build with MODEL_IMPL_TYPE=flax_nnx` ([#17182](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17182)) — FAILED

**Pass / fail summary**: 292 passed, 5 failed, 1 waiting_failed (out of 299). 2 real failures after filtering.

- **Performance threshold failure** (2 jobs): identical DP fingerprint to plain and vllm variants — confirms the DP regression is cross-impl, not vllm-specific (3 nightlies in a row).

  | Test | Short error |
  |---|---|
  | [tpu7x Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17182#019e167b-26d1-42a8-997c-3f7b7cbe7947) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.98x` |
  | [tpu6e Performance tests for DP (single-host)](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17182#019e167b-08fb-443c-948f-fae0ff88ff84) | `AssertionError: Data parallelism did not provide expected speedup (1.05x): 0.93x` |

---

## 2026-05-10

### Plain `nightly build` (build [#17118](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17118)) — FAILED

Branch: `main`, message: "nightly build". **939 passed, 11 failed, 7 waiting_failed** (out of 958). After filtering 5 downstream "Record …" reporters, **5 real root-cause failures** across three buckets. (The 7 `waiting_failed` are downstream steps that never ran due to upstream failures, plus one unfinished "Notify".)

**Root cause categories:**
- **Performance threshold failure** (3 jobs): `test_dp_performance` and `test_performance` (async scheduler) assert speedup ≥ 1.05x but measured below. tpu7x DP: 1.01x (baseline 10.93s / dp 10.86s — essentially equal). tpu6e DP: 0.93x (baseline 11.89s / dp 12.80s — still genuinely slower, ~7%). Async scheduler tpu7x: 1.04x (ref/async, borderline). Examples: `tpu7x Performance tests for DP (single-host)`, `tpu6e Performance tests for DP (single-host)`, `tpu7x Performance tests for async scheduler`.
- **[NEW CATEGORY] Accuracy collapse to 0.0** (1 job): `test_lm_eval_accuracy_v1_engine` for `google/gemma-4-26B-A4B-it` reports `measured accuracy: 0.0` against `expected: 0.41` (RTOL 0.03). The eval completes (1319/1319 prompts) — model loads and serves, but the lm_eval task produces no correct answers. This is qualitatively different from the prior gemma-4 startup timeouts (those never reached eval). Example: `tpu7x Accuracy for google/gemma-4-26B-A4B-it`.
- **Server startup timeout** (1 job): `moonshotai/Kimi-K2.6` Unit tests — `TIMEOUT: Waited 2753 seconds (limit was 2750). The string 'Application startup complete.' was NOT found.` 3 s over the limit; SentencePiece tokenizer extract from `tiktoken.model` also emitted a parse error earlier in the same log (model uses tiktoken, not SP — likely benign noise, but engine init never completed). Example: `tpu7x Unit tests for moonshotai/Kimi-K2.6`.

**Day-over-day vs 05-07 #16821:**

| Category | 05-07 (#16821) | 05-10 (#17118) | Delta |
|---|---|---|---|
| Pooling-path JAX compile-forbidden (`torchax.default_env` → `jax.random.key`) | 4 jobs | — | **FIXED** — almost certainly PR #2562 ("Nightly CI Fix: Stabilize Qwen3-Embedding with Sharding-Aware Pre-warming", merged 2026-05-08) |
| gemma-4 perf-benchmark server startup timeout | 2 jobs | — | **FIXED on plain variant** (gemma-4-31B-it perf still timing out on vllm variant — see #17120 below) |
| DP single-host perf below 1.05x | 2 jobs | 3 jobs | UNCHANGED + 1 (async scheduler joined; tpu7x DP improved 0.96x→1.01x but still under threshold) |
| [NEW] gemma-4-26B-A4B-it accuracy = 0.0 | — | 1 job | NEW — measured value is literally zero, suggests broken decoding/template/weights for this model |
| [NEW] Kimi-K2.6 startup timeout (2753s/2750s) | — | 1 job | NEW (3 s overshoot; bump timeout or shave init) |

### `nightly build with MODEL_IMPL_TYPE=vllm` (build [#17120](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17120)) — FAILED

592 passed, 17 failed, 13 waiting_failed (out of 623). 8 real failures after filtering downstream "Record …" reporters. Root causes:

- **Performance threshold failure** (4 jobs): tpu7x + tpu6e DP perf (same as plain), `tpu7x Performance benchmarks for google/gemma-4-31B-it` (`Request throughput comparison (>= 1.8): FAILED`), and `tpu6e Performance tests for Out-of-tree model support` (`flax_nnx vs vllm` perf diff 8.01% vs 8.00% threshold — 0.01% over).
- **[NEW CATEGORY] Unsupported activation function `gelu_tanh`** (1 job): `tpu7x Unit tests for google/gemma-4-26B-A4B-it` — `NotImplementedError: Unsupported activation function: gelu_tanh` from EngineCore init. The vllm impl path is missing a `gelu_tanh` activation registration that the flax_nnx/default path has.
- **[NEW CATEGORY] XLATensor / torch.Tensor mixed math in multimodal** (2 jobs): `tpu7x` + `tpu6e Correctness tests for Multimodal Inputs` (Qwen2.5-VL-3B-Instruct) — `AssertionError: Expect a Tensor or a View but got <class 'torch.Tensor'>; usually this means there is a mixed math between XLATensor and torch.Tensor` from `EngineCore`. vllm-impl-specific; flax_nnx/default paths route the vision tower differently.
- **Server startup timeout** (1 job): Kimi-K2.6 (same as plain).

### `nightly build with MODEL_IMPL_TYPE=flax_nnx` (build [#17124](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/17124)) — FAILED

292 passed, 5 failed, 1 waiting_failed (out of 299). 2 real failures after filtering:
- **Performance threshold failure** (2 jobs): same DP perf issue — `tpu7x Performance tests for DP (single-host)` and `tpu6e Performance tests for DP (single-host)`. The DP regression is cross-impl (not vllm-specific), confirming it's in the DP scheduling/comms path itself, not a model-impl layer.

**Action items (in priority order):**
- **gemma-4-26B accuracy=0.0 (plain)**: highest priority. The model serves but emits wrong tokens — likely a tokenizer/chat-template mismatch or weight load corruption. `git log --since=2026-05-08 -- tpu_inference/models/common/model_loader.py` and the gemma-4 model file.
- **gelu_tanh registration (vllm impl)**: add to the vllm-impl activation map; likely a small fix in the vllm-impl model registry.
- **XLATensor multimodal assertion (vllm impl)**: trace where vision_tower output crosses into the LM path under MODEL_IMPL_TYPE=vllm. Probably needs a `torch_view` / `torchax.tensor.Tensor` wrap somewhere.
- **DP perf regression**: 0.93x on tpu6e is real (8% slowdown). Still no fix landed since 05-06 #16692. The new tpu7x async-scheduler 1.04x is borderline-flake-y but worth watching.
- **Kimi-K2.6 timeout**: 3 s over — either bump to 3000 s or shave Kimi init.

---

## 2026-05-07 — Build [#16821](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/16821) — FAILED

Branch: `main`, message: "nightly build". 17 failed out of 958 — **8 root-cause test failures** in three categories. Other 9 failed = 8 "Record …" reporters + 1 "Notify test results". Same fingerprint as 05-06 #16692 (the regressions are 2 days old now); represents the steady state on `main` while no fixes have landed.

### Category 1: pooling-path JAX compile-forbidden — 4 jobs (NEW since 05-06)

Root cause: `RuntimeError: JAX compilation occurred but was forbidden in this context.` raised by `tpu_inference/runner/utils.py:177` (`ForbidCompile` wrapper around `pxla._cached_lowering_to_hlo`). Traceback bottom:

```
tpu_inference/runner/tpu_runner.py:921 _execute_model
  pooler_output = pooler_fn(...)
tpu_inference/models/common/model_loader.py:468 compute_pooler_output
  torch_states = torch_view(hidden_states)
torchax/interop.py:199 _torch_view → tensor.Tensor(t, torchax.default_env())
torchax/__init__.py:64 default_env → tensor.Environment()
torchax/tensor.py:367 __init__ → jax.random.key(torch.initial_seed() % (1<<63))
jax.random.key → threefry_seed → ForbidCompile.raise
```

Mechanism: `compute_pooler_output` calls `torch_view(hidden_states)` which lazily instantiates `torchax.default_env()` on first use. The env constructor calls `jax.random.key(...)` → `threefry_seed` which traces a JAX computation that was not in the precompile cache, so `VLLM_XLA_CHECK_RECOMPILATION=1` raises. This is identical to the Build 14383 / Qwen2.5-VL pattern (cache-miss in a path that wasn't covered by precompile coverage), but here in the embedding/pooler path. Likely fix: warm `torchax.default_env()` (and therefore `jax.random.key`) during precompile, or move the `default_env()` call out of the per-step path.

Affected (all `test_step_pooling.py::test_step_pooling_e2e` — the "Accuracy for Qwen/Qwen3-Embedding-8B" job actually runs the same test file as "Step Pooling (Embedding)"):
- tpu7x Accuracy for `Qwen/Qwen3-Embedding-8B`
- tpu7x Correctness tests for Step Pooling (Embedding)
- tpu6e Accuracy for `Qwen/Qwen3-Embedding-8B`
- tpu6e Correctness tests for Step Pooling (Embedding)

### Category 2: DP single-host performance regression — 2 jobs (NEW since 05-06)

`test_data_parallel.py::test_dp_performance` asserts `dp_time / baseline_time` ≥ 1.05x, but DP is now **slower** than baseline:
- tpu7x: baseline 11.62s, dp 12.08s → **0.96x** (need ≥1.05x)
- tpu6e: baseline 12.04s, dp 13.10s → **0.92x** (need ≥1.05x)

Both numbers are below 1.0, i.e. DP is genuinely slower than single-device on the same workload (2048 prompts). Not a borderline-flake — magnitude is well below threshold and reproduced on both architectures. Suggests a real DP-scheduling or comms regression in a recent commit.

Affected:
- tpu7x Performance tests for DP (single-host)
- tpu6e Performance tests for DP (single-host)

### Category 3: gemma-4 perf-benchmark server startup timeout — 2 jobs (ongoing since 05-05 #16572)

Same fingerprint as #16572: vLLM server never reaches `Application startup complete.` within the per-job budget. `gemma-4-26B-A4B-it` engine init alone consumes ~459 s of the 600 s timeout; `gemma-4-31B-it` is still mid-`Precompile worker0 backbone with embeds` at 720 s. Unchanged for 3 nightlies in a row (#16572, #16692, #16821) — no fix landed.

Affected:
- tpu7x Performance benchmarks for `google/gemma-4-26B-A4B-it`
- tpu7x Performance benchmarks for `google/gemma-4-31B-it`

### Day-over-day comparison (vs 05-05 #16572)

| Category | 05-05 (#16572) | 05-07 (#16821) | Status |
|---|---|---|---|
| gemma-4 perf-benchmark server startup timeout | 2 jobs | 2 jobs | UNCHANGED — no fix yet (3 nightlies) |
| Pooling-path JAX compile-forbidden (`torchax.default_env` → `jax.random.key`) | — | 4 jobs | NEW since 05-06 #16692 |
| DP single-host perf below 1.05x threshold | — | 2 jobs | NEW since 05-06 #16692 |
| (#16692 only) Llama4 MLperf E2E + spec-decoding E2E | — (not run) | — | Cleared / not surfaced this run |

Also noted: the previous recorded main nightly was #16572 (05-05). Intermediate #16692 (05-06) was not recorded — it had the same Cat 1 + Cat 2 + Cat 3 set as #16821 plus two extra E2E failures (Llama4 MLperf, speculative decoding) that did not repeat today.

Action items:
- **Cat 1 (highest priority — blocks 4 jobs across both archs):** find the change that introduced a JAX-compiling path through `torchax.default_env()` in the pooler hot path. Either (a) move `torchax.default_env()` warm-up into precompile, or (b) construct it eagerly during model init so the cache is hot before `VLLM_XLA_CHECK_RECOMPILATION` arms. `git log --since=2026-05-05 -- tpu_inference/models/common/model_loader.py tpu_inference/runner/tpu_runner.py` is the place to start.
- **Cat 2:** dig into recent changes to `test_data_parallel.py` infra and DP scheduling (`tpu_inference/runner` DP path). 0.92x is a 13% slowdown — too large to be variance.
- **Cat 3:** still no owner for the gemma-4 startup-timeout problem. Either bump the per-job timeout (cheap, but masks the symptom) or trim precompile coverage for these perf benchmarks.

---

## 2026-05-06 — Build [#16773](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/16773) — FAILED (release branch)

Branch: `releases/v0.20.0`, message: "nightly build". 6 failed out of 935 — only **2 real test failures**, both gemma-4 perf-benchmark server-startup timeouts. The other 4 failed are 2 "Record" reporters + 1 "Commit support matrices" + 1 broken "Publish vllm/vllm-tpu nightly image" (the broken publish is the standard release-branch pattern; same on prior release nightly #16597).

### Category 1: gemma-4 perf-benchmark server startup timeout — 2 jobs

Root cause: vLLM server log never shows `Application startup complete.` within the per-job timeout. Same fingerprint as main #16572 (05-05): for `gemma-4-26B-A4B-it` engine init alone took ~459 s (compilation 325 s) and the API server then ran out of the 600 s budget; for `gemma-4-31B-it`, the 720 s timeout fired while still on `Precompile worker0 backbone with embeds --> num_tokens=64`, well before warmup completes.

```
TIMEOUT: Waited 601 seconds (limit was 600). The string 'Application startup complete.' was NOT found.
TIMEOUT: Waited 720 seconds (limit was 720). The string 'Application startup complete.' was NOT found.
```

**Pre-existing on `main`, not a release-branch regression.** Confirmed against latest plain main nightly #16821 (2026-05-07) — both `tpu7x Performance benchmarks for google/gemma-4-26B-A4B-it` and `…gemma-4-31B-it` are also failing there. This is the same startup-timeout category first surfaced on main on 05-05 (#16572) when the `d_block_sizes` revert (PR #2490) unblocked the upstream gemma-4 unit tests. No fix has landed yet, so any release nightly that runs the perf step is expected to inherit it.

Affected:
- tpu7x Performance benchmarks for `google/gemma-4-26B-A4B-it` (timed out at 601 s)
- tpu7x Performance benchmarks for `google/gemma-4-31B-it` (timed out at 720 s)

### Day-over-day comparison (vs prior release nightly #16597, 2026-05-05)

| Category | 05-05 (#16597) | 05-06 (#16773) | Status |
|---|---|---|---|
| gemma-4 unit tests (`d_block_sizes` kwarg mismatch) | 3 jobs (tpu7x 26B+31B, tpu6e 31B) | — | Fixed by PR #2490 revert; need to verify cherry-pick into release branch (most likely already in) |
| async scheduler correctness | 1 job | — | Cleared / not run |
| gemma-4 perf-benchmark server startup timeout | (blocked upstream by unit tests) | 2 jobs | NEW surface on release branch — same as main, fix needed on `main` first then cherry-pick |
| Publish vllm/vllm-tpu nightly image (broken) | 1 | 1 | Stable / known |

Action items:
- Do **not** patch on release branch. Wait for fix on `main` (either timeout bump or precompile-coverage trim for perf benchmarks) and cherry-pick.
- Worth re-running #16773 once before declaring the timeout reproducible vs borderline-flaky — engine init for the 26B model is already 459 s of the 600 s budget, so any compile-time variance pushes it over.
- The broken "Publish vllm/vllm-tpu nightly image" job has been broken on both 05-05 and 05-06 release nightlies — confirm with infra whether the release-branch publish path is intentionally not wired up.

---

## 2026-05-05 — Build [#16572](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/16572) — FAILED

Branch: `main`, commit `58921051aaae`, message: "nightly build". 5 failed + 1 `waiting_failed` out of 936 — only **2 real test failures**, both gemma-4 perf-benchmark server-startup timeouts. Other 3 failed are 2 "Record" reporters + 1 "Notify test results"; the 1 `waiting_failed` is a downstream Record. Big improvement vs 05-04 (#16480, 3 root-cause regressions): the d_block_sizes regression that dominated 05-04 is gone, fixed by [PR #2490](https://github.com/vllm-project/tpu-inference/pull/2490) — Revert "Add env variable for overriding rpa block sizes" — landed in #16550 at 2026-05-05 01:51.

### Category 1: gemma-4 perf-benchmark server startup timeout — 2 jobs (NEW, only failures)

Root cause: vLLM server log never shows `Application startup complete.` within the per-job timeout. The server is mid-precompile when the timeout fires — engine init alone for gemma-4-26B-A4B took **457.92 s** (compilation: 325.63 s) before reaching the API server bringup, leaving only ~140 s of the 600 s budget for warmup. gemma-4-31B-it timed out at the 720 s limit while still on `Precompile worker0 backbone with embeds --> num_tokens=256`, well before warmup completes.

```
TIMEOUT: Waited 601 seconds (limit was 600). The string 'Application startup complete.' was NOT found.
TIMEOUT: Waited 720 seconds (limit was 720). The string 'Application startup complete.' was NOT found.
```

These tpu7x perf jobs were `waiting_failed` in #16480 (and earlier nightlies since #2338 merged on 05-01) because they depended on the gemma-4 unit tests that were failing on `d_block_sizes`. Now that the d_block_sizes revert (#2490) unblocked unit tests, the perf benchmarks actually ran for the first time in this window — and surfaced a startup-timeout problem. **Cannot tell from this build alone whether this is a new regression or a pre-existing slow-startup issue that has been masked all week** by the upstream unit-test failure.

Affected:
- tpu7x Performance benchmarks for `google/gemma-4-26B-A4B-it` (timed out at 601 s)
- tpu7x Performance benchmarks for `google/gemma-4-31B-it` (timed out at 720 s)

### Day-over-day comparison (vs 05-04 #16480)

| Category | 05-04 (#16480) | 05-05 (#16572) | Status |
|---|---|---|---|
| gemma-4 `ragged_paged_attention` kwarg mismatch (`d_block_sizes`) | 3 jobs | — | FIXED (PR #2490 revert landed in #16550) |
| gemma-4 perf-benchmark server startup timeout | (masked, `waiting_failed`) | 2 jobs | NEW surface — was hidden by the d_block_sizes failure upstream |

Action items:
- Decide if 600 s / 720 s startup timeouts are right for gemma-4 multimodal models given current precompile cost, or if precompile coverage should be trimmed for perf benchmarks. Engine init was 457.92 s for the 26B model — that's the long pole.
- Re-run #16572 once to see if the timeout is reproducible or borderline-flaky before declaring it a hard regression.

---

## 2026-05-04 — Build [#16480](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/16480) — FAILED

Branch: `main`, commit `e2a65826970e`, message: "nightly build". 7 failed + 13 `waiting_failed` out of 935 — only **3 real test failures**, all the same root cause (gemma-4 unit tests). The other 4 failed are 3 "Record" reporters + 1 "Notify test results"; the 13 `waiting_failed` are gemma-4 Accuracy/Performance jobs blocked by the unit-test failures. Big drop from 04-30 (#16172, 14 root-cause regressions): the `named_modules` regression that dominated 04-30 is gone, fixed by PRs [#2487](https://github.com/vllm-project/tpu-inference/pull/2487) (named_modules() shim on JaxModule) and [#2488](https://github.com/vllm-project/tpu-inference/pull/2488) (re-enable weight tracking now that it works).

### Category 1: gemma-4 `ragged_paged_attention()` kwarg mismatch — 3 jobs (NEW, dominant)

Root cause: `TypeError: ragged_paged_attention() got an unexpected keyword argument 'd_block_sizes'` thrown from `tpu_inference/layers/common/attention_interface.py:406` during `model_fn` execution → `EngineCore` death → `EngineDeadError` at the suite level.

PR [#2338](https://github.com/vllm-project/tpu-inference/pull/2338) "Add env variable for overriding rpa block sizes" (merged 2026-05-01 14:31) added a kwargs branch in `attention_interface.py:380-405`:
```python
d_block_sizes, p_block_sizes, m_block_sizes = None, None, None
if not use_hd64:
    d_block_sizes, p_block_sizes, m_block_sizes = get_env_block_sizes()
...
if not use_hd64:
    kwargs.update(d_block_sizes=..., p_block_sizes=..., m_block_sizes=...)
return func(*args, **kwargs)
```
…but the local kernel signature in `tpu_inference/kernels/ragged_paged_attention/v2/kernel.py:705` accepts only `num_kv_pages_per_block`, `num_queries_per_block`, `vmem_limit_bytes` — not the three `*_block_sizes` kwargs. Wherever `get_env_block_sizes()` is supposed to be plumbed, the local v2 kernel signature was not updated.

Gemma-4 is the **only** affected model because it's the only one routinely hitting `head_dim ≠ 64` — gemma-4 alternates `global_head_dim` and `head_dim` per layer (`tpu_inference/models/jax/gemma4.py:325-332`), at least one of which falls into the non-hd64 branch. All other tested models go through `ragged_paged_attention_hd64` which doesn't take these kwargs.

This regression has been present every nightly since #2338 merged (#16284 May-1, #16372 May-2, #16399 May-3, #16480 May-4) — it has not been fixed yet.

Affected:
- tpu7x Unit tests for `google/gemma-4-26B-A4B-it`
- tpu7x Unit tests for `google/gemma-4-31B-it`
- tpu6e Unit tests for `google/gemma-4-31B-it`

Downstream blocked (`waiting_failed`):
- tpu7x Accuracy / Performance benchmarks for both gemma-4 variants (4 jobs each ×2 = 8)
- tpu6e Accuracy / Performance benchmarks for gemma-4-31B-it (4 jobs)
- Record verified commit hashes (1 job)

### Day-over-day comparison (vs 04-30 #16172)

| Category | 04-30 (#16172) | 05-04 (#16480) | Status |
|---|---|---|---|
| vllm `track_weights_loading` (named_modules) | 14 jobs | — | FIXED (PR #2487 added shim, #2488 re-enabled tracking) |
| Multimodal similarity too low | 2 jobs | — | FIXED (PR #1947 revert + tp tweak landed at #16201) |
| rl_integration `TPUModelRunner.get_model` | 2 jobs | — | FIXED (PR #2460 landed at #16234) |
| Docker Hub 504 | 2 jobs | — | Transient — not observed today |
| gemma-4 `ragged_paged_attention` kwarg mismatch | — | 3 jobs | NEW (PR #2338 since May 1; not yet fixed) |

Note: this analysis covers only the latest completed nightly (#16480). Three nightlies in between (#16284 May-1, #16372 May-2, #16399 May-3) were not recorded — the same Category 1 regression is expected in those given the merge timing of #2338.

Action items:
- Fix `attention_interface.py` ↔ `kernels/ragged_paged_attention/v2/kernel.py` signature mismatch from #2338. Either (a) update the v2 kernel to accept the three `*_block_sizes` kwargs, or (b) translate them to the existing `num_kv_pages_per_block` / `num_queries_per_block` / `vmem_limit_bytes` at the call site, or (c) gate the kwargs branch on something more precise than `not use_hd64`. The hd64 vs non-hd64 dispatch likely also needs review since both kernels are local now.
- Confirm the same Category 1 failure pattern in #16284/#16372/#16399 if a backfill is needed.

---

## 2026-04-30 — Build [#16172](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/16172) — FAILED

Branch: `main`, message: "nightly build". 39 failed jobs out of 935 — 20 real test failures + 19 downstream "Record"/"Notify" reporters. Pipeline uploaded cleanly this time (no repeat of the 04-28 dup-key blocker).

### Category 1: vllm `track_weights_loading` AttributeError on flax_nnx models — 14 jobs (NEW, dominant)

Root cause: vllm's `default_loader.py:414` (`track_weights_loading`) iterates `model.named_modules()`, but tpu-inference flax_nnx model classes (`Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Gemma4ForConditionalGeneration`) inherit from `nnx.Module` which exposes `iter_modules()`, not the PyTorch `named_modules`. Manifests as `AttributeError: '<ModelClass>' object has no attribute 'named_modules'. Did you mean: 'iter_modules'?` during `EngineCore` init → suite-level `RuntimeError: Engine core initialization failed`.

PR [#2447](https://github.com/vllm-project/tpu-inference/pull/2447) "[Dont Review] Disable Weight Loading Tracking" landed at build #16162 (Apr 30 05:28) — 90 min before this nightly — and was *intended* to suppress this call. Either the disable knob doesn't cover the loader path the nightly exercises, or it was reverted; needs investigation. Note the same trace appears via both `/workspace/vllm` (LKG-pinned source) and `/usr/local/lib/python3.12/site-packages/vllm` (pip-installed) — confirming it's the upstream call site, not a path-specific issue.

Affected:
- tpu7x: Unit tests gemma-4-26B-A4B-it, Unit tests gemma-4-31B-it, Accuracy Qwen3-4B, Accuracy Qwen3-32B, KV Cache Offload, OOT perf, async scheduler (7 jobs)
- tpu6e: Unit tests gemma-4-31B-it, Accuracy Qwen3-4B, Accuracy Qwen3-32B, KV Cache Offload, OOT perf, async scheduler (6 jobs)
- pypi: Performance benchmarks for Qwen/Qwen3-4B (1 job)

### Category 2: Multimodal text similarity below threshold — 2 jobs (PERSISTS from 04-24)

`test_multi_modal_inference` on `Qwen/Qwen2.5-VL-3B-Instruct` — `AssertionError: Text similarity too low (0.28)`. Same failure pattern observed since 04-24 (#15626); PR [#1947](https://github.com/vllm-project/tpu-inference/pull/1947) revert + tp config tweaks landed at #16201 (Apr 30 14:50, *after* this nightly), so the next nightly should re-evaluate.

Affected:
- tpu7x Correctness tests for Multimodal Inputs
- tpu6e Correctness tests for Multimodal Inputs

### Category 3: rl_integration `TPUModelRunner.get_model` AttributeError — 2 jobs (NEW)

Root cause: `AttributeError: 'TPUModelRunner' object has no attribute 'get_model'`. vllm RL integration code path expects a `get_model()` method that tpu-inference's `TPUModelRunner` doesn't define. Likely surfaced by the same vllm bump as Category 1.

PR [#2460](https://github.com/vllm-project/tpu-inference/pull/2460) "[Bug fix] Fix RL Integration tests by setting vllm_config context" merged at #16234 (Apr 30 19:33, *after* this nightly) — should fix this category in the next nightly.

Affected:
- tpu7x Correctness tests for rl_integration
- tpu6e Correctness tests for rl_integration

### Category 4: Docker Hub registry 504 Gateway Time-out — 2 jobs (transient infra)

`failed to resolve source metadata for docker.io/library/python:3.12-slim-bookworm: 504 Gateway Time-out` while pulling/building the base image. Not a code bug — Docker Hub flake. Both happen during the early image-prep phase, before any test runs.

Affected:
- PyPI Test Performance benchmarks for meta-llama/Llama-3.1-8B-Instruct
- tpu6e Correctness Test | Runai Model Streamer Torchax UniProcExecutor

### Day-over-day comparison (vs 04-28 #15891)

| Category | 04-28 (#15891) | 04-30 (#16172) | Status |
|---|---|---|---|
| Pipeline upload rejected (dup key) | 1 job (425 waiting_failed) | — | FIXED |
| Kernel test API mismatch (`fused_gdn_kernel_test`) | 2 jobs | — | FIXED |
| vllm `track_weights_loading` (named_modules) | — | 14 jobs | NEW (dominant; #2447 disable didn't hold) |
| Multimodal similarity too low | not observable (v7x blocked) | 2 jobs | PERSISTS from 04-24 (fix coming via #1947 revert at #16201) |
| rl_integration `TPUModelRunner.get_model` | not observable | 2 jobs | NEW (fix landed at #16234, after this nightly) |
| Docker Hub 504 | — | 2 jobs | NEW (transient infra) |

Note: The 04-28 nightly (#15891) had the v7x pipeline upload blocked, so 425 v7x steps were `waiting_failed` and the long-running issues from 04-24 (DP correctness, gemma4-26B startup timeout, async-scheduler perf threshold) couldn't be observed. Today's nightly has clean v7x coverage but is dominated by the new `named_modules` regression. DP correctness, gemma4-26B startup timeout, and async-scheduler perf-threshold failures from 04-24 are all *absent* from today's failed list — they may have been quietly fixed, or the new failure pattern is shadowing them (e.g. async scheduler now fails on `named_modules` before reaching the perf assertion).

Action items:
- Verify why PR #2447's disable of `track_weights_loading` doesn't cover this loader path; either widen the disable or define a `named_modules` shim on the flax_nnx model base. Likely needs follow-up vllm PR or local override.
- Confirm Multimodal similarity recovers in next nightly (post #16201).
- Confirm rl_integration recovers in next nightly (post #16234).

---

## 2026-04-28 — Build [#15891](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/15891) — FAILED

Branch: `main`, message: "nightly build". Commit `99e0d153e48d`. Job states: 78 passed, 5 failed, **425 waiting_failed**, 1 unknown (out of 509). The v7x pipeline upload failed, so most v7x test steps never ran — coverage today is essentially v6e-only plus whatever ran before the v7x upload step.

### Category 1: Pipeline upload rejected — duplicate step key — 1 job (NEW, BLOCKER)

Root cause: `Pipeline upload rejected: The key "KV_Cache_Offload_CorrectnessTest" has already been used by another step in this build`. The v7x pipeline group failed to upload, blocking 425 dependent v7x test steps as `waiting_failed`. The v6e group uploaded successfully just before. Likely introduced by PR [#2390](https://github.com/vllm-project/tpu-inference/pull/2390) "[TPU KV Offloading] [Feat] KV cache offloading to host memory" merged 2026-04-28T04:56 (~2h before this nightly).

Affected:
- Upload Tests for Models & Features (auto)
- 425 v7x downstream steps (waiting_failed)

### Category 2: Kernel test API mismatch — 2 jobs (NEW)

Root cause: `TypeError: ragged_gated_delta_rule() missing 1 required positional argument: 'has_initial_state'` at `tests/kernels/fused_gdn_kernel_test.py:134`. The kernel's signature was changed to require `has_initial_state` but the test wasn't updated. Fails on the very first parameterized case (`test_basic0`) and stops the suite.

Affected:
- tpu7x JAX unit tests - kernels
- tpu6e JAX unit tests - kernels

### Category 3: TPU Test Notification — 2 jobs (downstream)

Notification steps that depend on the kernel job results — fail because the kernels job failed. Not independent failures.

Affected:
- tpu7x TPU Test Notification
- tpu6e TPU Test Notification

### Day-over-day comparison (vs 04-24 #15626)

| Category | 04-24 (#15626) | 04-28 (#15891) | Status |
|---|---|---|---|
| Multimodal similarity too low | 2 jobs | — | Not observable (v7x blocked; v6e MM job not in failed list) |
| Gemma4-26B server startup timeout | 1 job | — | Not observable (v7x blocked) |
| DP correctness | 2 jobs | — | Not observable (v7x blocked; v6e job not in failed list) |
| Async scheduler perf | 1 job | — | Not observable (v7x blocked) |
| Pipeline upload rejected (dup key) | — | 1 job | NEW (introduced by #2390) |
| Kernel test API mismatch (`fused_gdn_kernel_test`) | — | 2 jobs | NEW |

Note: because the v7x upload failed and 425 steps were `waiting_failed`, this build doesn't tell us whether the prior persistent failures (DP correctness, async perf, MM similarity, gemma4-26B timeout) have improved or regressed. Need to retry once the duplicate-key bug is fixed.

Action items:
- Dedupe `KV_Cache_Offload_CorrectnessTest` step key in the v7x pipeline yaml (likely from #2390).
- Update `tests/kernels/fused_gdn_kernel_test.py` to pass `has_initial_state` to `ragged_gated_delta_rule`.

---

## 2026-04-24 — Build [#15626](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/15626) — FAILED

Branch: `main`, message: "nightly build". Commit `9cf8e9fece7f`. 6 real test failures + 6 "Record" downstream + 2 notify.

### Category 1: Multimodal similarity score too low — 2 jobs (NEW pattern)

Root cause: `test_multi_modal_inference` on `Qwen/Qwen2.5-VL-3B-Instruct` — generated caption diverges too far from the expected string. `difflib.SequenceMatcher` ratio threshold is **0.85**, actual was far below. Model loads and runs, but output wording differs semantically (still plausible, just not the exact expected sentence).

- tpu7x: similarity **0.28** (`[False]`); `[True]` then crashed at `jax._src.xla_bridge._init_backend` — cascading EngineCore init failure because TPU wasn't released after the first test's assertion.
- tpu6e: similarity **0.17** (`[False]`).

Note: these same tests failed on 04-20 with the pydantic `fuse_minimax_qk_norm` ValidationError. PR [#2371](https://github.com/vllm-project/tpu-inference/pull/2371) "[CI] Fix Correctness tests for Multimodal Inputs" landed 04-23 and unblocked the tests, now exposing the underlying similarity issue.

Affected:
- tpu7x Correctness tests for Multimodal Inputs
- tpu6e Correctness tests for Multimodal Inputs

### Category 2: Gemma4-26B-A4B server startup timeout on tpu7x — 1 job (NEW)

Root cause: `vllm serve` for `google/gemma-4-26B-A4B-it` never emitted `Application startup complete.` within 600s. Log shows EngineCore finished `init engine (profile, create kv cache, warmup model)` at 463s and began compilation (`Enabled custom fusions: norm_quant, act_quant`), then the wait loop timed out at 601s. Likely needs a longer timeout (PR #2377 added per-model timeout env) or compile is regressing.

Affected:
- tpu7x Performance benchmarks for google/gemma-4-26B-A4B-it

### Category 3: DP correctness — text match rate too low — 2 jobs (PERSISTS)

`test_dp_correctness` text match rate below 60% threshold.
- tpu7x: **50.00%** (same as 04-20)
- tpu6e: **56.25%** (same as 04-20)

Borderline flaky — identical numbers day-over-day suggests it may actually be deterministic and genuinely regressed.

Affected:
- tpu7x Correctness tests for DP (single-host)
- tpu6e Correctness tests for DP (single-host)

### Category 4: Async scheduler perf threshold — 1 job (PERSISTS)

Speedup: **1.05x**, threshold: **1.10x**. Slightly worse than 04-20 (1.07x). Borderline.

Affected:
- tpu7x Performance tests for async scheduler

### Day-over-day comparison (vs 04-20 #15073)

| Category | 04-20 (#15073) | 04-24 (#15626) | Status |
|---|---|---|---|
| Multimodal pydantic ValidationError | 2 jobs | — | FIXED by #2371 |
| Multimodal similarity too low | — | 2 jobs | NEW (unmasked by #2371) |
| Gemma4-26B OOM on tpu6e | 2 jobs | — | Gone (tests not in plain nightly on main) |
| Gemma4-26B accuracy on tpu7x | 1 job | — | Gone |
| Gemma4-26B server startup timeout on tpu7x | — | 1 job | NEW |
| DP correctness (tpu7x 50% / tpu6e 56.25%) | 2 jobs | 2 jobs | PERSISTS (same numbers) |
| Async scheduler perf | 1.07x | 1.05x | PERSISTS (slight regression) |
| Infra (secrets/publish) | 2 jobs | — | Gone (main branch, not release) |

Note: 04-20 was a release-candidate build on `releases/v0.19.0`; 04-24 is the plain nightly on `main`, so coverage differs slightly. Previous plain nightly #15529 (04-23) had the same DP + MM + async failures plus a `gemma-4-31B-it` perf benchmark failure that's absent today.

---

## 2026-04-20 — Build [#15073](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/15073) — FAILED (releases/v0.19.0 release candidate)

Branch: `releases/v0.19.0`, message: "Test for release v0.19.0". 892 passed, 18 failed (8 real test failures + 8 "Record" downstream + 1 infra + 1 broken dependency).

### Category 1: Multimodal pydantic ValidationError — 2 jobs (NEW)

Root cause: `pydantic_core.ValidationError: pass_config.fuse_minimax_qk_norm — Input should be a valid boolean, got None`. The vllm LKG has a `CompilationConfig` field `fuse_minimax_qk_norm` that defaults to `None`, but pydantic expects `bool`. Crashes immediately at LLM init — no model loading attempted.

Affected:
- tpu7x Correctness tests for Multimodal Inputs
- tpu6e Correctness tests for Multimodal Inputs

### Category 2: Gemma4-26B OOM on tpu6e — 2 jobs (NEW)

Root cause: `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED` during `process_unquantized_moe_weights` — needs 1.38G but only 823M free on tpu6e. The gemma-4-26B-A4B (MoE) model doesn't fit on v6e chips during weight processing.

Affected:
- tpu6e Unit tests for google/gemma-4-26B-A4B-it
- tpu6e Performance benchmarks for google/gemma-4-31B-it (same OOM → engine core init failure)

### Category 3: Gemma4-26B accuracy failure on tpu7x — 1 job (NEW)

Measured accuracy: **0.417**, expected >= **0.53** (0.56 - 0.03 tolerance). Test `test_lm_eval_accuracy_v1_engine` for `google/gemma-4-26B-A4B-it`. The model loads and runs on tpu7x but produces low-quality outputs.

Affected:
- tpu7x Accuracy for google/gemma-4-26B-A4B-it

### Category 4: DP correctness — text match rate too low — 2 jobs

Root cause: `test_dp_correctness` text match rate below 60% threshold. tpu7x got **50.00%**, tpu6e got **56.25%**. Borderline flaky — these are close to threshold.

Affected:
- tpu7x Correctness tests for DP (single-host)
- tpu6e Correctness tests for DP (single-host)

### Category 5: Async scheduler perf threshold — 1 job

Speedup: **1.07x**, threshold: **1.10x**. Close miss, likely flaky/borderline.

Affected:
- tpu7x Performance tests for async scheduler

### Category 6: Infra — 2 jobs (not code bugs)

- **Commit support matrices**: `GITHUB_PAT` secret not found (404) — Buildkite secrets config issue for release branches.
- **Publish vllm/vllm-tpu nightly image**: broken (dependency on failed step).

### Release Blocking Assessment

| Category | Blocking? | Action needed |
|---|---|---|
| Multimodal pydantic error | **Yes** | Fix `fuse_minimax_qk_norm` default in vllm LKG or pin older CompilationConfig |
| Gemma4-26B OOM on tpu6e | **Maybe** | New model, may need to skip tpu6e or increase TP |
| Gemma4-26B accuracy on tpu7x | **Maybe** | New model, may need threshold tuning or model fix |
| DP correctness (50-56%) | **Flaky** | Borderline — rerun or lower threshold slightly |
| Async scheduler perf (1.07x) | **Flaky** | Borderline — rerun or lower threshold to 1.05x |
| Infra (secrets/publish) | **No** | Buildkite config, not code |

---

## 2026-04-16 — Build [#14708](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708) — FAILING (still running at time of analysis)

### Category 1: NEW — Gemma4 unsupported by transformers — 4 jobs

Root cause: `pydantic_core.ValidationError: The checkpoint you are trying to load has model type 'gemma4' but Transformers does not recognize this architecture.` vllm pins `transformers<5,>=4.56.0`; docker built with 4.57.6 which does not support `gemma4`. PR #2243 ([Gemma4] add gemma4 to support matrix, merged Apr 15) added these tests without updating the transformers pin or vllm LKG.

Affected:
- tpu7x Unit tests for google/gemma-4-26B-A4B-it
- tpu7x Unit tests for google/gemma-4-31B-it
- tpu6e Unit tests for google/gemma-4-26B-A4B-it
- tpu6e Unit tests for google/gemma-4-31B-it

### Category 2: Performance threshold failures — 3 jobs (OOT severely regressed)

- Out-of-tree model support (`test_flax_nnx_vs_vllm_performance`): gap = **55.90%**, threshold = 8%. **Severe regression from 6.80% yesterday.**
- tpu7x DP (`test_data_parallel.py::test_dp_performance`): speedup = **1.01x**, threshold = 1.05x. **NEW failure today.**
- tpu6e DP (`test_data_parallel.py::test_dp_performance`): speedup = **0.95x**, threshold = 1.05x. Persists from yesterday (was 0.96x).

### Category 3: Engine core initialization failed — 2 jobs (changed from device busy)

Root cause: `RuntimeError: Engine core initialization failed. See root cause above.` The tpu6e PP multi-host job shows `EngineCore` worker exited with code 1; multimodal shows empty `{}`. Both were "TPU device busy" on Apr 15 — today the device is reachable but the engine crashes during init. Need earlier log lines to determine root cause.

Affected:
- tpu6e Performance tests for PP (multi-host)
- tpu6e Correctness tests for Multimodal Inputs

### Category 4: Server startup timeout — 1 job (persists)

- tpu7x Performance benchmarks for Qwen/Qwen3.5-397B-A17B-FP8: server never bound port 8000 in 40 minutes. Same as Apr 15. Likely OOM or crash on model load.

### Category 5: SCHEDULED (stuck / not yet run) — 15+ jobs

Build started 07:00 UTC; the following tpu7x jobs remain in SCHEDULED state 5+ hours later. These are all the jobs that failed with `W8A8BlockFp8LinearOp ImportError` yesterday. Status unclear: either (a) queue backlog on `tpu_v7x_2_queue`, or (b) they'll run and may still hit the ImportError.

Stuck jobs include:
- [tpu7x lora unit tests on multi chips](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3976-47f6-bfcf-071448fe43d4)
- [tpu7x lora e2e tests for JAX + vLLM models multi chips](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3975-4a40-9e4b-63e77a6d8db9)
- [tpu7x E2E lm_eval accuracy check qwen3 coder with fused moe.](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3976-4e35-b23c-d7e8158bd10b)
- [tpu7x E2E lm_eval accuracy check qwen3 coder with gmm kernel.](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3976-4bc2-a502-0465e92ec394)
- [tpu7x E2E lm_eval accuracy check gpt oss.](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3977-44fc-9812-818b75d5c0fc)
- [tpu7x E2E lm_eval accuracy check Qwen3.5-397B-A17B with gmm kernel.](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-397d-42bf-af43-e78f0a87baa4)
- [tpu7x Perf regression test for qwen3 coder 1k 8k with fused moe kernel.](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3977-4afe-b322-62c77efaf41c)
- [tpu7x Perf regression test for qwen3 coder 8k 1k with fused moe kernel.](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3978-43b7-9ee5-1dd1fcc4be48)
- [tpu7x Perf regression test for qwen3 coder 1k 8k with gmm kernel.](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3978-40a3-868e-8d9c565af6c1)
- [tpu7x Perf regression test for qwen3 coder 8k 1k with gmm kernel.](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3978-47a9-b183-6368befc364f)
- [tpu7x Perf regression test for Qwen3.5-397B-A17B 1k 8k with gmm kernel.](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-397d-4920-a783-9ebbf00f1b6a)
- [tpu7x Perf regression test for Qwen3.5-397B-A17B 8k 1k with gmm kernel.](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-397e-4ab4-bf49-dcdd0cdce061)
- [tpu7x Test EP recompilation.](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3979-4c96-a494-8d2586720f51)
- [tpu7x E2E test for DCN-based P/D disaggregation](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3979-421d-9e2e-e4b3cff08203)
- [tpu7x Correctness Test | Runai Model Streamer Torchax RayDistributedExecutor](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-397c-4628-8eae-e4c1db1aa0c7)
- [tpu7x JAX unit tests - collective kernels](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3972-4e2a-869b-fe3c33749099)
- [tpu7x E2E MLperf tests for Llama4 models](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-396c-48bd-bd87-4f1ff5473cf9)
- [tpu7x E2E MLPerf tests for JAX + vLLM models on multiple chips](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14708#019d9517-3973-45fa-9657-678a04027ae2)

### Day-over-Day Comparison (Apr 15 → Apr 16)

| Category | Apr 15 | Apr 16 | Delta |
|---|---|---|---|
| ImportError W8A8BlockFp8LinearOp | 12 jobs failed | 15+ jobs SCHEDULED (not run) | Unclear if fixed; all stuck in queue |
| Gemma4 transformers incompatibility | — | 4 jobs NEW | From PR #2243 merged Apr 15 |
| tpu6e device busy | 4 jobs | 0 | Cleared |
| Engine core init failed | — | 2 jobs (tpu6e PP multi-host, Multimodal) | NEW; was device busy on Apr 15 |
| flax_nnx perf gap | 6.80% | **55.90%** | Severe regression |
| async scheduler speedup | 1.06x | SCHEDULED (not run) | TBD |
| tpu7x DP perf | (not listed) | 1.01x < 1.05x | NEW failure |
| tpu6e DP perf | 0.96x < 1.05x | 0.95x < 1.05x | Stable |
| Server startup timeout | 1 (Qwen3.5-FP8) | 1 (same) | Unchanged |

**Summary:** Three new issues today: (1) Gemma4 tests broken by missing transformers support — fix needed in transformers pin or vllm LKG; (2) OOT flax_nnx perf gap exploded from 7% to 56% — likely a real regression introduced today; (3) tpu6e engine core init failures replaced yesterday's device-busy noise. 15+ jobs are stuck in SCHEDULED — likely queue backlog from a saturated `tpu_v7x_2_queue`; outcome (pass or ImportError) unknown until they run.

---

## 2026-04-15 — Build [#14616](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14616) — FAILED

### Category 1: ImportError: W8A8BlockFp8LinearOp — 12 jobs (same root cause, persists)

Same `compressed_tensors_w8a8_fp8.py:27` import error as Apr 14. Unfixed.

Affected tests (tpu7x) — same set minus Qwen3.5-397B-A17B-FP8 perf benchmark (see Category 4):
- lora unit tests on multi chips
- lora e2e tests for JAX + vLLM models multi chips
- E2E lm_eval accuracy check — qwen3 coder with fused moe kernel
- E2E lm_eval accuracy check — qwen3 coder with gmm kernel
- E2E lm_eval accuracy check — gpt oss
- E2E lm_eval accuracy check — Qwen3.5-397B-A17B with gmm kernel
- Perf regression — qwen3 coder 8k 1k fused moe kernel
- Perf regression — qwen3 coder 8k 1k gmm kernel
- Perf regression — Qwen3.5-397B-A17B 8k 1k gmm kernel
- Test EP recompilation
- Correctness Test — Runai Model Streamer Torchax RayDistributedExecutor
- E2E test for DCN-based P/D disaggregation

### Category 2: tpu6e — TPU device busy — 3 jobs (persists, slightly different set)

Same infrastructure issue as Apr 14.

Affected tests (tpu6e):
- Performance tests for PP (single-host) — test_pipeline_parallel_configs[tp] (only [tp] failed; [jax_model], [vllm_model], [dp] passed)
- Performance tests for PP (multi-host) ← **NEW today** (not in Apr 14)
- Correctness tests for Multimodal Inputs
- (Speculative Decoding: Ngram no longer listed — may have passed or not scheduled)

### Category 3: Performance threshold failures — 3 jobs (persists, numbers changed)

- Out-of-tree model support (`test_flax_nnx_vs_vllm_performance`): gap = **6.80%**, threshold = 6%. **Improved significantly from 35.55% yesterday** but still over threshold. Very close; possibly borderline/flaky now.
- Async scheduler (`test_performance`): speedup = **1.06x**, threshold = 1.1x. **Slightly worse than yesterday (1.0955x)**. Both days borderline; test appears flaky.
- tpu6e DP (single-host) (`test_data_parallel.py::test_dp_performance`): speedup = **0.96x**, threshold = 1.05x. DP is actually slower than baseline — not a device busy issue as initially categorized.

### Category 4: Qwen3.5-397B-A17B-FP8 perf benchmark — server startup timeout — 1 job (NEW)

Root cause: vllm server never started — `Connection refused` on port 8000 for the full 40-minute timeout, then script exits 1. Yesterday this job failed with ImportError (Category 1); today it got past import but server failed to come up. Likely an OOM or model-load crash during 8-chip startup — need server log from earlier in the job to confirm.

Affected:
- Performance benchmarks for Qwen/Qwen3.5-397B-A17B-FP8 (tpu7x)

### Day-over-Day Comparison (Apr 14 → Apr 15)

| Category | Apr 14 | Apr 15 | Delta |
|---|---|---|---|
| ImportError W8A8BlockFp8LinearOp | 14 jobs | 12 jobs | -2 (Qwen3.5 perf split off to Cat 4) |
| tpu6e device busy | 4 jobs | 4 jobs (different set) | PP multi-host added, Speculative Decoding dropped |
| flax_nnx perf gap | 35.55% | 6.80% | Improved; still over 6% threshold |
| async scheduler speedup | 1.0955x | 1.06x | Slightly worse; both borderline |
| Server startup timeout | — | 1 new job | Qwen3.5-397B-A17B-FP8 perf benchmark |

**Summary:** The W8A8BlockFp8LinearOp import break is the dominant ongoing issue — blocking 12+ tests and now 2 days old. tpu6e device busy is persistent infra noise. The flax_nnx perf gap improved dramatically (35% → 7%) but is now a borderline threshold issue. A new failure appeared: Qwen3.5-397B-A17B-FP8 perf benchmark server timeout — warrants investigation of the server log.

---

## 2026-04-14 — Build [#14509](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/14509) — FAILED

### Category 1: ImportError: W8A8BlockFp8LinearOp — 14 jobs

Root cause: `W8A8BlockFp8LinearOp` removed/renamed from `vllm.model_executor.layers.quantization.utils.fp8_utils` in current LKG vllm commit. `tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:27` imports it eagerly at module load time, so any test that imports `tpu_inference` fails at collection/startup.

Affected tests (tpu7x):
- lora unit tests on multi chips
- lora e2e tests for JAX + vLLM models multi chips
- E2E lm_eval accuracy check — qwen3 coder with fused moe kernel
- E2E lm_eval accuracy check — qwen3 coder with gmm kernel
- E2E lm_eval accuracy check — gpt oss
- E2E lm_eval accuracy check — Qwen3.5-397B-A17B with gmm kernel
- Perf regression — qwen3 coder 8k 1k fused moe kernel
- Perf regression — qwen3 coder 8k 1k gmm kernel
- Perf regression — Qwen3.5-397B-A17B 8k 1k gmm kernel
- Test EP recompilation
- Correctness Test — Runai Model Streamer Torchax RayDistributedExecutor
- E2E test for DCN-based P/D disaggregation (benchmark ran, but benchmark client exits 1 due to plugin load failure)
- Performance benchmarks for Qwen3.5-397B-A17B-FP8

Fix: Update `compressed_tensors_w8a8_fp8.py:27` to use the new symbol name (or update vllm LKG).

### Category 2: tpu6e — TPU device busy — 4 jobs

Root cause: `FAILED_PRECONDITION: TPU initialization failed: open(/dev/vfio/N): Device or resource busy` — another process is holding the TPU device. Infrastructure/agent issue, not a code bug.

Affected tests (tpu6e):
- Performance tests for PP (single-host) — test_pipeline_parallel_configs[tp]
- Performance tests for DP (single-host)
- Correctness tests for Multimodal Inputs — test_multi_modal_inference[False], [True]
- Performance tests for Speculative Decoding: Ngram

### Category 3: Performance threshold failures — 2 jobs

- Out-of-tree model support (`test_model_loader.py::test_flax_nnx_vs_vllm_performance`): flax_nnx vs vllm throughput gap = **35.55%**, threshold = 6%. Large regression.
- Async scheduler (`test_async_scheduler.py::test_performance`): speedup = **1.0955x**, threshold = 1.1x. Borderline; likely flaky.
